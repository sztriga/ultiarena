#!/usr/bin/env python3
"""Standalone Betli Engine — solver for play, specialized NN for evaluation.

Subcommands:
    generate  — Generate solver-labeled training data
    train     — Train BetliNet on generated data
    evaluate  — Compare betli engine vs old NN on same hands
    calibrate — Check calibration: predicted vs actual win rate

Usage:
    python3 scripts/betli_engine.py generate --samples 50000
    python3 scripts/betli_engine.py train --epochs 50
    python3 scripts/betli_engine.py evaluate --games 200
    python3 scripts/betli_engine.py calibrate
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
#  Generate
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    from trickster.betli.datagen import generate_samples

    print(f"\n  Generating {args.samples} solver-labeled betli samples")
    print(f"  PIMC dets: {args.dets}, alpha: {args.alpha}, seed: {args.seed}")
    print()

    t0 = time.perf_counter()
    features, labels = generate_samples(
        n_samples=args.samples,
        n_dets=args.dets,
        alpha=args.alpha,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=features, labels=labels)

    # Stats
    win_frac = (labels > 0).mean()
    print(f"\n  Done in {elapsed:.0f}s ({args.samples / elapsed:.0f} samples/s)")
    print(f"  Shape: features={features.shape}, labels={labels.shape}")
    print(f"  Label stats: mean={labels.mean():.3f}, std={labels.std():.3f}")
    print(f"  Win fraction (label > 0): {win_frac:.1%}")
    print(f"  Saved to {out}")


# ---------------------------------------------------------------------------
#  Train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    from trickster.betli.train import train

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"  ERROR: Data file not found: {data_path}")
        print(f"  Run 'generate' first.")
        sys.exit(1)

    data = np.load(data_path)
    features = data["features"]
    labels = data["labels"]

    print(f"\n  Training BetliNet")
    print(f"  Data: {features.shape[0]} samples from {data_path}")
    print(f"  Epochs: {args.epochs}, batch: {args.batch_size}, lr: {args.lr}")
    print()

    net = train(
        features,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.output,
    )

    # Quick sanity check
    import torch
    with torch.no_grad():
        preds = net(torch.from_numpy(features[:1000])).numpy()
    corr = np.corrcoef(preds, labels[:1000])[0, 1]
    print(f"\n  Correlation (first 1k): {corr:.3f}")


# ---------------------------------------------------------------------------
#  Evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    from trickster.betli.datagen import deal_betli, solver_label_fast
    from trickster.betli.player import BetliEngine
    from trickster.games.ulti.adapter import UltiGame
    from trickster.games.ulti.cards import BETLI_STRENGTH
    from trickster.games.ulti.game import (
        declare_all_marriages,
        is_terminal,
        next_player,
        play_card,
        set_contract,
    )
    from trickster.games.ulti.hybrid import HybridPlayer
    from trickster.mcts import MCTSConfig
    from trickster.model import UltiNetWrapper
    from trickster.training.model_io import load_wrappers

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"  ERROR: BetliNet not found: {model_path}")
        sys.exit(1)

    engine = BetliEngine(model_path, pimc_dets=args.dets)

    # Load old NN for comparison if available
    old_wrapper = None
    if args.baseline:
        wrappers = load_wrappers(args.baseline)
        old_wrapper = wrappers.get("betli")
        if old_wrapper:
            print(f"  Baseline: {args.baseline} betli model loaded")
        else:
            print(f"  WARNING: No betli model in {args.baseline}")

    game = UltiGame(restrictions=[])
    rng = random.Random(args.seed)
    n_games = args.games

    print(f"\n  Evaluating BetliNet engine vs solver ground truth")
    print(f"  Games: {n_games}, PIMC dets: {args.dets}")
    print()

    # Track results
    engine_wins = 0
    engine_correct = 0  # evaluation matches solver
    total = 0

    for i in range(n_games):
        hands, talon = deal_betli(rng, alpha=args.alpha)

        # Engine evaluation
        engine_val = engine.evaluate_hand(hands[0])

        # Solver ground truth
        gt = solver_label_fast(hands, n_dets=args.gt_dets, rng=rng)
        gt_label = 2.0 * gt - 1.0  # [0,1] → [-1,1]

        # Check sign agreement
        if (engine_val > 0) == (gt_label > 0):
            engine_correct += 1
        if gt > 0.5:
            engine_wins += 1
        total += 1

        if (i + 1) % 50 == 0:
            acc = engine_correct / total
            wr = engine_wins / total
            print(f"\r  {i + 1}/{n_games}  acc={acc:.1%}  solver_wr={wr:.1%}", end="", flush=True)

    print()
    print(f"\n  Results ({total} games):")
    print(f"  Engine accuracy (sign match): {engine_correct / total:.1%}")
    print(f"  Solver win rate: {engine_wins / total:.1%}")


# ---------------------------------------------------------------------------
#  Calibrate
# ---------------------------------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> None:
    from trickster.betli.datagen import deal_betli, solver_label_fast
    from trickster.betli.player import BetliEngine

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"  ERROR: BetliNet not found: {model_path}")
        sys.exit(1)

    engine = BetliEngine(model_path, pimc_dets=30)
    rng = random.Random(args.seed)

    print(f"\n  Calibration check: predicted value vs solver win rate")
    print(f"  Hands: {args.hands}, PIMC dets: {args.dets}")
    print()

    # Collect predictions and labels
    preds = []
    actuals = []

    for i in range(args.hands):
        hands, _ = deal_betli(rng, alpha=args.alpha)
        pred = engine.evaluate_hand(hands[0])
        actual = solver_label_fast(hands, n_dets=args.dets, rng=rng)
        preds.append(pred)
        actuals.append(actual)

        if (i + 1) % 100 == 0:
            print(f"\r  {i + 1}/{args.hands}", end="", flush=True)

    print()

    preds = np.array(preds)
    actuals = np.array(actuals)  # [0, 1] win fraction

    # Bin by predicted value
    n_bins = 10
    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)

    print(f"\n  {'Pred range':<15} {'Count':>6} {'Avg pred':>9} {'Avg actual':>11} {'Gap':>7}")
    print(f"  {'─' * 50}")

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (preds >= lo) & (preds < hi)
        if b == n_bins - 1:
            mask |= (preds >= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = preds[mask].mean()
        avg_actual = actuals[mask].mean()
        # Convert actual [0,1] to [-1,1] for comparison
        avg_actual_scaled = 2.0 * avg_actual - 1.0
        gap = avg_pred - avg_actual_scaled
        print(f"  [{lo:+.1f}, {hi:+.1f})  {count:>6}  {avg_pred:>+8.3f}  {avg_actual_scaled:>+10.3f}  {gap:>+6.3f}")

    corr = np.corrcoef(preds, actuals)[0, 1]
    print(f"\n  Overall correlation: {corr:.3f}")
    mse = ((preds - (2.0 * actuals - 1.0)) ** 2).mean()
    print(f"  MSE: {mse:.4f}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Betli Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate solver-labeled training data")
    p_gen.add_argument("--samples", type=int, default=50_000)
    p_gen.add_argument("--dets", type=int, default=30, help="PIMC determinizations per sample")
    p_gen.add_argument("--alpha", type=float, default=3.0, help="Dealing bias strength")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--output", default="data/betli_train.npz")

    # --- train ---
    p_train = sub.add_parser("train", help="Train BetliNet on generated data")
    p_train.add_argument("--data", default="data/betli_train.npz")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--output", default="models/betli/betli_eval.pt")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="Evaluate engine vs solver ground truth")
    p_eval.add_argument("--model", default="models/betli/betli_eval.pt")
    p_eval.add_argument("--baseline", default=None, help="Baseline model source (e.g. trinity)")
    p_eval.add_argument("--games", type=int, default=200)
    p_eval.add_argument("--dets", type=int, default=30, help="PIMC dets for engine play")
    p_eval.add_argument("--gt-dets", type=int, default=50, help="PIMC dets for ground truth")
    p_eval.add_argument("--alpha", type=float, default=3.0)
    p_eval.add_argument("--seed", type=int, default=42)

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="Check calibration: predicted vs actual")
    p_cal.add_argument("--model", default="models/betli/betli_eval.pt")
    p_cal.add_argument("--hands", type=int, default=500)
    p_cal.add_argument("--dets", type=int, default=50, help="PIMC dets for ground truth")
    p_cal.add_argument("--alpha", type=float, default=3.0)
    p_cal.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)


if __name__ == "__main__":
    main()
