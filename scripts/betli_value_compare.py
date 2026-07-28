#!/usr/bin/env python3
"""Compare UltiNet eval head vs BetliNet on betli hand evaluation.

Generates biased betli hands (like the dojo), plays them out with the model,
and shows calibration tables for both evaluators side by side.

Usage:
    python scripts/betli_value_compare.py rook --games 500 --workers 4
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.betli.encoder import encode_hand
from trickster.betli.model import BetliNet, load_model as load_betli
from trickster.games.ulti.adapter import UltiGame
from trickster.games.ulti.cards import Card
from trickster.model import UltiNet, make_wrapper
from trickster.training.model_io import auto_device
from trickster.training.tiers import TIERS

# Reuse dojo infrastructure
from dojo import BetliDojo, _init_worker, _play_batch_in_worker, play_one_dojo_game
from trickster.games.ulti.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig


def _load_ultinet(source: str, device: str) -> tuple[UltiNet, object]:
    game = UltiGame(restrictions=[])
    model_pt = Path(f"models/ulti/{source}/final/betli/model.pt")
    if not model_pt.exists():
        print(f"Error: no model at {model_pt}")
        sys.exit(1)
    cp = torch.load(model_pt, weights_only=False, map_location=device)
    net = UltiNet(
        input_dim=cp.get("input_dim", game.state_dim),
        body_units=cp.get("body_units", 256),
        body_layers=cp.get("body_layers", 4),
        action_dim=cp.get("action_dim", game.action_space_size),
    )
    net.load_state_dict(cp["model_state_dict"], strict=False)
    net.to(device)
    net.eval()
    wrapper = make_wrapper(net, device=device)
    return net, wrapper


def _load_betlinet(source: str, device: str) -> BetliNet | None:
    path = Path(f"models/ulti/{source}/final/betli/betli_hand_eval.pt")
    if not path.exists():
        return None
    net = load_betli(path)
    net.to(device)
    net.eval()
    return net


def _betlinet_eval(betli_net: BetliNet, hand_10: list[Card], device: str) -> float:
    """Get BetliNet prediction for a 10-card hand."""
    feats = encode_hand(hand_10)
    with torch.no_grad():
        t = torch.from_numpy(feats).unsqueeze(0).to(device)
        return betli_net(t).item()


def _extract_record(
    samples, quality: float, sol_eval: float, sol_hand_10: list[Card],
    betli_net: BetliNet | None, device: str,
) -> dict:
    sol_reward = next(r for _, _, _, r, is_sol in samples if is_sol)
    won = sol_reward > 0
    betli_pred = None
    if betli_net is not None:
        betli_pred = _betlinet_eval(betli_net, sol_hand_10, device)
    return {
        "quality": quality,
        "ulti_eval": sol_eval,
        "betli_eval": betli_pred,
        "reward": sol_reward,
        "won": won,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare UltiNet vs BetliNet on betli hands")
    parser.add_argument("model", help="Model source (e.g. rook, trinity)")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--suit-sigma", type=float, default=1.0)
    parser.add_argument("--sol-sims", type=int, default=40)
    parser.add_argument("--sol-dets", type=int, default=2)
    parser.add_argument("--def-sims", type=int, default=20)
    parser.add_argument("--def-dets", type=int, default=2)
    parser.add_argument("--endgame-tricks", type=int, default=6)
    parser.add_argument("--pimc-dets", type=int, default=20)
    parser.add_argument("--solver-temp", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    source = args.model
    if args.device:
        device = args.device
    else:
        tier = TIERS.get(source)
        body_units = tier.body_units if tier else 256
        body_layers = tier.body_layers if tier else 4
        device = auto_device(body_units, body_layers)

    print(f"  Model: {source}")
    print(f"  Games: {args.games}")
    if args.workers > 1:
        print(f"  Workers: {args.workers}")
    print(f"  Device: {device}")
    print()

    # Load models
    net, wrapper = _load_ultinet(source, device)
    betli_net = _load_betlinet(source, device)
    if betli_net is None:
        print(f"  Warning: no BetliNet found for {source}, only showing UltiNet")
    else:
        print(f"  BetliNet: loaded")
    print()

    game = UltiGame(restrictions=[])
    sol_mcts = MCTSConfig(
        simulations=args.sol_sims, determinizations=args.sol_dets,
        use_value_head=True, use_policy_priors=True,
    )
    def_mcts = MCTSConfig(
        simulations=args.def_sims, determinizations=args.def_dets,
        use_value_head=True, use_policy_priors=True,
    )

    # Collect results
    records: list[dict] = []
    t0 = time.perf_counter()
    all_seeds = [args.seed + g for g in range(args.games)]

    if args.workers > 1:
        # --- Parallel via dojo worker pool ---
        from concurrent.futures import ProcessPoolExecutor

        body_layers_n = len([m for m in net.backbone if isinstance(m, torch.nn.Linear)])
        net_kwargs = {
            "input_dim": net.input_dim,
            "body_units": net.body_units,
            "body_layers": body_layers_n,
            "action_dim": net.action_dim,
        }
        all_weights = {k: v.cpu() for k, v in net.state_dict().items()}

        # Split seeds into batches per worker
        batches = []
        base = 0
        per_w = args.games // args.workers
        extra = args.games % args.workers
        for w in range(args.workers):
            n = per_w + (1 if w < extra else 0)
            if n == 0:
                continue
            batches.append((
                all_weights, sol_mcts, def_mcts,
                all_seeds[base:base + n],
                args.alpha, args.suit_sigma, False,  # kontra=False
                args.endgame_tricks, args.pimc_dets, args.solver_temp,
                "betli",
            ))
            base += n

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(net_kwargs,),
        ) as executor:
            for batch_results in executor.map(_play_batch_in_worker, batches):
                for samples, quality, sol_eval, sol_hand_10 in batch_results:
                    records.append(_extract_record(
                        samples, quality, sol_eval, sol_hand_10,
                        betli_net, device,
                    ))
                # Progress
                elapsed = time.perf_counter() - t0
                wins = sum(1 for r in records if r["won"])
                print(
                    f"\r  {len(records)}/{args.games}  "
                    f"win={wins/len(records):.0%}  "
                    f"{len(records)/elapsed:.1f} g/s  "
                    f"{elapsed:.0f}s",
                    end="", flush=True,
                )
    else:
        # --- Sequential ---
        dojo = BetliDojo()
        rng = random.Random(args.seed)
        sol_player = HybridPlayer(
            game, wrapper, mcts_config=sol_mcts,
            endgame_tricks=args.endgame_tricks,
            pimc_determinizations=args.pimc_dets,
            solver_temperature=args.solver_temp,
        )
        def_player = HybridPlayer(
            game, wrapper, mcts_config=def_mcts,
            endgame_tricks=args.endgame_tricks,
            pimc_determinizations=args.pimc_dets,
            solver_temperature=args.solver_temp,
        )

        for g in range(args.games):
            samples, quality, sol_eval, sol_hand_10 = play_one_dojo_game(
                game, wrapper, sol_player, def_player, rng,
                args.alpha, args.suit_sigma, kontra=False, dojo=dojo,
            )
            records.append(_extract_record(
                samples, quality, sol_eval, sol_hand_10,
                betli_net, device,
            ))

            if (g + 1) % 50 == 0 or g == args.games - 1:
                elapsed = time.perf_counter() - t0
                wins = sum(1 for r in records if r["won"])
                print(
                    f"\r  {g+1}/{args.games}  "
                    f"win={wins/(g+1):.0%}  "
                    f"{(g+1)/elapsed:.1f} g/s  "
                    f"{elapsed:.0f}s",
                    end="", flush=True,
                )

    print("\n")

    # Build calibration tables
    N_BINS = 8

    def _calibration_table(
        values: list[float], wins: list[bool], rewards: list[float], label: str,
    ) -> None:
        if not values:
            print(f"  {label}: no data")
            return

        arr = np.array(values)
        win_arr = np.array(wins)
        rew_arr = np.array(rewards)

        # Use quantile-based bins for even sample distribution
        edges = np.quantile(arr, np.linspace(0, 1, N_BINS + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            print(f"  {label}: all predictions identical ({arr[0]:.3f})")
            return

        print(f"  {label}")
        print(f"  {'Pred range':>16s}  {'N':>5s}  {'Win%':>5s}  {'Avg rew':>8s}  {'Avg pred':>9s}  {'Bar (win%)'}")
        print(f"  {'─'*16}  {'─'*5}  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*20}")

        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i < len(edges) - 2:
                mask = (arr >= lo) & (arr < hi)
            else:
                mask = (arr >= lo) & (arr <= hi)

            n = mask.sum()
            if n == 0:
                continue

            wr = win_arr[mask].mean()
            avg_rew = rew_arr[mask].mean()
            avg_pred = arr[mask].mean()
            bar_len = int(wr * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            print(f"  [{lo:+6.3f}, {hi:+6.3f})  {n:5d}  {wr:4.0%}  {avg_rew:+7.3f}  {avg_pred:+8.3f}   {bar}")

        corr_win = np.corrcoef(arr, win_arr.astype(float))[0, 1]
        corr_rew = np.corrcoef(arr, rew_arr)[0, 1]
        mse = float(np.mean((arr - rew_arr) ** 2))
        print(f"  Corr (pred vs win): {corr_win:+.3f}  |  Corr (pred vs reward): {corr_rew:+.3f}  |  MSE (pred vs reward): {mse:.3f}")
        print()

    # Prepare data
    ulti_vals = [r["ulti_eval"] for r in records]
    wins = [r["won"] for r in records]
    rewards = [r["reward"] for r in records]

    _calibration_table(ulti_vals, wins, rewards, "UltiNet eval head (value_fc_sol)")

    if betli_net is not None:
        betli_vals = [r["betli_eval"] for r in records]
        _calibration_table(betli_vals, wins, rewards, "BetliNet (specialized)")

    # Summary
    total_wins = sum(1 for r in records if r["won"])
    avg_reward = np.mean(rewards)
    print(f"  Overall: {total_wins}/{len(records)} = {total_wins/len(records):.1%} win rate, avg reward: {avg_reward:+.3f}")


if __name__ == "__main__":
    main()
