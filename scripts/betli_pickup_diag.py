#!/usr/bin/env python3
"""Diagnose betli pickup decisions.

Simulates tournament-style random deals and shows which hands trigger
betli bids, what the eval predictions were, and whether they win.

Usage:
    python scripts/betli_pickup_diag.py rook --deals 2000 --workers 6
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

from trickster.betli.hand_evaluator import BetliHandEvaluator
from trickster.betli.model import load_model as load_betli
from trickster.bidding.auction_runner import run_auction, setup_bid_game
from trickster.games.ulti.adapter import UltiGame
from trickster.games.ulti.cards import make_deck
from trickster.games.ulti.game import GameState, next_player
from trickster.games.ulti.hybrid import HybridPlayer
from trickster.games.ulti.rewards import simple_outcome
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, make_wrapper
from trickster.training.model_io import auto_device, load_talon_prior
from trickster.training.tiers import TIERS


# ---------------------------------------------------------------------------
#  Worker globals
# ---------------------------------------------------------------------------

_W_GAME: UltiGame | None = None
_W_WRAPPERS: dict | None = None
_W_HAND_EVALUATORS: dict | None = None
_W_SOL_PLAYER: HybridPlayer | None = None
_W_DEF_PLAYER: HybridPlayer | None = None
_W_TALON_PRIOR: object | None = None


def _load_wrappers(source: str, device: str) -> dict:
    game = UltiGame()
    wrappers = {}
    for ck in ["parti", "betli", "ulti", "40-100", "durchmars"]:
        p = Path(f"models/ulti/{source}/final/{ck}/model.pt")
        if not p.exists():
            continue
        cp = torch.load(p, weights_only=False, map_location=device)
        net = UltiNet(
            input_dim=cp.get("input_dim", game.state_dim),
            body_units=cp.get("body_units", 256),
            body_layers=cp.get("body_layers", 4),
            action_dim=cp.get("action_dim", game.action_space_size),
        )
        net.load_state_dict(cp["model_state_dict"], strict=False)
        net.to(device)
        net.eval()
        wrappers[ck] = make_wrapper(net, device=device)
    return wrappers


def _load_hand_evaluators(source: str, device: str) -> dict | None:
    p = Path(f"models/ulti/{source}/final/betli/betli_hand_eval.pt")
    if not p.exists():
        return None
    net = load_betli(p)
    net.to(device)
    net.eval()
    return {"betli": BetliHandEvaluator(net, device=device)}


def _init_worker(source: str, endgame_tricks: int, pimc_dets: int, solver_temp: float) -> None:
    global _W_GAME, _W_WRAPPERS, _W_HAND_EVALUATORS, _W_SOL_PLAYER, _W_DEF_PLAYER, _W_TALON_PRIOR
    _W_GAME = UltiGame()
    _W_WRAPPERS = _load_wrappers(source, "cpu")
    _W_HAND_EVALUATORS = _load_hand_evaluators(source, "cpu")
    _W_TALON_PRIOR = load_talon_prior(source)

    betli_wrapper = _W_WRAPPERS.get("betli", list(_W_WRAPPERS.values())[0])
    sol_mcts = MCTSConfig(simulations=40, determinizations=2, use_value_head=True, use_policy_priors=True)
    def_mcts = MCTSConfig(simulations=20, determinizations=2, use_value_head=True, use_policy_priors=True)
    _W_SOL_PLAYER = HybridPlayer(
        _W_GAME, betli_wrapper, mcts_config=sol_mcts,
        endgame_tricks=endgame_tricks, pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )
    _W_DEF_PLAYER = HybridPlayer(
        _W_GAME, betli_wrapper, mcts_config=def_mcts,
        endgame_tricks=endgame_tricks, pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )


def _worker_fn(args: tuple) -> list[dict | None]:
    seeds, pickup_quantile, n_talon_samples, thresholds = args
    results = []
    for seed in seeds:
        results.append(_process_one_deal(seed, pickup_quantile, n_talon_samples, thresholds))
    return results


def _process_one_deal(
    seed: int, pickup_quantile: float, n_talon_samples: int,
    thresholds: dict[str, float] | None,
) -> dict | None:
    rng = random.Random(seed)
    deck = make_deck()
    rng.shuffle(deck)
    hands = [deck[:10], deck[10:20], deck[20:30]]
    talon = deck[30:32]
    dealer = seed % 3

    gs = GameState(
        hands=[list(h) for h in hands],
        trump=None, betli=False, soloist=0, dealer=dealer,
        captured=[[], [], []], scores=[0, 0, 0],
        leader=next_player(dealer), trick_no=0,
        trick_cards=[], last_trick=None,
    )

    seat_wrappers = [_W_WRAPPERS, _W_WRAPPERS, _W_WRAPPERS]
    result = run_auction(
        gs, talon, dealer, seat_wrappers,
        pickup_quantile=pickup_quantile,
        n_talon_samples=n_talon_samples,
        rng=rng,
        hand_evaluators=_W_HAND_EVALUATORS,
        pickup_thresholds=thresholds,
        talon_prior=_W_TALON_PRIOR,
    )

    if result.bid is None:
        return {"contract": "pass"}

    ck = result.bid.contract_key
    if ck != "betli":
        return {"contract": ck}

    # Play out betli
    soloist = result.soloist
    node = setup_bid_game(
        _W_GAME, gs, soloist, dealer, result.bid,
        initial_bidder=result.auction.history[0][0] if result.auction.history else -1,
    )

    state = node
    while not _W_GAME.is_terminal(state):
        player = _W_GAME.current_player(state)
        is_sol = (player == soloist)
        hp = _W_SOL_PLAYER if is_sol else _W_DEF_PLAYER
        _, action, _ = hp.choose_action_with_policy(state, player, rng)
        state = _W_GAME.apply(state, action)

    sol_reward = simple_outcome(state, soloist)

    # "Direct" = first bidder who bid on their first turn (saw 12 cards from deal)
    # "Overbidder" = anyone who picked up the talon (10→12 card path)
    soloist_picked_up = any(
        p == soloist and action == "pickup"
        for p, action, _ in result.auction.history
    )

    return {
        "contract": "betli",
        "seed": seed,
        "eval_pts": result.bid.game_pts,
        "best_discard_pts": result.bid.best_discard.game_pts,
        "best_discard_val": result.bid.best_discard.value,
        "reward": sol_reward,
        "won": sol_reward > 0,
        "is_piros": result.bid.is_piros,
        "is_direct": not soloist_picked_up,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose betli pickup decisions")
    parser.add_argument("model", help="Model source (e.g. rook)")
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pickup-quantile", type=float, default=0.5)
    parser.add_argument("--pickup-threshold", type=float, default=0.0)
    parser.add_argument("--n-talon-samples", type=int, default=20)
    parser.add_argument("--endgame-tricks", type=int, default=6)
    parser.add_argument("--pimc-dets", type=int, default=20)
    parser.add_argument("--solver-temp", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    source = args.model
    print(f"  Model: {source}")
    print(f"  Deals: {args.deals}")
    print(f"  Workers: {args.workers}")
    print(f"  Quantile: {args.pickup_quantile}")
    print(f"  Betli threshold: {args.pickup_threshold}")
    print()

    thresholds = {"betli": args.pickup_threshold} if args.pickup_threshold != 0.0 else None
    all_seeds = [args.seed + d for d in range(args.deals)]

    # Split into small chunks for smoother progress
    CHUNK_SIZE = 10  # deals per work unit
    chunks: list[tuple] = []
    base = 0
    while base < args.deals:
        n = min(CHUNK_SIZE, args.deals - base)
        chunks.append((
            all_seeds[base:base + n],
            args.pickup_quantile,
            args.n_talon_samples,
            thresholds,
        ))
        base += n

    all_results: list[dict] = []
    t0 = time.perf_counter()

    def _print_progress():
        elapsed = time.perf_counter() - t0
        betli_so_far = [r for r in all_results if r.get("contract") == "betli"]
        bw = sum(1 for b in betli_so_far if b.get("won"))
        nb = len(betli_so_far)
        print(
            f"\r  {len(all_results)}/{args.deals}  "
            f"betli={nb} wr={bw/max(nb,1):.0%}  "
            f"{len(all_results)/max(elapsed,0.1):.1f} d/s  "
            f"{elapsed:.0f}s",
            end="", flush=True,
        )

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(source, args.endgame_tricks, args.pimc_dets, args.solver_temp),
        ) as executor:
            for batch_results in executor.map(_worker_fn, chunks, chunksize=1):
                all_results.extend(batch_results)
                _print_progress()
    else:
        _init_worker(source, args.endgame_tricks, args.pimc_dets, args.solver_temp)
        for chunk in chunks:
            batch_results = _worker_fn(chunk)
            all_results.extend(batch_results)
            _print_progress()

    print("\n")

    # Separate results
    passes = [r for r in all_results if r.get("contract") == "pass"]
    betli_bids = [r for r in all_results if r.get("contract") == "betli"]
    other_bids = [r for r in all_results if r.get("contract") not in ("pass", "betli")]
    total_bids = len(betli_bids) + len(other_bids)

    if not betli_bids:
        print(f"  No betli bids in {args.deals} deals ({len(passes)} passes, {len(other_bids)} other bids).")
        return

    n = len(betli_bids)
    wins = sum(1 for b in betli_bids if b["won"])
    piros = sum(1 for b in betli_bids if b["is_piros"])

    print(f"  ┌─ BETLI PICKUP DIAGNOSTICS ──────────────────────")
    print(f"  │  Total deals: {args.deals}")
    print(f"  │  Passes: {len(passes)}  Other bids: {len(other_bids)}  Betli bids: {n}")
    print(f"  │  Betli rate: {n/args.deals:.1%} of deals, {n/max(total_bids,1):.0%} of bids")
    print(f"  │  Piros betli: {piros}")
    print(f"  │  Win rate: {wins}/{n} = {wins/n:.1%}")
    print(f"  │  Avg reward: {np.mean([b['reward'] for b in betli_bids]):+.3f}")
    print(f"  │")

    # Prediction distribution for wins vs losses
    win_preds = [b["best_discard_val"] for b in betli_bids if b["won"]]
    loss_preds = [b["best_discard_val"] for b in betli_bids if not b["won"]]

    if win_preds:
        print(f"  │  WINS ({len(win_preds)}):")
        print(f"  │    Pred val:  mean={np.mean(win_preds):+.3f}  "
              f"min={np.min(win_preds):+.3f}  max={np.max(win_preds):+.3f}")
        print(f"  │    Avg reward: {np.mean([b['reward'] for b in betli_bids if b['won']]):+.3f}")
    if loss_preds:
        print(f"  │  LOSSES ({len(loss_preds)}):")
        print(f"  │    Pred val:  mean={np.mean(loss_preds):+.3f}  "
              f"min={np.min(loss_preds):+.3f}  max={np.max(loss_preds):+.3f}")
        print(f"  │    Avg reward: {np.mean([b['reward'] for b in betli_bids if not b['won']]):+.3f}")

    print(f"  │")

    # Calibration: binned by prediction
    vals = np.array([b["best_discard_val"] for b in betli_bids])
    won_arr = np.array([b["won"] for b in betli_bids])
    rew_arr = np.array([b["reward"] for b in betli_bids])

    edges = np.quantile(vals, np.linspace(0, 1, 6))
    edges = np.unique(edges)

    if len(edges) >= 2:
        print(f"  │  Calibration (binned by best_discard prediction):")
        print(f"  │  {'Pred range':>16s}  {'N':>4s}  {'Win%':>5s}  {'Avg rew':>8s}")
        print(f"  │  {'─'*16}  {'─'*4}  {'─'*5}  {'─'*8}")
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (vals >= lo) & (vals <= hi) if i == len(edges) - 2 else (vals >= lo) & (vals < hi)
            cnt = mask.sum()
            if cnt == 0:
                continue
            wr = won_arr[mask].mean()
            ar = rew_arr[mask].mean()
            print(f"  │  [{lo:+6.3f}, {hi:+6.3f})  {cnt:4d}  {wr:4.0%}  {ar:+7.3f}")

    # Threshold sensitivity
    print(f"  │")
    print(f"  │  Threshold sensitivity (best_discard_val):")
    print(f"  │  {'Threshold':>10s}  {'Would bid':>9s}  {'Win%':>5s}  {'Avg rew':>8s}")
    print(f"  │  {'─'*10}  {'─'*9}  {'─'*5}  {'─'*8}")
    for thr in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = vals >= thr
        cnt = mask.sum()
        if cnt == 0:
            print(f"  │  {thr:>10.1f}  {cnt:>9d}     --       --")
            continue
        wr = won_arr[mask].mean()
        ar = rew_arr[mask].mean()
        print(f"  │  {thr:>10.1f}  {cnt:>9d}  {wr:4.0%}  {ar:+7.3f}")

    # Direct (12 cards from deal) vs Overbidder (10→12 pickup path)
    direct_bids = [b for b in betli_bids if b["is_direct"]]
    overbid_bids = [b for b in betli_bids if not b["is_direct"]]

    print(f"  │")
    print(f"  │  DIRECT (12 cards) vs OVERBIDDER (10+talon) breakdown:")
    for label, subset in [("Direct", direct_bids), ("Overbidder", overbid_bids)]:
        if not subset:
            print(f"  │  {label}: 0 bids")
            continue
        sw = sum(1 for b in subset if b["won"])
        sv = np.array([b["best_discard_val"] for b in subset])
        sr = np.array([b["reward"] for b in subset])
        print(f"  │")
        print(f"  │  {label}: {len(subset)} bids, "
              f"win={sw}/{len(subset)} ({sw/len(subset):.0%}), "
              f"avg rew={sr.mean():+.3f}")
        print(f"  │    Pred val: mean={sv.mean():+.3f}  "
              f"min={sv.min():+.3f}  max={sv.max():+.3f}")
        # Mini threshold sensitivity
        sw_arr = np.array([b["won"] for b in subset])
        print(f"  │    Threshold  Would bid  Win%  Avg rew")
        for thr in [0.0, 0.2, 0.4, 0.6, 0.8]:
            mask = sv >= thr
            cnt = mask.sum()
            if cnt == 0:
                continue
            print(f"  │    {thr:>9.1f}  {cnt:>9d}  {sw_arr[mask].mean():4.0%}  {sr[mask].mean():+7.3f}")

    # Worst-predicted betli bids that still made it to a game
    worst_pred = sorted(betli_bids, key=lambda b: b["best_discard_val"])
    if worst_pred:
        print(f"  │")
        print(f"  │  WORST PREDICTIONS (lowest pred that still played):")
        print(f"  │  {'Seed':>8s}  {'Reward':>7s}  {'Pred':>6s}  {'Type'}")
        print(f"  │  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*10}")
        for b in worst_pred[:10]:
            label = "direct" if b["is_direct"] else "overbid"
            won = "W" if b["won"] else "L"
            print(f"  │  {b['seed']:>8d}  {b['reward']:+6.3f}  {b['best_discard_val']:+5.3f}  {label} {won}")

    print(f"  └─────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
