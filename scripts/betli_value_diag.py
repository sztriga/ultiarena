#!/usr/bin/env python3
"""Betli value head vs solver diagnostic.

Deals biased betli hands, compares three signals:
  1. Value head prediction (trick-0, instant)
  2. Perfect-info solver verdict (trick-0, exact but slow on some hands)
  3. Neural self-play outcome (MCTS playout)

Usage:
    python3 scripts/betli_value_diag.py morpheus --games 200
    python3 scripts/betli_value_diag.py morpheus --games 500 --solver-timeout 5
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.bidding.evaluator import _make_eval_state
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import ALL_SUITS, BETLI_STRENGTH, make_deck
from trickster.games.ulti.game import (
    GameState,
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
)
from trickster.games.ulti.hybrid import HybridPlayer
from trickster.games.ulti.rewards import simple_outcome
from trickster.mcts import MCTSConfig
from trickster.training.model_io import load_wrappers

try:
    from trickster._solver_core import solve_root as _solve_root
except ImportError:
    from trickster.games.ulti.solver import solve_root as _solve_root


# -- Helpers ----------------------------------------------------------------

def _solve_worker(gs: GameState, result_queue: mp.Queue) -> None:
    """Run solver in a child process so we can hard-kill on timeout."""
    try:
        vals = _solve_root(gs, contract="betli")
        result_queue.put(max(vals.values()))
    except Exception:
        result_queue.put(None)


def solver_verdict(gs: GameState, timeout_s: float) -> float | None:
    """Perfect-info solver with process-based timeout (kills nogil Cython)."""
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_solve_worker, args=(gs, q))
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.kill()
        p.join()
        return None
    if not q.empty():
        return q.get_nowait()
    return None


def _weighted_sample(deck, weights, n, rng):
    items = list(deck)
    ws = list(weights)
    chosen = []
    for _ in range(n):
        total = sum(ws)
        r = rng.random() * total
        cum = 0.0
        for i, w in enumerate(ws):
            cum += w
            if cum >= r:
                chosen.append(items.pop(i))
                ws.pop(i)
                break
    return chosen, items


def deal_betli(rng, alpha, suit_sigma):
    deck = make_deck()
    a = rng.uniform(0, alpha)
    suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in ALL_SUITS}
    weights = [
        math.exp(-a * BETLI_STRENGTH[c.rank]) * suit_mult[c.suit]
        for c in deck
    ]
    sol_hand, rest = _weighted_sample(deck, weights, 10, rng)
    rng.shuffle(rest)
    hands = [sol_hand, rest[:10], rest[10:20]]
    talon = rest[20:22]
    return hands, talon


def hand_quality(hand):
    return sum(7 - BETLI_STRENGTH[c.rank] for c in hand) / 70.0


def main():
    parser = argparse.ArgumentParser(description="Betli value head vs solver diagnostic")
    parser.add_argument("model", help="Model source (e.g. morpheus)")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--suit-sigma", type=float, default=1.0)
    parser.add_argument("--solver-timeout", type=float, default=10.0,
                        help="Per-hand solver timeout in seconds (default 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    game = UltiGame(restrictions=[])
    cdef = CONTRACT_DEFS["betli"]

    print(f"Loading {args.model} betli model...")
    wrappers = load_wrappers(args.model, device="cpu")
    wrapper = wrappers["betli"]

    mcts_cfg = MCTSConfig(
        simulations=40, determinizations=2,
        use_value_head=True, use_policy_priors=True,
    )
    sol_player = HybridPlayer(
        game, wrapper, mcts_config=mcts_cfg,
        endgame_tricks=6, pimc_determinizations=20,
        solver_temperature=0.5,
    )
    def_player = HybridPlayer(
        game, wrapper, mcts_config=mcts_cfg,
        endgame_tricks=6, pimc_determinizations=20,
        solver_temperature=0.5,
    )

    # Collection
    data = []  # (quality, vhead_pred, solver_win, play_outcome)
    n_timeout = 0
    t_start = time.time()

    for i in range(args.games):
        hands, talon = deal_betli(rng, args.alpha, args.suit_sigma)
        dealer = 2
        soloist = 0
        q = hand_quality(hands[soloist])

        gs = GameState(
            hands=hands, trump=None, betli=False,
            soloist=soloist, dealer=dealer,
            captured=[[], [], []], scores=[0, 0, 0],
            leader=next_player(dealer),
            trick_no=0, trick_cards=[], last_trick=None,
        )

        # Pickup + smart discard via value head
        pickup_talon(gs, soloist, talon)
        best_val = -float("inf")
        best_pair = (gs.hands[soloist][0], gs.hands[soloist][1])
        for c1, c2 in combinations(gs.hands[soloist], 2):
            node = _make_eval_state(
                gs, soloist, trump=None, discards=(c1, c2),
                contract_def=cdef, is_piros=False, dealer=dealer,
            )
            feats = game.encode_state(node, soloist)
            val = wrapper.predict_value(feats)
            if val > best_val:
                best_val = val
                best_pair = (c1, c2)
        discard_talon(gs, list(best_pair))

        # Set contract
        set_contract(gs, soloist, trump=None, betli=True)
        gs.training_mode = "betli"
        declare_all_marriages(gs, soloist_marriage_restrict=None)

        # Value head prediction (trick-0)
        node = UltiNode(
            gs=gs,
            known_voids=(frozenset(), frozenset(), frozenset()),
            bid_rank=0, is_red=False,
            contract_components=frozenset({"betli"}),
            dealer=dealer,
        )
        feats = game.encode_state(node, soloist)
        vhead = wrapper.predict_value(feats)

        # Perfect-info solver verdict
        sv = solver_verdict(gs, args.solver_timeout)
        if sv is None:
            n_timeout += 1
            solver_win = None
        else:
            solver_win = sv > 5.0  # 10.0 = win, 0.0 = lose

        # Neural self-play
        state = node
        while not game.is_terminal(state):
            player = game.current_player(state)
            hp = sol_player if player == soloist else def_player
            _, action, _ = hp.choose_action_with_policy(state, player, rng)
            state = game.apply(state, action)
        outcome = simple_outcome(state, soloist)
        play_win = outcome > 0

        data.append((q, vhead, solver_win, play_win))

        if (i + 1) % 25 == 0:
            solved = sum(1 for _, _, s, _ in data if s is not None)
            wins_play = sum(1 for _, _, _, p in data if p)
            wins_solver = sum(1 for _, _, s, _ in data if s)
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{args.games}]  play_win={wins_play}/{i+1}  "
                  f"solver_win={wins_solver}/{solved}  "
                  f"timeout={n_timeout}  {elapsed:.0f}s")

    # -- Analysis --
    quals = np.array([d[0] for d in data])
    vheads = np.array([d[1] for d in data])
    play_wins = np.array([d[3] for d in data])
    play_outcomes = np.where(play_wins, 1.0, -1.0)

    # Solver mask (exclude timeouts)
    solver_mask = np.array([d[2] is not None for d in data])
    solver_wins = np.array([d[2] if d[2] is not None else False for d in data])

    elapsed = time.time() - t_start
    n_solved = solver_mask.sum()

    print(f"\n{'='*70}")
    print(f"  BETLI VALUE HEAD vs SOLVER — {args.model}")
    print(f"{'='*70}")
    print(f"  Games: {args.games}  Solver timeout: {args.solver_timeout}s  "
          f"Elapsed: {elapsed:.0f}s")
    print(f"  Solver completed: {n_solved}/{args.games}  "
          f"Timeouts: {n_timeout}")

    # Overall stats
    print(f"\n  OVERALL")
    print(f"  {'':>20}  {'Win%':>6}  {'Count':>5}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*5}")
    print(f"  {'Neural play':>20}  {100*play_wins.mean():5.1f}%  {len(data):5d}")
    if n_solved > 0:
        print(f"  {'Solver (perfect)':>20}  "
              f"{100*solver_wins[solver_mask].mean():5.1f}%  {n_solved:5d}")

    # Correlation
    corr_play = np.corrcoef(vheads, play_outcomes)[0, 1]
    print(f"\n  Correlation (vhead vs play outcome): {corr_play:.3f}")
    if n_solved > 0:
        solver_outcomes = np.where(solver_wins[solver_mask], 1.0, -1.0)
        corr_solver = np.corrcoef(vheads[solver_mask], solver_outcomes)[0, 1]
        print(f"  Correlation (vhead vs solver):       {corr_solver:.3f}")
        # Agreement between play and solver
        agree = (play_wins[solver_mask] == solver_wins[solver_mask]).mean()
        print(f"  Agreement (play vs solver):           {100*agree:.1f}%")

    # Calibration table: vhead bins → solver win%, play win%
    print(f"\n  CALIBRATION BY VALUE HEAD PREDICTION")
    header = f"  {'Bin':>12}  {'Count':>5}  {'Avg pred':>9}"
    if n_solved > 0:
        header += f"  {'Solver%':>8}"
    header += f"  {'Play%':>6}  {'Avg qual':>9}"
    print(header)
    print(f"  {'─'*12}  {'─'*5}  {'─'*9}", end="")
    if n_solved > 0:
        print(f"  {'─'*8}", end="")
    print(f"  {'─'*6}  {'─'*9}")

    edges = [-2.0, -0.5, -0.2, 0.0, 0.2, 0.5, 2.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (vheads >= lo) & (vheads < hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_p = vheads[mask].mean()
        pw = 100 * play_wins[mask].mean()
        avg_q = quals[mask].mean()
        label = f"[{lo:+.1f},{hi:+.1f})"
        line = f"  {label:>12}  {n:5d}  {avg_p:+9.3f}"
        if n_solved > 0:
            sm = mask & solver_mask
            if sm.sum() > 0:
                sw = 100 * solver_wins[sm].mean()
                line += f"  {sw:7.1f}%"
            else:
                line += f"  {'n/a':>8}"
        line += f"  {pw:5.1f}%  {avg_q:9.3f}"
        print(line)

    # Confusion: vhead vs solver
    if n_solved > 0:
        print(f"\n  CONFUSION: VALUE HEAD vs SOLVER")
        print(f"  (threshold=0.0 for vhead → bid/no-bid)")
        vhead_pos = vheads[solver_mask] > 0.0
        sv = solver_wins[solver_mask]
        tp = (vhead_pos & sv).sum()
        fp = (vhead_pos & ~sv).sum()
        fn = (~vhead_pos & sv).sum()
        tn = (~vhead_pos & ~sv).sum()
        print(f"                    Solver WIN  Solver LOSE")
        print(f"    VHead > 0.0       {tp:5d}       {fp:5d}   (precision: "
              f"{100*tp/max(tp+fp,1):.0f}%)")
        print(f"    VHead ≤ 0.0       {fn:5d}       {tn:5d}   (neg pred:  "
              f"{100*tn/max(tn+fn,1):.0f}%)")
        print(f"    Recall: {100*tp/max(tp+fn,1):.0f}%   "
              f"Specificity: {100*tn/max(tn+fp,1):.0f}%")

        # Same at 0.5 threshold
        print(f"\n  (threshold=0.5 for vhead → bid/no-bid)")
        vhead_pos5 = vheads[solver_mask] > 0.5
        tp5 = (vhead_pos5 & sv).sum()
        fp5 = (vhead_pos5 & ~sv).sum()
        fn5 = (~vhead_pos5 & sv).sum()
        tn5 = (~vhead_pos5 & ~sv).sum()
        print(f"                    Solver WIN  Solver LOSE")
        print(f"    VHead > 0.5       {tp5:5d}       {fp5:5d}   (precision: "
              f"{100*tp5/max(tp5+fp5,1):.0f}%)")
        print(f"    VHead ≤ 0.5       {fn5:5d}       {tn5:5d}   (neg pred:  "
              f"{100*tn5/max(tn5+fn5,1):.0f}%)")
        print(f"    Recall: {100*tp5/max(tp5+fn5,1):.0f}%   "
              f"Specificity: {100*tn5/max(tn5+fp5,1):.0f}%")

    # By hand quality
    print(f"\n  BY HAND QUALITY")
    header = f"  {'Quality':>12}  {'Count':>5}  {'Avg pred':>9}"
    if n_solved > 0:
        header += f"  {'Solver%':>8}"
    header += f"  {'Play%':>6}"
    print(header)
    print(f"  {'─'*12}  {'─'*5}  {'─'*9}", end="")
    if n_solved > 0:
        print(f"  {'─'*8}", end="")
    print(f"  {'─'*6}")

    q_edges = [0.0, 0.3, 0.5, 0.6, 0.7, 1.0]
    for lo, hi in zip(q_edges[:-1], q_edges[1:]):
        mask = (quals >= lo) & (quals < hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_p = vheads[mask].mean()
        pw = 100 * play_wins[mask].mean()
        label = f"[{lo:.1f},{hi:.1f})"
        line = f"  {label:>12}  {n:5d}  {avg_p:+9.3f}"
        if n_solved > 0:
            sm = mask & solver_mask
            if sm.sum() > 0:
                sw = 100 * solver_wins[sm].mean()
                line += f"  {sw:7.1f}%"
            else:
                line += f"  {'n/a':>8}"
        line += f"  {pw:5.1f}%"
        print(line)

    print()


if __name__ == "__main__":
    main()
