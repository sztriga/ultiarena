#!/usr/bin/env python3
"""Diagnose betli pickup decisions — show the exact value head numbers.

Runs the real auction, finds deals where betli gets selected, then
re-runs evaluate_pickup with instrumentation to show the per-contract
quantile values and betli game_pts distribution.
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.bidding.constants import PICKUP_QUANTILE_OVERRIDES
from trickster.bidding.evaluator import evaluate_all_contracts
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame
from trickster.games.ulti.cards import BETLI_STRENGTH, make_deck
from trickster.games.ulti.game import deal, next_player
from trickster.model import UltiNet, make_wrapper
from trickster.train_utils import _GAME_PTS_MAX


def fmt_hand(hand):
    by_suit = {}
    for c in sorted(hand, key=lambda c: (c.suit.value, BETLI_STRENGTH[c.rank])):
        by_suit.setdefault(c.suit.name[0], []).append(c.short())
    return "  ".join(f"{s}: {' '.join(cards)}" for s, cards in by_suit.items())


def instrumented_pickup(gs, player, dealer, wrappers, rng):
    """Re-run evaluate_pickup logic with full instrumentation."""
    hand = gs.hands[player]
    assert len(hand) == 10

    all_cards = frozenset(make_deck())
    unknown = list(all_cards - set(hand))
    n_unknown = len(unknown)

    # Sample 20 talons (same as production)
    k = 20
    talon_samples = []
    seen = set()
    while len(talon_samples) < k:
        pair = tuple(sorted(rng.sample(range(n_unknown), 2)))
        if pair not in seen:
            seen.add(pair)
            talon_samples.append((unknown[pair[0]], unknown[pair[1]]))

    contract_pts: dict[tuple[str, bool], list[float]] = {}

    for talon_pair in talon_samples:
        gs_copy = copy.deepcopy(gs)
        gs_copy.hands[player] = list(hand) + list(talon_pair)

        evals = evaluate_all_contracts(
            gs_copy, player, dealer, wrappers=wrappers, min_bid_rank=0,
        )
        for ev in evals:
            ck_key = (ev.contract_key, ev.is_piros)
            contract_pts.setdefault(ck_key, []).append(ev.game_pts)

    # Quantile selection
    overrides = dict(PICKUP_QUANTILE_OVERRIDES)
    results = {}
    best_ck = None
    best_qval = -float("inf")

    for ck_key, pts_list in contract_pts.items():
        ck, is_piros = ck_key
        q = 0.5
        if ck in overrides:
            q = overrides[ck]
        pts_sorted = sorted(pts_list)
        idx = min(int(len(pts_sorted) * q), len(pts_sorted) - 1)
        qval = pts_sorted[idx]

        label = f"{'p.' if is_piros else ''}{ck}"
        results[label] = dict(
            pts_sorted=pts_sorted, q=q, idx=idx, qval=qval,
            mean=float(np.mean(pts_sorted)),
            mn=min(pts_sorted), mx=max(pts_sorted), n=len(pts_sorted),
        )
        if qval > best_qval:
            best_qval = qval
            best_ck = label

    value = best_qval / (_GAME_PTS_MAX / 2)
    return best_ck, best_qval, value, results


def main():
    game = UltiGame()

    wrappers = {}
    for ck in ["parti", "ulti", "40-100", "betli"]:
        model_pt = Path(f"models/ulti/trinity/final/{ck}/model.pt")
        if not model_pt.exists():
            continue
        cp = torch.load(model_pt, weights_only=False, map_location="cpu")
        n = UltiNet(
            input_dim=cp.get("input_dim", game.state_dim),
            body_units=cp.get("body_units", 256),
            body_layers=cp.get("body_layers", 4),
            action_dim=cp.get("action_dim", game.action_space_size),
        )
        n.load_state_dict(cp["model_state_dict"], strict=False)
        n.eval()
        wrappers[ck] = make_wrapper(n, device="cpu")

    print(f"  Loaded: {list(wrappers.keys())}")

    from trickster.bidding.auction_runner import run_auction

    base_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    max_betli = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    betli_found = 0
    all_betli_qvals = []

    for deal_idx in range(5000):
        seed = base_seed + deal_idx
        gs_orig, talon = deal(seed=seed, dealer=deal_idx % 3)
        dealer = deal_idx % 3

        # Run real auction
        gs_auction = copy.deepcopy(gs_orig)
        seat_wrappers = [wrappers, wrappers, wrappers]
        result = run_auction(
            gs_auction, talon, dealer, seat_wrappers,
            pickup_quantile=0.5,
            quantile_overrides=dict(PICKUP_QUANTILE_OVERRIDES),
            rng=random.Random(seed),
        )
        if result.bid is None or result.bid.contract_key != "betli":
            continue

        soloist = result.soloist
        betli_found += 1

        # Re-run instrumented pickup on the original 10-card hand
        gs_instr = copy.deepcopy(gs_orig)
        rng2 = random.Random(seed + soloist * 1000)  # different rng for talon sampling
        winner, best_qval, value, per_contract = instrumented_pickup(
            gs_instr, soloist, dealer, wrappers, rng2,
        )

        print(f"\n  ═══ Deal {deal_idx} (seed {seed}), Soloist P{soloist} ═══")
        print(f"  10-card hand: {fmt_hand(gs_orig.hands[soloist])}")
        print(f"  Real talon:   {' '.join(c.short() for c in talon)}")
        print()

        # Table of all contracts
        print(f"  {'Contract':<14} {'Q':>4} {'Idx':>3} {'Qval':>7} {'Mean':>7} {'Min':>7} {'Max':>7} {'N':>3}")
        print(f"  {'─' * 60}")
        for label in sorted(per_contract.keys()):
            r = per_contract[label]
            marker = " ◄ WINNER" if label == winner else ""
            print(f"  {label:<14} {r['q']:>4.2f} {r['idx']:>3d} {r['qval']:>+7.2f} "
                  f"{r['mean']:>+7.2f} {r['mn']:>+7.2f} {r['mx']:>+7.2f} "
                  f"{r['n']:>3d}{marker}")

        # Show betli distribution
        for bk in ("betli", "p.betli"):
            if bk in per_contract:
                pts = per_contract[bk]["pts_sorted"]
                print(f"\n  {bk} game_pts (sorted):")
                print(f"    {' '.join(f'{p:+.2f}' for p in pts)}")
                all_betli_qvals.append(per_contract[bk]["qval"])

        print(f"\n  Decision: winner={winner}, qval={best_qval:+.3f}, "
              f"normalised_value={value:+.4f}, threshold=0.0")

        # Danger analysis
        hand = gs_orig.hands[soloist]
        for suit_name in "AHLB":
            suit_cards = [c for c in hand if c.suit.name[0] == suit_name[0]]
            if not suit_cards:
                continue
            max_str = max(BETLI_STRENGTH[c.rank] for c in suit_cards)
            if max_str >= 5:
                print(f"  DANGER {suit_name}: {' '.join(c.short() for c in suit_cards)} "
                      f"(max betli_str={max_str})")

        if betli_found >= max_betli:
            break

    print(f"\n  ══════════════════════════════════════════════════")
    print(f"  Found {betli_found} betli selections in {min(deal_idx + 1, 5000)} deals")
    if all_betli_qvals:
        print(f"  Betli qval range: {min(all_betli_qvals):+.3f} to {max(all_betli_qvals):+.3f}")
        print(f"  Betli qval mean:  {np.mean(all_betli_qvals):+.3f}")


if __name__ == "__main__":
    main()
