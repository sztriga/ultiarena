"""De-risk: how expensive is the god-exact leaf precompute?

The CFR trainer needs, per deal, a leaf payoff table V[player][contract] =
god-exact EV/def if `player` solos `contract` using the value model's best
discard (real talon, real opponents). That's 3 players × ~7 contract/trump
combos = ~21 god solves per deal, each preceded by a value-model best-discard
search over ≤66 discards.

This script measures:
  (1) god solve time per contract type on a 10-card committed position,
  (2) full leaf-table time for one deal,
  (3) extrapolation to a 50k-deal precompute.

Usage: python benchmark.py
"""
from __future__ import annotations

import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))

from _lib import _ev_per_def                       # noqa: E402
from ulti.card import SUITS, fresh_deck            # noqa: E402
from ulti.solvers import pis                            # noqa: E402
from ulti.eval.pimc_matchup import god_says_soloist_wins  # noqa: E402
from ulti.vnet.pickup import featurize, CONTRACT_CONFIGS  # noqa: E402
from ulti.vnet.pickup.composite import CompositePickup  # noqa: E402

EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent.parent / "19_colorless_split"


def deal_10_10_10_2(seed):
    """Standard Ulti deal: three 10-card hands + a fixed 2-card talon."""
    rng = random.Random(seed)
    deck = fresh_deck()
    rng.shuffle(deck)
    return deck[:10], deck[10:20], deck[20:30], deck[30:32]


def _bid_options():
    """(contract, trump) combos a soloist may declare, mirroring the harness:
    parti is piros-only; ulti any suit; betli/duri trumpless."""
    opts = [('parti', 'hearts')]
    opts += [('ulti', s) for s in SUITS]
    opts += [('betli', None), ('durchmars', None)]
    return opts


def best_discard_leaf(picker, hand10, talon, defA, defB, contract, trump):
    """Value-model best discard from the 12 = hand10+talon, then ONE god solve
    of the resulting position → god-exact EV/def."""
    cfg = CONTRACT_CONFIGS[contract]
    hand12 = list(hand10) + list(talon)
    discards = list(itertools.combinations(hand12, 2))
    finals = []
    keep = []
    for dp in discards:
        rem = [c for c in hand12 if c not in dp]
        if contract == 'ulti' and not any(
                c.suit == trump and c.rank == '7' for c in rem):
            continue
        finals.append(rem)
        keep.append(dp)
    if not finals:
        return None  # e.g. ulti with no way to keep trump-7
    X = np.stack([featurize(h, trump, cfg.has_trump) for h in finals])
    ps = picker.predict(X, contract)
    bi = int(np.argmax(ps))
    sol10 = finals[bi]
    discard = keep[bi]
    pos = pis.build_position(
        hands=[sol10, list(defA), list(defB)], soloist=0, leader=0,
        contract=contract, trump=trump, talon=list(discard))
    win = god_says_soloist_wins(pos, contract=contract)
    piros = (trump == 'hearts')
    return _ev_per_def(contract, piros, 1.0 if win else 0.0), float(ps[bi]), win


def main():
    picker = CompositePickup.load(
        trump_weights=EXP18 / "multihead_v18a.pt",
        betli_weights=EXP19 / "colorless_betli.pt",
        durchmars_weights=EXP19 / "colorless_durchmars.pt",
    )
    print("composite model loaded\n")

    # (1) raw god solve time per contract type
    print("=== (1) single god solve time per contract ===")
    h0, h1, h2, talon = deal_10_10_10_2(0)
    for contract, trump in [('parti', 'hearts'), ('ulti', 'hearts'),
                            ('betli', None), ('durchmars', None)]:
        # build a legal 10-card position (drop talon influence; just time solve)
        sol10 = h0
        if contract == 'ulti':
            # ensure trump-7 present for a meaningful solve; skip if absent
            if not any(c.suit == trump and c.rank == '7' for c in sol10):
                print(f"  {contract:>10}/{trump}: (no trump-7 in sample, skipped)")
                continue
        pos = pis.build_position(
            hands=[list(sol10), list(h1), list(h2)], soloist=0, leader=0,
            contract=contract, trump=trump, talon=list(talon))
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            god_says_soloist_wins(pos, contract=contract)
        dt = (time.perf_counter() - t0) / reps
        print(f"  {contract:>10}/{str(trump):>6}: {dt*1000:7.2f} ms/solve")

    # (2) full leaf table for several deals
    print("\n=== (2) full leaf table per deal (3 players × 7 combos) ===")
    opts = _bid_options()
    n_deals = 8
    t0 = time.perf_counter()
    n_solves = 0
    sample = None
    for seed in range(n_deals):
        hands = list(deal_10_10_10_2(seed))
        talon = hands[3]
        players = hands[:3]
        table = {}
        for p in range(3):
            solh = players[p]
            defA = players[(p + 1) % 3]
            defB = players[(p + 2) % 3]
            for contract, trump in opts:
                r = best_discard_leaf(picker, solh, talon, defA, defB,
                                      contract, trump)
                n_solves += 1
                if r is not None:
                    table[(p, contract, trump)] = r[0]
        if sample is None:
            sample = table
    dt = time.perf_counter() - t0
    per_deal = dt / n_deals
    print(f"  {n_deals} deals: {dt:.2f}s total  →  {per_deal*1000:.1f} ms/deal"
          f"  ({n_solves/n_deals:.0f} solves/deal)")
    print(f"  extrapolation: 50k deals ≈ {per_deal*50000/60:.1f} min "
          f"single-thread, ≈ {per_deal*50000/60/8:.1f} min on 8 cores")

    print("\n=== sample leaf table (deal seed=7), EV/def by (player, contract) ===")
    for (p, c, t), ev in sorted(sample.items()):
        tag = f"{c}/{t or 'colorless'}"
        print(f"  P{p}  {tag:>20}  EV/def={ev:+.2f}")


if __name__ == "__main__":
    main()
