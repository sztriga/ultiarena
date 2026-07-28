"""Sidequest v2: realistic ulti-biased deals where one defender holds
exactly 4 trumps + 2-2-2 of the non-trump suits.

Difference from run_def_4222.py:
  v1 used a uniform sampler for sol's other 9 cards (sol's hand was
  weak on average → upper bound on tank's lethality). v2 samples via
  ``deal_ulti_biased`` (sol gets fat trump + biased side strength,
  matching realistic ulti-bid hands) and rejection-filters for the
  4-2-2-2 tank shape.

Acceptance rate is low (~0.5–2%) so this is slower than v1. Aim for
the requested ACCEPT_TARGET and let the dealer churn.
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.eval.dojo import deal_ulti_biased
from ulti.solvers import pis

ACCEPT_TARGET = 5_000      # number of 4-2-2-2 deals to solve
ALPHA         = 0.6
SEED_BASE     = 800_000_000
N_WORKERS     = 4
BATCH_PER_WORKER = 200      # seeds per dispatched task


def _classify_defender(hand, trump):
    """If this defender has exactly 4 trumps + 2 of each non-trump suit,
    return True; else False."""
    by_suit = {}
    for c in hand:
        by_suit[c.suit] = by_suit.get(c.suit, 0) + 1
    if by_suit.get(trump, 0) != 4:
        return False
    for s, n in by_suit.items():
        if s == trump:
            continue
        if n != 2:
            return False
    # And every non-trump suit must be present (else it's not 2-2-2)
    non_trump_suits_in_hand = [s for s in by_suit if s != trump]
    return len(non_trump_suits_in_hand) == 3


def worker(args):
    """Process a contiguous block of seeds. For each, generate a
    biased deal and check if either defender matches the 4-2-2-2
    shape. Solve only accepted deals."""
    seed_lo, seed_hi = args
    out = []
    for seed in range(seed_lo, seed_hi):
        deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
        d1_tank = _classify_defender(deal.def1_hand, deal.trump)
        d2_tank = _classify_defender(deal.def2_hand, deal.trump)
        if not (d1_tank or d2_tank):
            continue
        tank = 1 if d1_tank else 2
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        pos = pis.build_position(
            hands=hands, soloist=0, leader=0, contract='ulti',
            trump=deal.trump, talon=list(deal.talon),
        )
        vals = pis.solve_all(pos, contract='ulti')
        sol_can_make = max(vals.values()) >= 0.5
        sol_trumps = sum(1 for c in hands[0] if c.suit == deal.trump)
        out.append((seed, tank, deal.trump, sol_can_make, sol_trumps))
    return out


def main():
    t0 = time.perf_counter()
    rows = []
    seeds_consumed = 0
    batch = 0
    while len(rows) < ACCEPT_TARGET:
        # Dispatch N_WORKERS * BATCH_PER_WORKER seeds in parallel
        n_seeds = N_WORKERS * BATCH_PER_WORKER * 10
        chunks = []
        for w in range(N_WORKERS * 10):
            lo = SEED_BASE + seeds_consumed + w * BATCH_PER_WORKER
            hi = lo + BATCH_PER_WORKER
            chunks.append((lo, hi))
        seeds_consumed += n_seeds
        with Pool(N_WORKERS) as pool:
            for result in pool.imap_unordered(worker, chunks):
                rows.extend(result)
        wall = time.perf_counter() - t0
        accept_rate = len(rows) / seeds_consumed
        print(f"  consumed={seeds_consumed:>7}  accepted={len(rows):>5}  "
              f"rate={accept_rate*100:.2f}%  wall={wall:.0f}s", flush=True)
        batch += 1
        if batch > 50:
            print("  (giving up after 50 batches)", flush=True)
            break

    wall = time.perf_counter() - t0

    n = len(rows)
    if n == 0:
        print("No accepted deals. Aborting summary.")
        return
    total_won = sum(1 for r in rows if r[3])
    total_failed = n - total_won
    print()
    print(f"=== Overall (N={n} accepted of {seeds_consumed} sampled, "
          f"accept={n/seeds_consumed*100:.2f}%) ===")
    print(f"  sol made ulti:   {total_won:5d}  ({total_won/n:.4f})")
    print(f"  sol failed:      {total_failed:5d}  ({total_failed/n:.4f})")

    for tank in (1, 2):
        sub = [r for r in rows if r[1] == tank]
        ns = len(sub)
        if ns == 0:
            continue
        nw = sum(1 for r in sub if r[3])
        nf = ns - nw
        print()
        print(f"=== Tank = def{tank} (N={ns}) ===")
        print(f"  sol made ulti:   {nw:5d}  ({nw/ns:.4f})")
        print(f"  sol failed:      {nf:5d}  ({nf/ns:.4f})")

    # Sol trump count breakdown — does the lethality depend on sol's trump strength?
    print()
    print("=== Sol trump count vs ulti success ===")
    print(f"  {'sol_trumps':>10}  {'N':>6}  {'made':>5}  {'rate':>8}")
    by_st = {}
    for r in rows:
        by_st.setdefault(r[4], []).append(r)
    for st in sorted(by_st):
        sub = by_st[st]
        nw = sum(1 for r in sub if r[3])
        print(f"  {st:>10}  {len(sub):>6}  {nw:>5}  {nw/len(sub)*100:>7.2f}%")

    print()
    print(f"Wall: {wall:.0f}s, {N_WORKERS} workers, alpha={ALPHA}")


if __name__ == "__main__":
    main()
