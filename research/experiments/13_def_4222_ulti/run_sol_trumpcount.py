"""How does sol's trump count affect ulti winrate when the talon has no
trumps?

Setup:
  - deal_ulti_biased(alpha=0.6) — realistic ulti hands, sol always holds
    the trump-7 (mandatory).
  - Reject any deal where the 2-card talon contains a trump.
  - Group accepted deals by sol's trump count (1=just the 7, up to 8).
  - For each group, run god solver and report P(sol makes ulti).
"""
from __future__ import annotations

import sys, time
from multiprocessing import Pool

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from eval.dojo import deal_ulti_biased
from solvers import pis

ACCEPT_TARGET   = 10_000
ALPHA           = 0.6
SEED_BASE       = 1_000_000_000
N_WORKERS       = 4
BATCH_PER_TASK  = 500
BATCHES_PER_ROUND = 40


def worker(args):
    lo, hi = args
    out = []
    for seed in range(lo, hi):
        deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
        if any(c.suit == deal.trump for c in deal.talon):
            continue
        sol_trumps = sum(1 for c in deal.sol_hand if c.suit == deal.trump)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        pos = pis.build_position(
            hands=hands, soloist=0, leader=0, contract='ulti',
            trump=deal.trump, talon=list(deal.talon),
        )
        vals = pis.solve_all(pos, contract='ulti')
        sol_can_make = max(vals.values()) >= 0.5
        out.append((seed, sol_trumps, sol_can_make))
    return out


def main():
    t0 = time.perf_counter()
    rows = []
    seeds_consumed = 0
    while len(rows) < ACCEPT_TARGET:
        chunks = []
        for w in range(BATCHES_PER_ROUND):
            lo = SEED_BASE + seeds_consumed + w * BATCH_PER_TASK
            hi = lo + BATCH_PER_TASK
            chunks.append((lo, hi))
        seeds_consumed += BATCHES_PER_ROUND * BATCH_PER_TASK
        with Pool(N_WORKERS) as pool:
            for result in pool.imap_unordered(worker, chunks):
                rows.extend(result)
        wall = time.perf_counter() - t0
        print(f"  consumed={seeds_consumed:>7}  accepted={len(rows):>5}  "
              f"rate={len(rows)/seeds_consumed*100:.2f}%  wall={wall:.0f}s",
              flush=True)
        if seeds_consumed > 2_000_000:
            print("  (cap reached)", flush=True)
            break

    wall = time.perf_counter() - t0
    n = len(rows)
    total_won = sum(1 for r in rows if r[2])
    print()
    print(f"=== Overall (N={n} accepted from {seeds_consumed}, "
          f"accept={n/seeds_consumed*100:.2f}%) ===")
    print(f"  sol made ulti:   {total_won:5d}  ({total_won/n*100:.2f}%)")

    print()
    print("=== By sol trump count (incl. the 7), talon trump-void ===")
    print(f"  {'sol_trumps':>10}  {'N':>6}  {'made':>5}  {'rate':>8}")
    by_st = {}
    for r in rows:
        by_st.setdefault(r[1], []).append(r)
    for st in sorted(by_st):
        sub = by_st[st]
        nw = sum(1 for r in sub if r[2])
        print(f"  {st:>10}  {len(sub):>6}  {nw:>5}  {nw/len(sub)*100:>7.2f}%")

    print()
    print(f"Wall: {wall:.0f}s, {N_WORKERS} workers, alpha={ALPHA}")


if __name__ == "__main__":
    main()
