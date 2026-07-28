"""Focus on the sol-has-4-trumps cohort within the 4-2-2-2 tank setup.

Drill-down: among deals where (a) one defender holds 4 trumps + 2-2-2
and (b) sol holds 4 trumps (incl. the 7), what determines whether sol
can make ulti? Break down by which high trumps sol holds (ace, 10,
king) and which trumps the tank holds.
"""
from __future__ import annotations

import sys, time
from multiprocessing import Pool

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from eval.dojo import deal_ulti_biased
from solvers import pis

ACCEPT_TARGET = 3_000
ALPHA         = 0.6
SEED_BASE     = 900_000_000
N_WORKERS     = 4
BATCH_PER_TASK = 500


def _is_tank(hand, trump):
    by_suit = {}
    for c in hand:
        by_suit[c.suit] = by_suit.get(c.suit, 0) + 1
    if by_suit.get(trump, 0) != 4:
        return False
    non_trump = [s for s in by_suit if s != trump]
    if len(non_trump) != 3:
        return False
    return all(by_suit[s] == 2 for s in non_trump)


def worker(args):
    lo, hi = args
    out = []
    for seed in range(lo, hi):
        deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
        d1_tank = _is_tank(deal.def1_hand, deal.trump)
        d2_tank = _is_tank(deal.def2_hand, deal.trump)
        if not (d1_tank or d2_tank):
            continue
        sol_trumps = [c for c in deal.sol_hand if c.suit == deal.trump]
        if len(sol_trumps) != 4:
            continue
        tank = 1 if d1_tank else 2
        tank_hand = deal.def1_hand if d1_tank else deal.def2_hand
        tank_trump_ranks = sorted(c.rank for c in tank_hand if c.suit == deal.trump)
        sol_trump_ranks = sorted(c.rank for c in sol_trumps)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        pos = pis.build_position(
            hands=hands, soloist=0, leader=0, contract='ulti',
            trump=deal.trump, talon=list(deal.talon),
        )
        vals = pis.solve_all(pos, contract='ulti')
        sol_can_make = max(vals.values()) >= 0.5
        out.append((seed, tank, deal.trump, sol_can_make,
                    tuple(sol_trump_ranks), tuple(tank_trump_ranks)))
    return out


def main():
    t0 = time.perf_counter()
    rows = []
    seeds_consumed = 0
    BATCHES_PER_ROUND = 80
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
        print(f"  consumed={seeds_consumed:>8}  accepted={len(rows):>5}  "
              f"rate={len(rows)/seeds_consumed*100:.3f}%  wall={wall:.0f}s",
              flush=True)
        if seeds_consumed > 5_000_000:
            print("  (cap on consumed seeds reached)", flush=True)
            break

    wall = time.perf_counter() - t0
    n = len(rows)
    if n == 0:
        print("No accepted deals.")
        return
    total_won = sum(1 for r in rows if r[3])
    print()
    print(f"=== Overall (N={n} accepted from {seeds_consumed} seeds, "
          f"accept={n/seeds_consumed*100:.3f}%) ===")
    print(f"  sol made ulti:   {total_won:5d}  ({total_won/n:.4f})")
    print(f"  sol failed:      {n-total_won:5d}  ({(n-total_won)/n:.4f})")

    # Break down by what high trumps sol holds (A, 10, K)
    print()
    print("=== Sol's high-trump holdings vs ulti success ===")
    print(f"  {'has A':>5}  {'has 10':>6}  {'has K':>5}  {'N':>5}  {'made':>5}  {'rate':>8}")
    by_key = {}
    for r in rows:
        sr = set(r[4])
        key = ('ace' in sr, '10' in sr, 'king' in sr)
        by_key.setdefault(key, []).append(r)
    for key in sorted(by_key, key=lambda k: (-sum(k),)):
        sub = by_key[key]
        nw = sum(1 for x in sub if x[3])
        a, t, k = key
        print(f"  {('Y' if a else '.'):>5}  {('Y' if t else '.'):>6}  "
              f"{('Y' if k else '.'):>5}  {len(sub):>5}  {nw:>5}  "
              f"{nw/len(sub)*100:>7.2f}%")

    # Break down by tank's high trumps
    print()
    print("=== Tank's high-trump holdings vs ulti success ===")
    print(f"  {'has A':>5}  {'has 10':>6}  {'has K':>5}  {'N':>5}  {'made':>5}  {'rate':>8}")
    by_tk = {}
    for r in rows:
        tr = set(r[5])
        key = ('ace' in tr, '10' in tr, 'king' in tr)
        by_tk.setdefault(key, []).append(r)
    for key in sorted(by_tk, key=lambda k: (-sum(k),)):
        sub = by_tk[key]
        nw = sum(1 for x in sub if x[3])
        a, t, k = key
        print(f"  {('Y' if a else '.'):>5}  {('Y' if t else '.'):>6}  "
              f"{('Y' if k else '.'):>5}  {len(sub):>5}  {nw:>5}  "
              f"{nw/len(sub)*100:>7.2f}%")

    # Joint: top sol trump rank vs tank's top trump rank
    print()
    print("=== Top sol trump (excl 7) vs top tank trump ===")
    RANK_ORDER = ['7','8','9','lower','upper','king','10','ace']
    def top_rank(ranks):
        return max(ranks, key=lambda r: RANK_ORDER.index(r))
    by_top = {}
    for r in rows:
        sol_non7 = [x for x in r[4] if x != '7']
        if not sol_non7:
            continue
        st = top_rank(sol_non7)
        tt = top_rank(r[5])
        by_top.setdefault((st, tt), []).append(r)
    rank_order = RANK_ORDER
    # Sort DESCENDING so the strongest trump (ace) sits in the top-left
    # corner — easier to read which combos dominate.
    sol_tops = sorted({k[0] for k in by_top}, key=lambda x: -rank_order.index(x))
    tank_tops = sorted({k[1] for k in by_top}, key=lambda x: -rank_order.index(x))
    print(f"  {'sol\\tank':>10} " + " ".join(f"{t:>8}" for t in tank_tops))
    for s in sol_tops:
        cells = []
        for t in tank_tops:
            sub = by_top.get((s, t), [])
            if not sub:
                cells.append(f"{'—':>8}")
            else:
                nw = sum(1 for x in sub if x[3])
                cells.append(f"{nw/len(sub)*100:>5.1f}% ")
        print(f"  {s:>10} " + " ".join(cells))

    print()
    print(f"Wall: {wall:.0f}s, {N_WORKERS} workers, alpha={ALPHA}")


if __name__ == "__main__":
    main()
