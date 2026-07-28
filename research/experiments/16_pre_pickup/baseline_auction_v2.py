"""Calibrated, threshold-free auction baseline. Uses auction_v2.simulate."""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from auction_v2 import simulate

N         = int(os.environ.get("N_DEALS", 300))
SEED_BASE = 100_000
N_WORKERS = 8


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(simulate, seeds):
            rows.append(r)
            if len(rows) % 25 == 0:
                wall = time.perf_counter() - t0
                print(f"  {len(rows)}/{N}  wall={wall:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    print()
    print(f"=== Calibrated auction (N={N}, def=god, sol=PIMC32) ===")
    print(f"  wall: {wall:.0f}s ({wall/60:.1f} min)")
    print()

    pickups_dist = Counter(r['n_pickups'] for r in rows)
    print("Pickups per deal:")
    for k in sorted(pickups_dist):
        print(f"  {k} pickup{'s' if k != 1 else ''}: {pickups_dist[k]:>4}  "
              f"({pickups_dist[k]/N*100:.1f}%)")
    print()

    winner_dist = Counter(r['winner_pid'] for r in rows)
    n_penalty = winner_dist.get(None, 0)
    print("Winner:")
    for pid in (0, 1, 2):
        print(f"  P{pid}: {winner_dist[pid]:>4} ({winner_dist[pid]/N*100:.1f}%)")
    if n_penalty:
        print(f"  PASS PENALTY: {n_penalty} ({n_penalty/N*100:.1f}%)")
    print()

    contract_dist = Counter(r['winning_bid'] for r in rows)
    print("Winning bid:")
    for k in sorted(contract_dist, key=lambda x: -contract_dist[x]):
        print(f"  {k:>22}  {contract_dist[k]:>4}  ({contract_dist[k]/N*100:.1f}%)")
    print()

    print("Total GP per player:")
    for pid in (0, 1, 2):
        total = sum(r['gps'][pid] for r in rows)
        mean = total / N
        print(f"  P{pid}: total={total:+8.1f}  mean/deal={mean:+.3f}")
    print()

    print("Soloist performance:")
    for pid in (0, 1, 2):
        my_deals = [r for r in rows if r['winner_pid'] == pid]
        if not my_deals:
            continue
        my_won = sum(1 for r in my_deals if r['gp_per_def'] > 0)
        avg = sum(r['gp_per_def'] for r in my_deals) / len(my_deals)
        print(f"  P{pid}: bids={len(my_deals)}  won={my_won} "
              f"({my_won/len(my_deals)*100:.1f}%)  avg GP/def={avg:+.3f}")
    print()

    print("Per-contract performance (winner's view):")
    by_contract = {}
    for r in rows:
        by_contract.setdefault(r['winning_bid'], []).append(r['gp_per_def'])
    print(f"  {'bid':>22}  {'n':>4}  {'avg GP/def':>11}  {'won':>4}  {'won %':>7}")
    for k in sorted(by_contract, key=lambda x: -len(by_contract[x])):
        vals = by_contract[k]
        won = sum(1 for v in vals if v > 0)
        print(f"  {k:>22}  {len(vals):>4}  {_mean(vals):>+10.2f}  "
              f"{won:>4}  {won/len(vals)*100:>6.1f}%")
    print()

    print("Per-player per-contract:")
    print(f"  {'bid':>22}  {'player':>6}  {'n':>4}  {'avg GP/def':>11}  "
          f"{'won':>4}  {'won %':>7}")
    by_pc = {}
    for r in rows:
        if r['winner_pid'] is None:
            continue
        by_pc.setdefault((r['winning_bid'], r['winner_pid']), []).append(r['gp_per_def'])
    for bid in sorted(by_contract, key=lambda x: -len(by_contract[x])):
        if bid == 'PASS_PENALTY':
            continue
        for pid in (0, 1, 2):
            vals = by_pc.get((bid, pid), [])
            if not vals:
                continue
            won = sum(1 for v in vals if v > 0)
            print(f"  {bid:>22}  P{pid:>4}  {len(vals):>4}  "
                  f"{_mean(vals):>+10.2f}  {won:>4}  {won/len(vals)*100:>6.1f}%")


if __name__ == "__main__":
    main()
