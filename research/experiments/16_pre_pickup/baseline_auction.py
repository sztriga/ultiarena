"""Run auction simulator over N deals; report rich diagnostics."""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from auction import simulate

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

    # ─── Aggregates ──────────────────────────────────────────────
    print()
    print(f"=== Auction baseline (N={N}, def=god, sol=PIMC32) ===")
    print(f"  wall: {wall:.0f}s ({wall/60:.1f} min)")
    print()

    # n_pickups distribution
    pickups_dist = Counter(r['n_pickups'] for r in rows)
    print("Pickups per deal:")
    for k in sorted(pickups_dist):
        print(f"  {k} pickup{'s' if k != 1 else ''}: {pickups_dist[k]:>4}  "
              f"({pickups_dist[k]/N*100:.1f}%)")
    print(f"  mean: {_mean([r['n_pickups'] for r in rows]):.2f}")
    print()

    # Winner distribution
    winner_dist = Counter(r['winner_pid'] for r in rows)
    n_penalty = winner_dist.get(None, 0)
    print("Winner (final bidder):")
    for pid in (0, 1, 2):
        print(f"  P{pid}: {winner_dist[pid]:>4} ({winner_dist[pid]/N*100:.1f}%)")
    if n_penalty:
        print(f"  PASS PENALTY (P0 declined to bid): {n_penalty} ({n_penalty/N*100:.1f}%)")
    print()

    # Winning contract distribution
    contract_dist = Counter(r['winning_bid'] for r in rows)
    print("Winning bid:")
    for k in sorted(contract_dist, key=lambda x: -contract_dist[x]):
        print(f"  {k:>22}  {contract_dist[k]:>4}  ({contract_dist[k]/N*100:.1f}%)")
    print()

    # GP per player
    print("Total GP per player (across all deals):")
    for pid in (0, 1, 2):
        total = sum(r['gps'][pid] for r in rows)
        mean = total / N
        print(f"  P{pid}: total={total:+8.1f}  mean/deal={mean:+.3f}")
    print()

    # Win rate per soloist (only counting deals where they were the winner)
    print("Soloist performance (when they ended up as soloist):")
    for pid in (0, 1, 2):
        my_deals = [r for r in rows if r['winner_pid'] == pid]
        if not my_deals:
            continue
        my_won = sum(1 for r in my_deals if r['gp_per_def'] > 0)
        my_gp_per_deal = sum(r['gp_per_def'] for r in my_deals) / len(my_deals)
        print(f"  P{pid}: bids={len(my_deals)}  won={my_won} ({my_won/len(my_deals)*100:.1f}%)  "
              f"avg GP/def per bid={my_gp_per_deal:+.3f}")
    print()

    # Per-contract win % (soloist's perspective)
    print("Per-contract performance (winner's perspective):")
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

    # Per-player per-contract breakdown
    print("Per-player per-contract performance (when that player was soloist):")
    print(f"  {'bid':>22}  {'player':>6}  {'n':>4}  {'avg GP/def':>11}  "
          f"{'won':>4}  {'won %':>7}")
    by_pc = {}  # (contract, pid) -> list of gp_per_def
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
    print()

    # Forced-bid analysis: when P0 opened weak (negative open EV), what happened?
    forced = []
    for r in rows:
        open_e = r['log'][0]
        if open_e.get('action') == 'open_pass_penalty':
            continue
        if open_e.get('ev', 0) < 0:
            forced.append(r)
    n_forced = len(forced)
    n_forced_overtaken = sum(1 for r in forced
                             if r['winner_pid'] not in (0, None))
    print(f"P0 forced-bid analysis (open EV < 0): {n_forced} deals "
          f"({n_forced/N*100:.1f}%)")
    if n_forced:
        print(f"  overtaken: {n_forced_overtaken}/{n_forced} ({n_forced_overtaken/n_forced*100:.1f}%)")
        gp_p0_forced = sum(r['gps'][0] for r in forced) / n_forced
        print(f"  P0 mean GP on forced opens: {gp_p0_forced:+.2f}")


if __name__ == "__main__":
    main()
