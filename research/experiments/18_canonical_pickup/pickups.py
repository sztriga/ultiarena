"""How many pickups happen per deal in the v18a symmetric auction?

Resolves the auction only (play_out=False — skips the PIMC32-vs-god
playout) so it's fast. Reports the distribution of commits per deal
(P0 open + each overtake), how often each seat re-takes the bid, and
the most common bid sequences.

Usage: N_DEALS=3000 python pickups.py
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import simulate
from ulti.vnet.pickup.v18 import Exp18Pickup

EXP_DIR   = Path(__file__).parent
WEIGHTS   = EXP_DIR / "multihead_v18a.pt"
N         = int(os.environ.get("N_DEALS", 3000))
SEED_BASE = 100_000
N_WORKERS = 8

_PICKER = None


def _get():
    global _PICKER
    if _PICKER is None:
        _PICKER = Exp18Pickup.load(WEIGHTS)
    return _PICKER


def _run_one(seed):
    p = _get()
    return simulate(seed, [p, p, p], play_out=False)


def main():
    print(f"=== v18a auction pickup distribution: N={N} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_run_one, seeds, chunksize=8):
            rows.append(r)
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{N}  wall={time.perf_counter()-t0:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0
    print(f"  wall: {wall:.0f}s\n")

    # ── n_pickups distribution ───────────────────────────────────────
    dist = Counter(r['n_pickups'] for r in rows)
    total_pickups = sum(r['n_pickups'] for r in rows)
    print("Pickups per deal (P0 open + each overtake):")
    print(f"  {'#pickups':>9}  {'deals':>6}  {'%':>6}")
    for k in sorted(dist):
        label = f"{k}" + ("  (forced pass)" if k == 0 else
                           "  (P0 opens, no overtake)" if k == 1 else
                           f"  ({k-1} overtake{'s' if k-1 != 1 else ''})")
        print(f"  {label:>9}  {dist[k]:>6}  {dist[k]/N*100:>5.1f}")
    print(f"\n  mean pickups/deal: {total_pickups/N:.3f}")
    overtaken = sum(1 for r in rows if r['n_pickups'] >= 2)
    print(f"  deals with ≥1 overtake: {overtaken} ({overtaken/N*100:.1f}%)")
    print(f"  total commits across all deals: {total_pickups}")

    # ── how often each seat commits / re-takes ───────────────────────
    print("\nCommits by seat (how often each seat picks up at all):")
    seat_commits = Counter()
    seat_multi = Counter()   # seat picked up ≥2× in one deal
    for r in rows:
        pids = [b[0] for b in r['bid_seq']]
        for pid in set(pids):
            seat_commits[pid] += 1
        c = Counter(pids)
        for pid, cnt in c.items():
            if cnt >= 2:
                seat_multi[pid] += 1
    print(f"  {'seat':>4}  {'deals it commits in':>20}  {'deals it re-takes (≥2×)':>24}")
    for pid in (0, 1, 2):
        print(f"  P{pid:>3}  {seat_commits[pid]:>20}  {seat_multi[pid]:>24}")

    # ── most common bid sequences (by contract path) ─────────────────
    print("\nMost common commit sequences (seat:contract), top 12:")
    def seqkey(r):
        return " → ".join(f"P{p}:{c}" for p, c, _ in r['bid_seq']) or "(pass)"
    seqs = Counter(seqkey(r) for r in rows)
    for s, n in seqs.most_common(12):
        print(f"  {n:>5}  ({n/N*100:>4.1f}%)  {s}")

    # ── longest auctions ─────────────────────────────────────────────
    mx = max(r['n_pickups'] for r in rows)
    print(f"\n  longest auction: {mx} commits")
    for r in rows:
        if r['n_pickups'] == mx:
            print(f"    seed {r['seed']}: " +
                  " → ".join(f"P{p}:{c}/{t or 'col'}" for p, c, t in r['bid_seq']))
            break


if __name__ == "__main__":
    main()
