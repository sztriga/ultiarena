"""Thread sweep over the minigame eval. Fixed N deals, sweep
N_WORKERS ∈ {1, 2, 4, 8}, report wall time and per-deal seconds."""
from __future__ import annotations

import sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from _lib import deal_12_10_10, eval_one_deal, best_record

N         = 8
PIMC_N    = 32
SEED_BASE = 50_000


def worker(seed):
    sol12, d1, d2 = deal_12_10_10(seed)
    recs = eval_one_deal(sol12, d1, d2, pimc_n=PIMC_N, seed=seed * 17)
    best = best_record(recs)
    return seed, best


def main():
    seeds = [SEED_BASE + i for i in range(N)]

    print(f"=== Minigame perf sweep ===")
    print(f"  N deals: {N}, PIMC_N: {PIMC_N}")
    print()
    print(f"  {'workers':>8}  {'wall':>7}  {'s/deal':>7}  {'speedup':>8}")

    baseline = None
    for nw in (1, 2, 4, 8):
        t0 = time.perf_counter()
        with Pool(nw) as pool:
            results = list(pool.imap_unordered(worker, seeds))
        wall = time.perf_counter() - t0
        if baseline is None:
            baseline = wall
        sp = baseline / wall
        print(f"  {nw:>8}  {wall:>6.1f}s  {wall/N:>6.2f}s  {sp:>7.2f}x")


if __name__ == "__main__":
    main()
