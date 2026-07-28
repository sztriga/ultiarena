"""Smoke timing test for v-net data gen on betli.

Per deal: 12-card sol hand → 66 talon discards → for each post-discard
10-card hand, label P(make betli) via PIMC32 (32 PIMC samples, god
solver per sample). This produces 66 labeled records per dealt hand.

Measures: wall time, calls/sec, per-record cost. Prints a sample of
labeled records so we can sanity-check the labels.
"""
from __future__ import annotations

import itertools, random, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.card import fresh_deck
from ulti.solvers import pis, pimc as _pimc

# Smoke knobs
N_HANDS    = 20      # number of 12-card sol hands
PIMC_N     = 32
SEED_BASE  = 500_000
N_WORKERS  = 4


def deal_12_10_10(seed):
    rng = random.Random(seed)
    deck = fresh_deck()
    rng.shuffle(deck)
    return deck[:12], deck[12:22], deck[22:32]


def label_betli(remaining_10, def1_10, def2_10, talon, pimc_n, seed):
    pos = pis.build_position(
        hands=[remaining_10, def1_10, def2_10], soloist=0, leader=0,
        contract='betli', trump=None, talon=talon,
    )
    _, avg = _pimc.pimc_decision(
        true_pos=pos, contract='betli', n_samples=pimc_n, seed=seed,
    )
    if not avg:
        return 0.0
    # max_avg is sol's best-move P(make) on the new 0/1 binary scale
    return max(0.0, min(1.0, max(avg.values())))


def worker(seed):
    sol12, d1, d2 = deal_12_10_10(seed)
    records = []
    rng_seed = seed
    discards = list(itertools.combinations(sol12, 2))
    for discard_pair in discards:
        remaining = [c for c in sol12 if c not in discard_pair]
        talon = list(discard_pair)
        rng_seed += 1
        p = label_betli(remaining, d1, d2, talon, PIMC_N, rng_seed)
        records.append((tuple(str(c) for c in remaining),
                        tuple(str(c) for c in discard_pair), p))
    return seed, records


def main():
    seeds = [SEED_BASE + i for i in range(N_HANDS)]
    print(f"=== Betli data-gen smoke ===")
    print(f"  N_HANDS:    {N_HANDS}")
    print(f"  PIMC_N:     {PIMC_N}")
    print(f"  workers:    {N_WORKERS}")
    print(f"  records expected: {N_HANDS * 66}")
    print()

    t0 = time.perf_counter()
    all_records = []
    with Pool(N_WORKERS) as pool:
        for seed, recs in pool.imap_unordered(worker, seeds):
            all_records.extend(recs)
    wall = time.perf_counter() - t0
    n = len(all_records)

    print(f"  wall:           {wall:.1f}s")
    print(f"  records:        {n}")
    print(f"  s / record:     {wall/n*1000:.1f} ms")
    print(f"  records / sec:  {n/wall:.0f}")
    print(f"  s / 12-card deal: {wall/N_HANDS:.2f}s")
    print()

    # Cost projections
    for total_records in (10_000, 100_000, 1_000_000):
        proj_wall = wall * total_records / n
        print(f"  projected wall for {total_records:>8,} records: "
              f"{proj_wall:.0f}s = {proj_wall/60:.1f} min = {proj_wall/3600:.2f} h")

    print()
    # Sample labels: pick a few records with high and low P
    sorted_recs = sorted(all_records, key=lambda r: r[2])
    print("=== Lowest-P records (likely poor betli hands) ===")
    for h, d, p in sorted_recs[:3]:
        print(f"  P_betli={p:.3f}  hand={h}  discarded={d}")
    print()
    print("=== Highest-P records (good betli hands) ===")
    for h, d, p in sorted_recs[-3:]:
        print(f"  P_betli={p:.3f}  hand={h}  discarded={d}")

    # Distribution histogram (rough)
    buckets = [0]*11
    for _, _, p in all_records:
        buckets[min(10, int(p*10))] += 1
    print()
    print("=== P(make betli) histogram ===")
    for i, n_b in enumerate(buckets):
        lo = i*0.1; hi = (i+1)*0.1 if i<10 else 1.01
        bar = '█' * int(n_b/max(buckets)*40)
        print(f"  [{lo:.1f}, {hi:.2f})  {n_b:>5}  {bar}")


if __name__ == "__main__":
    main()
