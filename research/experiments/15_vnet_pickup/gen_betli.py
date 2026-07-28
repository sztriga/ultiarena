"""Generate ~10k labeled betli pickup records via deal_betli + PIMC32.

Each record = (32-dim multi-hot of sol's 10-card post-discard hand,
P_make_betli from PIMC32). Saved as .npz with arrays `X` and `y`.

Uses deal_betli to ensure label distribution covers 0..1 (uniform deals
collapse to ~0).
"""
from __future__ import annotations

import itertools, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from eval.dojo import deal_betli
from ulti.card import Card
from solvers import pis, pimc as _pimc

N_RECORDS_TARGET = 10_000
PIMC_N           = 32
ALPHA            = 0.6
SEED_BASE        = 600_000_000
N_WORKERS        = 4
OUT_DIR          = Path(__file__).parent
OUT_PATH         = OUT_DIR / "betli_data_10k.npz"


def _hand_vec(hand) -> np.ndarray:
    v = np.zeros(32, dtype=np.float32)
    for c in hand:
        v[c.id] = 1.0
    return v


def worker(seed: int):
    """Generate one biased betli deal, label all 66 discards."""
    deal = deal_betli(seed=seed, alpha=ALPHA)
    # deal_betli returns a 10-card play hand + 2-card talon. For the
    # bidding-phase pickup we reconstruct the 12-card pre-discard hand
    # by unioning them.
    sol = list(deal.sol_hand) + list(deal.talon)
    assert len(sol) == 12, f"expected 12 cards, got {len(sol)}"

    d1 = list(deal.def1_hand)
    d2 = list(deal.def2_hand)
    records = []
    rng_seed = seed
    for discard_pair in itertools.combinations(sol, 2):
        remaining = [c for c in sol if c not in discard_pair]
        talon = list(discard_pair)
        pos = pis.build_position(
            hands=[remaining, d1, d2], soloist=0, leader=0,
            contract='betli', trump=None, talon=talon,
        )
        rng_seed += 1
        _, avg = _pimc.pimc_decision(
            true_pos=pos, contract='betli', n_samples=PIMC_N, seed=rng_seed,
        )
        if not avg:
            p = 0.0
        else:
            p = max(0.0, min(1.0, max(avg.values())))
        records.append((_hand_vec(remaining), p))
    return records


def main():
    # 66 records per deal → need ~152 deals for 10k records
    n_deals_needed = (N_RECORDS_TARGET + 65) // 66
    seeds = [SEED_BASE + i for i in range(n_deals_needed)]
    print(f"Generating {n_deals_needed} deals × 66 = ~{n_deals_needed*66} records")
    print(f"  PIMC_N={PIMC_N}  workers={N_WORKERS}  alpha={ALPHA}")

    t0 = time.perf_counter()
    all_records = []
    done = 0
    with Pool(N_WORKERS) as pool:
        for recs in pool.imap_unordered(worker, seeds):
            all_records.extend(recs)
            done += 1
            if done % 20 == 0:
                wall = time.perf_counter() - t0
                print(f"  {done}/{n_deals_needed} deals  records={len(all_records)}  "
                      f"wall={wall:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    X = np.stack([r[0] for r in all_records])
    y = np.array([r[1] for r in all_records], dtype=np.float32)
    np.savez_compressed(OUT_PATH, X=X, y=y)
    print()
    print(f"Saved {X.shape[0]} records → {OUT_PATH}")
    print(f"Wall: {wall:.0f}s  records/sec: {len(all_records)/wall:.0f}")

    # Quick distribution stats
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]
    hist, edges = np.histogram(y, bins=bins)
    print()
    print("Label distribution:")
    for i, h in enumerate(hist):
        bar = '█' * int(h/max(hist)*40)
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.2f})  {h:>5}  {bar}")
    print(f"  mean P_betli = {y.mean():.3f}, median = {np.median(y):.3f}")


if __name__ == "__main__":
    main()
