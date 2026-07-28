"""Parameterized data gen: python gen_data.py <contract>.

Generates ~10k labeled records for a contract using its biased dealer
+ PIMC32 labels. Saves to <contract>_data_10k.npz.
"""
from __future__ import annotations

import itertools, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from solvers import pis, pimc as _pimc
from _vlib import CONTRACT_CONFIGS, data_path, featurize, input_dim

N_RECORDS_TARGET = 10_000
PIMC_N           = 32
ALPHA            = 0.6
SEED_BASE        = 600_000_000
N_WORKERS        = 4


_WORKER_CONTRACT = None


def _init_worker(contract_name: str):
    global _WORKER_CONTRACT
    _WORKER_CONTRACT = contract_name


def worker(seed: int):
    cfg = CONTRACT_CONFIGS[_WORKER_CONTRACT]
    deal = cfg.dealer(seed=seed, alpha=ALPHA)
    sol12 = list(deal.sol_hand) + list(deal.talon)
    if len(sol12) != 12:
        raise RuntimeError(f"expected 12 cards, got {len(sol12)}")
    d1 = list(deal.def1_hand); d2 = list(deal.def2_hand)
    trump = deal.trump if cfg.has_trump else None
    records = []
    rng_seed = seed
    for discard_pair in itertools.combinations(sol12, 2):
        remaining = [c for c in sol12 if c not in discard_pair]
        talon = list(discard_pair)
        pos = pis.build_position(
            hands=[remaining, d1, d2], soloist=0, leader=0,
            contract=cfg.solver, trump=trump, talon=talon,
        )
        rng_seed += 1
        _, avg = _pimc.pimc_decision(
            true_pos=pos, contract=cfg.solver, n_samples=PIMC_N,
            seed=rng_seed,
        )
        p = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
        records.append((featurize(remaining, trump, cfg.has_trump), p))
    return records


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CONTRACT_CONFIGS:
        print(f"Usage: gen_data.py <contract>  (one of "
              f"{list(CONTRACT_CONFIGS)})")
        sys.exit(1)
    contract = sys.argv[1]
    cfg = CONTRACT_CONFIGS[contract]
    out_path = data_path(contract)
    n_deals = (N_RECORDS_TARGET + 65) // 66

    print(f"=== Gen data for contract: {contract} ===")
    print(f"  dealer:    {cfg.dealer.__name__}")
    print(f"  has_trump: {cfg.has_trump}  input_dim: {input_dim(cfg)}")
    print(f"  N_deals:   {n_deals} (×66 records each)")
    print(f"  PIMC_N:    {PIMC_N}  workers: {N_WORKERS}  alpha: {ALPHA}")

    seeds = [SEED_BASE + i for i in range(n_deals)]
    t0 = time.perf_counter()
    all_records = []
    done = 0
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(contract,)) as pool:
        for recs in pool.imap_unordered(worker, seeds):
            all_records.extend(recs)
            done += 1
            if done % 20 == 0:
                wall = time.perf_counter() - t0
                print(f"  {done}/{n_deals} deals  "
                      f"records={len(all_records)}  wall={wall:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0

    X = np.stack([r[0] for r in all_records])
    y = np.array([r[1] for r in all_records], dtype=np.float32)
    np.savez_compressed(out_path, X=X, y=y)
    print()
    print(f"Saved {X.shape[0]} records → {out_path}")
    print(f"Wall: {wall:.0f}s")

    # Distribution
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]
    hist, edges = np.histogram(y, bins=bins)
    print()
    print("Label distribution:")
    for i, h in enumerate(hist):
        bar = '█' * int(h/max(hist)*40) if hist.max() > 0 else ''
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.2f})  {h:>5}  {bar}")
    print(f"  mean P = {y.mean():.3f}, median = {np.median(y):.3f}")


if __name__ == "__main__":
    main()
