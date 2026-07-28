"""Tier 1 — calibration check on fresh held-out 50k α=0 deals/contract.

Pass criteria: no bin has |predicted − actual| > 0.05.
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from solvers import pis
from eval.pimc_matchup import god_says_soloist_wins
from vnet.pickup import CONTRACT_CONFIGS, featurize, pad_to_unified
from vnet.pickup.v17 import Exp17Pickup

EXP_DIR = Path(__file__).parent
WEIGHTS = EXP_DIR / "multihead_v17.pt"
N       = 50_000
SEED_BASE = 900_000_000   # disjoint from train (8e8) and eval (1e5)
N_WORKERS = 6

_WORKER_CONTRACT = None

def _init_worker(c):
    global _WORKER_CONTRACT
    _WORKER_CONTRACT = c

def worker(seed):
    cfg = CONTRACT_CONFIGS[_WORKER_CONTRACT]
    deal = cfg.dealer(seed=seed, alpha=0.0)
    sol12 = list(deal.sol_hand) + list(deal.talon)
    d1 = list(deal.def1_hand); d2 = list(deal.def2_hand)
    trump = deal.trump if cfg.has_trump else None
    rng = random.Random(seed ^ 0xA5A5A5A5)
    idx = rng.sample(range(12), 2)
    discard = [sol12[i] for i in idx]
    rem = [sol12[i] for i in range(12) if i not in idx]
    pos = pis.build_position(
        hands=[rem, d1, d2], soloist=0, leader=0,
        contract=cfg.solver, trump=trump, talon=discard,
    )
    label = 1.0 if god_says_soloist_wins(pos, contract=cfg.solver) else 0.0
    return featurize(rem, trump, cfg.has_trump), label


def gen(contract, n):
    seeds = [SEED_BASE + i for i in range(n)]
    t0 = time.perf_counter()
    Xs = []; ys = []
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(contract,)) as pool:
        for hv, label in pool.imap_unordered(worker, seeds, chunksize=256):
            Xs.append(hv); ys.append(label)
    return np.stack(Xs), np.array(ys, dtype=np.float32), time.perf_counter()-t0


def calib_table(name, X, y, net):
    p = net.predict(X, name)
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    print(f"\n=== {name} (N={len(y)}, pos rate={y.mean():.4f}) ===")
    mae = float(np.abs(p - y).mean()); brier = float(((p - y) ** 2).mean())
    print(f"  MAE={mae:.4f}  Brier={brier:.4f}  mean p̂={p.mean():.4f}")
    print(f"  {'bin':>14}  {'n':>6}  {'pred':>8}  {'actual':>8}  {'Δ':>7}  status")
    max_delta = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        pp = p[m].mean(); pa = y[m].mean(); d = pa - pp
        max_delta = max(max_delta, abs(d))
        bad = "FAIL" if abs(d) > 0.05 and m.sum() >= 30 else ""
        print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>6}  "
              f"{pp:>7.3f}  {pa:>7.3f}  {d:>+6.3f}  {bad}")
    return max_delta


def main():
    print(f"=== Tier 1 calibration check: N={N}/contract ===")
    print(f"  weights: {WEIGHTS.name}", flush=True)
    net = Exp17Pickup.load(WEIGHTS)
    overall_pass = True
    for c in CONTRACT_CONFIGS:
        X, y, wall = gen(c, N)
        print(f"  {c}: gen {wall:.0f}s  pos={int(y.sum())}/{len(y)}", flush=True)
        # Pad trumpless to 36-dim for the wrapper (it'll re-detect, but be explicit)
        max_d = calib_table(c, X, y, net)
        if max_d > 0.05:
            overall_pass = False
            print(f"  ⚠️  {c}: max |Δ| = {max_d:.3f}  > 0.05 threshold")
    print()
    print("=== Tier 1 verdict ===")
    print("PASS" if overall_pass else "PARTIAL (see ⚠️ contracts above)")


if __name__ == "__main__":
    main()
