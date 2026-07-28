"""Unified v-net vs PIMC32 head-to-head across all 4 contracts.

For each contract, sample N biased deals, label all 66 discards via
PIMC32, compare to v-net predictions. Report per-contract timing,
calibration, and top-1 discard agreement.
"""
from __future__ import annotations

import itertools, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from solvers import pis, pimc as _pimc
from _vlib import CONTRACT_CONFIGS, PickupNet, featurize, input_dim, weights_path

N         = 50
PIMC_N    = 32
ALPHA     = 0.6
SEED_BASE = 700_000_000
N_WORKERS = 4

_WORKER_CONTRACT = None


def _init_worker(contract_name: str):
    global _WORKER_CONTRACT
    _WORKER_CONTRACT = contract_name


def worker(seed: int):
    cfg = CONTRACT_CONFIGS[_WORKER_CONTRACT]
    deal = cfg.dealer(seed=seed, alpha=ALPHA)
    sol12 = list(deal.sol_hand) + list(deal.talon)
    d1 = list(deal.def1_hand); d2 = list(deal.def2_hand)
    trump = deal.trump if cfg.has_trump else None
    out = []
    rng_seed = seed
    for dp in itertools.combinations(sol12, 2):
        rem = [c for c in sol12 if c not in dp]
        talon = list(dp)
        pos = pis.build_position(
            hands=[rem, d1, d2], soloist=0, leader=0,
            contract=cfg.solver, trump=trump, talon=talon,
        )
        rng_seed += 1
        _, avg = _pimc.pimc_decision(
            true_pos=pos, contract=cfg.solver, n_samples=PIMC_N, seed=rng_seed,
        )
        p = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
        out.append((featurize(rem, trump, cfg.has_trump), p))
    return seed, out


def run_one_contract(name: str):
    cfg = CONTRACT_CONFIGS[name]
    seeds = [SEED_BASE + i for i in range(N)]

    # PIMC pass
    t0 = time.perf_counter()
    results = {}
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(name,)) as pool:
        for seed, recs in pool.imap_unordered(worker, seeds):
            results[seed] = recs
    pimc_wall = time.perf_counter() - t0

    # V-net pass
    net = PickupNet(in_dim=input_dim(cfg))
    net.load_state_dict(torch.load(weights_path(name), weights_only=True))
    net.eval()
    X_all = []; p_pimc = []
    for seed in seeds:
        for hv, p in results[seed]:
            X_all.append(hv); p_pimc.append(p)
    X_t = torch.from_numpy(np.stack(X_all))
    t1 = time.perf_counter()
    with torch.no_grad():
        p_vnet = net(X_t).numpy()
    vnet_wall = time.perf_counter() - t1
    p_pimc = np.array(p_pimc)

    # Top-1 discard agreement
    agree = 0
    for seed in seeds:
        recs = results[seed]
        idx_p = max(range(len(recs)), key=lambda i: recs[i][1])
        Xd = np.stack([r[0] for r in recs])
        with torch.no_grad():
            pv = net(torch.from_numpy(Xd)).numpy()
        if int(pv.argmax()) == idx_p:
            agree += 1

    return {
        'contract': name,
        'n_records': len(p_pimc),
        'pimc_wall': pimc_wall,
        'vnet_wall': vnet_wall,
        'mae':   float(np.abs(p_vnet - p_pimc).mean()),
        'brier': float(((p_vnet - p_pimc)**2).mean()),
        'r':     float(np.corrcoef(p_vnet, p_pimc)[0,1]),
        'agree_top1': agree,
        'agree_pct': agree / N * 100,
        'p_vnet': p_vnet, 'p_pimc': p_pimc,
    }


def main():
    rows = []
    print(f"=== V-net vs PIMC32 across 4 contracts (N={N} deals each) ===\n")
    for name in ('betli', 'durchmars', 'parti', 'ulti'):
        print(f"--- {name} ---")
        r = run_one_contract(name)
        rows.append(r)
        print(f"  records={r['n_records']}  PIMC wall={r['pimc_wall']:.1f}s  "
              f"v-net wall={r['vnet_wall']*1000:.1f}ms  "
              f"speedup={r['pimc_wall']/r['vnet_wall']:.0f}×")
        print(f"  MAE={r['mae']:.4f}  Brier={r['brier']:.4f}  r={r['r']:.3f}  "
              f"top-1 discard agree={r['agree_top1']}/{N} ({r['agree_pct']:.0f}%)")
        print()

    # Unified summary
    print("=== SUMMARY ===")
    print(f"  {'contract':>10}  {'records':>8}  {'PIMC (s)':>9}  "
          f"{'vnet (ms)':>10}  {'speedup':>9}  {'MAE':>6}  {'Brier':>6}  "
          f"{'r':>5}  {'top-1':>6}")
    for r in rows:
        print(f"  {r['contract']:>10}  {r['n_records']:>8}  "
              f"{r['pimc_wall']:>8.1f}s  {r['vnet_wall']*1000:>8.1f}ms  "
              f"{r['pimc_wall']/r['vnet_wall']:>7.0f}×  "
              f"{r['mae']:>6.3f}  {r['brier']:>6.3f}  "
              f"{r['r']:>5.2f}  {r['agree_pct']:>5.0f}%")

    # Per-contract calibration table
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    print()
    print("=== Calibration per contract ===")
    for r in rows:
        print(f"\n--- {r['contract']} ---")
        print(f"  {'bin':>14}  {'n':>5}  {'pred':>7}  {'actual':>7}  {'Δ':>7}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (r['p_vnet'] >= lo) & (r['p_vnet'] < hi)
            if m.sum() == 0:
                continue
            pp = r['p_vnet'][m].mean(); pa = r['p_pimc'][m].mean()
            print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>5}  "
                  f"{pp:>7.3f}  {pa:>7.3f}  {pa-pp:>+6.3f}")


if __name__ == "__main__":
    main()
