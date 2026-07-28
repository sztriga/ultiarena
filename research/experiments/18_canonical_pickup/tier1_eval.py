"""Tier 1 — calibration of all exp 18 variants + exp 17 baseline.

Generates (once, then caches to npz) the same eval set exp 17's Tier 1
used: 50k fresh α=0 deals/contract at seeds 900M+, god-labeled. Then
scores every available net on the identical set.

Pass criteria per net: no bin with n≥30 and |predicted − actual| > 0.05.

Usage: python tier1_eval.py [--gen-only]
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.vnet.pickup import CONTRACT_CONFIGS, featurize
from ulti.vnet.pickup.v17 import Exp17Pickup
from ulti.vnet.pickup.v18 import Exp18Pickup

EXP_DIR   = Path(__file__).parent
EXP17_DIR = EXP_DIR.parent / "17_clean_pickup_net"

N         = 50_000
SEED_BASE = 900_000_000   # same as exp 17 tier1: disjoint from train (8e8) and eval (1e5)
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


def eval_data_path(contract: str) -> Path:
    return EXP_DIR / f"tier1_eval_{contract}_50k.npz"


def ensure_eval_data(contract: str):
    path = eval_data_path(contract)
    if path.exists():
        d = np.load(path)
        return d['X'], d['y']
    print(f"  [gen] {contract}: {N} deals at seeds {SEED_BASE}+ ...", flush=True)
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    Xs = []; ys = []
    with Pool(N_WORKERS, initializer=_init_worker,
              initargs=(contract,)) as pool:
        for hv, label in pool.imap_unordered(worker, seeds, chunksize=256):
            Xs.append(hv); ys.append(label)
    X = np.stack(Xs); y = np.array(ys, dtype=np.float32)
    np.savez_compressed(path, X=X, y=y)
    print(f"  [gen] {contract}: done in {time.perf_counter()-t0:.0f}s  "
          f"pos={int(y.sum())}/{len(y)}", flush=True)
    return X, y


def calib_table(tag, name, X, y, net):
    p = net.predict(X, name)
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    print(f"\n--- {tag} / {name} (N={len(y)}, pos rate={y.mean():.4f}) ---")
    mae = float(np.abs(p - y).mean()); brier = float(((p - y) ** 2).mean())
    print(f"  MAE={mae:.4f}  Brier={brier:.4f}  mean p̂={p.mean():.4f}")
    print(f"  {'bin':>14}  {'n':>6}  {'pred':>8}  {'actual':>8}  {'Δ':>7}  status")
    max_delta = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        pp = p[m].mean(); pa = y[m].mean(); d = pa - pp
        if m.sum() >= 30:
            max_delta = max(max_delta, abs(d))
        bad = "FAIL" if abs(d) > 0.05 and m.sum() >= 30 else ""
        print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>6}  "
              f"{pp:>7.3f}  {pa:>7.3f}  {d:>+6.3f}  {bad}")
    return max_delta


def main():
    gen_only = '--gen-only' in sys.argv
    print(f"=== Exp 18 Tier 1: N={N}/contract, seeds {SEED_BASE}+ ===", flush=True)

    data = {}
    for c in CONTRACT_CONFIGS:
        data[c] = ensure_eval_data(c)
    if gen_only:
        print("Eval data cached; exiting (--gen-only).")
        return

    nets = {}
    v17_w = EXP17_DIR / "multihead_v17.pt"
    if v17_w.exists():
        nets['v17'] = Exp17Pickup.load(v17_w)
    for variant in ('a', 'b', 'c'):
        w = EXP_DIR / f"multihead_v18{variant}.pt"
        if w.exists():
            loader = Exp17Pickup if variant == 'b' else Exp18Pickup
            nets[f'v18{variant}'] = loader.load(w)
    print(f"  nets: {list(nets)}", flush=True)

    summary = {}
    for tag, net in nets.items():
        worst = {}
        for c in CONTRACT_CONFIGS:
            X, y = data[c]
            worst[c] = calib_table(tag, c, X, y, net)
        summary[tag] = worst

    print("\n=== Summary: max |Δ| over bins with n≥30 ===")
    cs = list(CONTRACT_CONFIGS)
    print(f"  {'net':>6}  " + "  ".join(f"{c:>10}" for c in cs) + "   verdict")
    for tag, worst in summary.items():
        verdict = "PASS" if all(v <= 0.05 for v in worst.values()) else "FAIL"
        print(f"  {tag:>6}  " +
              "  ".join(f"{worst[c]:>10.3f}" for c in cs) + f"   {verdict}")


if __name__ == "__main__":
    main()
