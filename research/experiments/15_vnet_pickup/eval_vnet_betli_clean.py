"""Clean head-to-head: v-net vs PIMC32 on deal_betli hands.

Sample N betli-biased deals, enumerate all 66 discards each, and for
each (hand, discard) compute both the v-net prediction and the
PIMC32 estimate. Report:
  - per-record MAE / Brier
  - timing for v-net (batched) vs PIMC32 batch
  - agreement on the discard sol picks (best EV among the 66)
"""
from __future__ import annotations

import itertools, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from eval.dojo import deal_betli
from solvers import pis, pimc as _pimc
from train_betli import BetliNet

N         = 50
PIMC_N    = 32
ALPHA     = 0.6
SEED_BASE = 700_000_000
N_WORKERS = 4
WEIGHTS   = Path(__file__).parent / "betli_vnet.pt"


def _hand_vec(hand):
    v = np.zeros(32, dtype=np.float32)
    for c in hand:
        v[c.id] = 1.0
    return v


def _load_vnet():
    m = BetliNet()
    m.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    m.eval()
    return m


def pimc_worker(args):
    """Compute PIMC32 P_make for all 66 discards of one deal."""
    seed = args
    deal = deal_betli(seed=seed, alpha=ALPHA)
    sol12 = list(deal.sol_hand) + list(deal.talon)
    d1 = list(deal.def1_hand); d2 = list(deal.def2_hand)
    discards = list(itertools.combinations(sol12, 2))
    out = []
    rng_seed = seed
    for dp in discards:
        rem = [c for c in sol12 if c not in dp]
        talon = list(dp)
        pos = pis.build_position(
            hands=[rem, d1, d2], soloist=0, leader=0,
            contract='betli', trump=None, talon=talon,
        )
        rng_seed += 1
        _, avg = _pimc.pimc_decision(
            true_pos=pos, contract='betli', n_samples=PIMC_N, seed=rng_seed,
        )
        p = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
        out.append((_hand_vec(rem), p, tuple(str(c) for c in dp)))
    return seed, out


def main():
    seeds = [SEED_BASE + i for i in range(N)]
    print(f"=== V-net vs PIMC32 on {N} betli-biased deals ({N*66} records) ===")

    # ── PIMC pass (parallel) ─────────────────────────────────────────
    t0 = time.perf_counter()
    pimc_results = {}
    with Pool(N_WORKERS) as pool:
        for seed, recs in pool.imap_unordered(pimc_worker, seeds):
            pimc_results[seed] = recs
    pimc_wall = time.perf_counter() - t0
    n_total = sum(len(v) for v in pimc_results.values())
    print(f"PIMC32 wall: {pimc_wall:.1f}s   ms/record: {pimc_wall/n_total*1000:.1f}")

    # ── V-net pass (batched, single thread) ──────────────────────────
    vnet = _load_vnet()
    X_all = []; p_pimc_all = []; seed_order = []
    for seed, recs in pimc_results.items():
        for hv, p, _ in recs:
            X_all.append(hv); p_pimc_all.append(p); seed_order.append(seed)
    X_t = torch.from_numpy(np.stack(X_all))
    t1 = time.perf_counter()
    with torch.no_grad():
        p_vnet = vnet(X_t).numpy()
    vnet_wall = time.perf_counter() - t1
    print(f"V-net wall:  {vnet_wall*1000:.1f}ms  ({vnet_wall/n_total*1e6:.1f} μs/record)")
    print(f"Speedup:     {pimc_wall/vnet_wall:.0f}×")

    # ── Accuracy ────────────────────────────────────────────────────
    p_pimc = np.array(p_pimc_all)
    mae = np.abs(p_vnet - p_pimc).mean()
    brier = ((p_vnet - p_pimc)**2).mean()
    print(f"\n=== Accuracy (target = PIMC32) ===")
    print(f"  MAE:    {mae:.4f}")
    print(f"  Brier:  {brier:.4f}")
    print(f"  Pearson r: {np.corrcoef(p_vnet, p_pimc)[0,1]:.4f}")

    # ── Discard agreement: best EV per deal ──────────────────────────
    agree_top = 0
    diff_pmake = []
    for seed in seeds:
        recs = pimc_results[seed]
        idx_pimc = max(range(len(recs)), key=lambda i: recs[i][1])
        # Re-batch this deal's vnet preds
        Xd = np.stack([r[0] for r in recs])
        with torch.no_grad():
            p_v = vnet(torch.from_numpy(Xd)).numpy()
        idx_vnet = int(p_v.argmax())
        if idx_vnet == idx_pimc:
            agree_top += 1
        diff_pmake.append(abs(p_v[idx_pimc] - p_v[idx_vnet]))

    print(f"\n=== Best-discard agreement ===")
    print(f"  v-net and PIMC pick same discard: {agree_top}/{N} ({agree_top/N*100:.0f}%)")
    print(f"  mean |Δp_make| when they disagree: {np.mean(diff_pmake):.4f}")

    # ── P bucket calibration ────────────────────────────────────────
    print()
    print("=== Calibration (bucket by v-net pred) ===")
    print(f"  {'bin':>14}  {'n':>5}  {'mean v-net':>10}  {'mean PIMC':>10}")
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_vnet >= lo) & (p_vnet < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>5}  "
              f"{p_vnet[m].mean():>10.4f}  {p_pimc[m].mean():>10.4f}")


if __name__ == "__main__":
    main()
