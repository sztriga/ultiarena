"""Train the betli + durchmars structural pickup nets (exp 19).

Trains on the biased god data (exp 15 *_god_250k.npz, 35-48% positive).
Structure is distribution-invariant, so the biased-trained model is
checked for calibration on the α=0 deployment distribution (cached 50k
sets from exp 18 Tier 1).

Usage: python train_colorless.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from vnet.pickup.colorless import (
    colorless_features, ColorlessStructNet, N_FEATURES,
)

EXP15 = Path(__file__).parent.parent / "15_vnet_pickup"
EXP17 = Path(__file__).parent.parent / "17_clean_pickup_net"
EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent

EPOCHS = 40
BATCH  = 1024
LR     = 1e-3
VAL_FRAC = 0.05
SEED   = 0


def weights_path(contract):
    return EXP19 / f"colorless_{contract}.pt"


def calib_table(tag, p, y):
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    brier = float(((p - y) ** 2).mean())
    mae = float(np.abs(p - y).mean())
    print(f"  --- {tag}: N={len(y)} pos={y.mean():.4f} "
          f"Brier={brier:.4f} MAE={mae:.4f} ---")
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 30:
            continue
        pp = p[m].mean(); pa = y[m].mean(); d = pa - pp
        worst = max(worst, abs(d))
        flag = "  <-- off" if abs(d) > 0.05 else ""
        print(f"    [{lo:.2f},{hi:.2f})  n={m.sum():>6}  "
              f"pred={pp:.3f}  actual={pa:.3f}  Δ={d:+.3f}{flag}")
    return brier, worst


def train_one(contract):
    print(f"\n===== training colorless net: {contract} =====", flush=True)
    # Train on the α=0 DEPLOYMENT distribution (same data v18a used).
    # Biased data calibrates within-distribution but overshoots on α=0
    # because the biased dealer correlates the hand with favourable
    # opponent splits — structure is not a sufficient statistic across
    # split distributions. The structural model's edge over v18a is its
    # low capacity (resists argmax overconfidence), not the biased data.
    d = np.load(EXP17 / f"{contract}_god_alpha0_1M.npz")
    X, y = d['X'], d['y'].astype(np.float32)
    F = colorless_features(X)
    print(f"  α=0 data: {len(y)} hands, {y.mean()*100:.3f}% pos "
          f"({int(y.sum())} positives), {F.shape[1]} features", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    nv = int(len(y) * VAL_FRAC)
    va, tr = perm[:nv], perm[nv:]
    mean = F[tr].mean(0); std = F[tr].std(0) + 1e-6
    Fn = (F - mean) / std

    Xtr = torch.from_numpy(Fn[tr]); ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy(Fn[va]); yva = y[va]

    torch.manual_seed(SEED)
    model = ColorlessStructNet(in_dim=N_FEATURES)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()
    n = len(tr)
    t0 = time.perf_counter()
    best_brier = float('inf')
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            pred = model(Xtr[idx])
            loss = loss_fn(pred, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(Xva).numpy()
        brier = float(((pv - yva) ** 2).mean())
        if brier < best_brier:
            best_brier = brier
            torch.save({'state_dict': model.state_dict(),
                        'in_dim': N_FEATURES,
                        'mean': mean.astype(float).tolist(),
                        'std': std.astype(float).tolist()},
                       weights_path(contract))
        if ep == 1 or ep % 10 == 0 or ep == EPOCHS:
            print(f"  ep {ep:>3}  train_loss={tot/n:.4f}  "
                  f"val_brier={brier:.4f}  {time.perf_counter()-t0:.0f}s",
                  flush=True)

    # reload best, report calibration on biased-val + α=0 deployment set
    ckpt = torch.load(weights_path(contract), weights_only=True)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    with torch.no_grad():
        pv = model(Xva).numpy()
    print(f"  best α=0 val_brier={best_brier:.4f}  "
          f"(predicted >0.5 on {int((pv>0.5).sum())}/{len(pv)} val hands)")
    calib_table("α=0 held-out val (from 1M)", pv, yva)

    a0 = EXP18 / f"tier1_eval_{contract}_50k.npz"
    if a0.exists():
        da = np.load(a0)
        Fa = (colorless_features(da['X']) - mean) / std
        with torch.no_grad():
            pa = model(torch.from_numpy(Fa.astype(np.float32))).numpy()
        calib_table("α=0 50k disjoint held-out (seeds 900M+)", pa,
                    da['y'].astype(np.float32))
    print(f"  weights → {weights_path(contract).name}")


def main():
    for c in ('durchmars', 'betli'):
        train_one(c)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
