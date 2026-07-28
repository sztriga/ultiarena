"""Train the multi-head pickup-net on combined god-labeled data.

All 4 contracts share a 36 → 128 → 64 body and route to per-contract
32 → 1 heads. Balanced sampling per contract per batch keeps the body
from being dominated by any one contract.

Usage: python train_multihead.py [tag]   (tag defaults to '250k')
Saves weights to multihead_v1.pt.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from vnet.pickup import (
    CONTRACT_CONFIGS, MultiHeadPickupNet, pad_to_unified, UNIFIED_DIM,
    random_suit_perm, apply_suit_perm,
)

EXP_DIR        = Path(__file__).parent
EPOCHS         = 40
PER_CONTRACT_BATCH = 128   # records per contract per step → batch=512 total
LR             = 1e-3
DROPOUT        = 0.1
VAL_FRAC       = 0.05
SEED           = 0


def god_data_path(contract: str, tag: str) -> Path:
    return EXP_DIR / f"{contract}_god_{tag}.npz"


def multihead_weights_path() -> Path:
    return EXP_DIR / "multihead_v1.pt"


def augment_batch(X: np.ndarray, has_trump: bool,
                  rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        sigma = random_suit_perm(has_trump, X[i], rng)
        out[i] = apply_suit_perm(X[i], sigma, has_trump)
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) >= 2 else "250k"
    contracts = list(CONTRACT_CONFIGS)

    # Load all contracts. For trumpless, pad to 36-dim.
    data = {}
    for c in contracts:
        cfg = CONTRACT_CONFIGS[c]
        d = np.load(god_data_path(c, tag))
        X = pad_to_unified(d['X'], cfg.has_trump)
        y = d['y']
        data[c] = {'X': X, 'y': y, 'has_trump': cfg.has_trump}

    print("=== Multi-head training ===")
    for c in contracts:
        n = len(data[c]['y'])
        npos = int(data[c]['y'].sum())
        print(f"  {c:>10}: {n} records, {npos/n*100:.1f}% positive, "
              f"has_trump={data[c]['has_trump']}")
    print(f"  arch: 36 → 128 → 64  body, 64 → 32 → 1 per head")
    print(f"  epochs={EPOCHS}  per-contract-batch={PER_CONTRACT_BATCH}  "
          f"lr={LR}  val_frac={VAL_FRAC}")

    rng = np.random.default_rng(SEED)
    # Per-contract train/val split
    splits = {}
    for c in contracts:
        n = len(data[c]['y'])
        perm = rng.permutation(n)
        nv = int(n * VAL_FRAC)
        val_idx, train_idx = perm[:nv], perm[nv:]
        splits[c] = {
            'X_tr': data[c]['X'][train_idx].copy(),
            'y_tr': data[c]['y'][train_idx].copy(),
            'X_va': torch.from_numpy(data[c]['X'][val_idx]),
            'y_va': torch.from_numpy(data[c]['y'][val_idx]),
            'has_trump': data[c]['has_trump'],
        }

    torch.manual_seed(SEED)
    model = MultiHeadPickupNet(dropout=DROPOUT, contracts=contracts)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    aug_rng = np.random.default_rng(SEED + 1)
    n_train = {c: len(splits[c]['y_tr']) for c in contracts}
    steps_per_epoch = min(n_train.values()) // PER_CONTRACT_BATCH

    t0 = time.perf_counter()
    best_val_brier = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perms = {c: rng.permutation(n_train[c]) for c in contracts}
        epoch_loss = {c: 0.0 for c in contracts}
        n_seen = {c: 0 for c in contracts}
        for step in range(steps_per_epoch):
            total_loss = 0.0
            for c in contracts:
                idx = perms[c][step * PER_CONTRACT_BATCH:
                               (step + 1) * PER_CONTRACT_BATCH]
                xb = augment_batch(splits[c]['X_tr'][idx],
                                   splits[c]['has_trump'], aug_rng)
                xb_t = torch.from_numpy(xb)
                yb_t = torch.from_numpy(splits[c]['y_tr'][idx])
                pred = model(xb_t, c)
                loss = loss_fn(pred, yb_t)
                total_loss = total_loss + loss
                epoch_loss[c] += loss.item() * len(idx)
                n_seen[c] += len(idx)
            opt.zero_grad(); total_loss.backward(); opt.step()

        # Val per contract
        model.eval()
        v_briers = {}; v_accs = {}
        with torch.no_grad():
            for c in contracts:
                vp = model(splits[c]['X_va'], c)
                v_briers[c] = ((vp - splits[c]['y_va']) ** 2).mean().item()
                v_accs[c]   = ((vp > 0.5).float() == splits[c]['y_va']).float().mean().item()
        v_brier_mean = sum(v_briers.values()) / len(v_briers)
        if v_brier_mean < best_val_brier:
            best_val_brier = v_brier_mean
            torch.save(model.state_dict(), multihead_weights_path())

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            wall = time.perf_counter() - t0
            losses = "  ".join(f"{c[:3]}={epoch_loss[c]/n_seen[c]:.3f}"
                               for c in contracts)
            briers = "  ".join(f"{c[:3]}={v_briers[c]:.3f}"
                               for c in contracts)
            print(f"  ep {epoch:>3} | tr {losses} | val_brier  {briers} | "
                  f"mean {v_brier_mean:.4f} | {wall:.0f}s",
                  flush=True)

    wall = time.perf_counter() - t0
    print(f"\nBest val_brier (mean): {best_val_brier:.4f}")
    print(f"Trained in {wall:.1f}s")
    print(f"Weights → {multihead_weights_path()}")

    # Reload best and dump per-contract calibration
    model.load_state_dict(torch.load(multihead_weights_path(),
                                     weights_only=True))
    model.eval()
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    print()
    for c in contracts:
        with torch.no_grad():
            vp = model(splits[c]['X_va'], c).numpy()
        yv = splits[c]['y_va'].numpy()
        print(f"=== Calibration: {c} ===")
        print(f"  {'bin':>14}  {'n':>5}  {'pred':>8}  {'actual':>8}  {'Δ':>7}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (vp >= lo) & (vp < hi)
            if m.sum() == 0:
                continue
            pp = vp[m].mean(); pa = yv[m].mean()
            print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>5}  "
                  f"{pp:>7.3f}  {pa:>7.3f}  {pa-pp:>+6.3f}")
        print()


if __name__ == "__main__":
    main()
