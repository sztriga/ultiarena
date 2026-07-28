"""Train exp 17 multi-head pickup net on α=0 god-labeled data.

Adapted from experiments/15_vnet_pickup/train_multihead.py:
- Reads from exp 17 dir (α=0 data).
- Saves to multihead_v17.pt.
- Increases steps_per_epoch and slightly more epochs (more data → can afford it).

Usage: python train.py [tag]   (tag defaults to '1M')
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.vnet.pickup import (
    CONTRACT_CONFIGS, MultiHeadPickupNet, pad_to_unified,
    random_suit_perm, apply_suit_perm,
)

EXP_DIR = Path(__file__).parent
EPOCHS  = 30
PER_CONTRACT_BATCH = 256   # batch = 1024 total
LR      = 1e-3
DROPOUT = 0.1
VAL_FRAC = 0.02
SEED    = 0


def data_path(contract: str, tag: str) -> Path:
    return EXP_DIR / f"{contract}_god_alpha0_{tag}.npz"


def weights_path() -> Path:
    return EXP_DIR / "multihead_v17.pt"


def augment_batch(X, has_trump, rng):
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        sigma = random_suit_perm(has_trump, X[i], rng)
        out[i] = apply_suit_perm(X[i], sigma, has_trump)
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) >= 2 else "1M"
    contracts = list(CONTRACT_CONFIGS)

    print("=== Exp 17 multi-head training ===")
    data = {}
    for c in contracts:
        cfg = CONTRACT_CONFIGS[c]
        d = np.load(data_path(c, tag))
        X = pad_to_unified(d['X'], cfg.has_trump)
        y = d['y']
        data[c] = {'X': X, 'y': y, 'has_trump': cfg.has_trump}
        print(f"  {c:>10}: {len(y)} records, {y.mean()*100:.2f}% pos, "
              f"has_trump={cfg.has_trump}")
    print(f"  arch: 36→128→64 body, 64→32→1 per head")
    print(f"  epochs={EPOCHS}  per-contract-batch={PER_CONTRACT_BATCH}  lr={LR}",
          flush=True)

    rng = np.random.default_rng(SEED)
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
    print(f"  total params: {sum(p.numel() for p in model.parameters())}")
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

        model.eval()
        v_briers = {}
        with torch.no_grad():
            for c in contracts:
                vp = model(splits[c]['X_va'], c)
                v_briers[c] = ((vp - splits[c]['y_va']) ** 2).mean().item()
        v_brier_mean = sum(v_briers.values()) / len(v_briers)
        if v_brier_mean < best_val_brier:
            best_val_brier = v_brier_mean
            torch.save(model.state_dict(), weights_path())

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            wall = time.perf_counter() - t0
            losses = "  ".join(f"{c[:3]}={epoch_loss[c]/n_seen[c]:.4f}"
                               for c in contracts)
            briers = "  ".join(f"{c[:3]}={v_briers[c]:.4f}"
                               for c in contracts)
            print(f"  ep {epoch:>3} | tr {losses} | val_brier  {briers} | "
                  f"mean {v_brier_mean:.5f} | {wall:.0f}s", flush=True)

    wall = time.perf_counter() - t0
    print(f"\nBest val_brier (mean): {best_val_brier:.5f}")
    print(f"Wall: {wall:.1f}s")
    print(f"Weights → {weights_path()}")

    # Calibration table on val
    model.load_state_dict(torch.load(weights_path(), weights_only=True))
    model.eval()
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    print()
    for c in contracts:
        with torch.no_grad():
            vp = model(splits[c]['X_va'], c).numpy()
        yv = splits[c]['y_va'].numpy()
        print(f"=== Train-val calibration: {c} ===")
        print(f"  {'bin':>14}  {'n':>6}  {'pred':>8}  {'actual':>8}  {'Δ':>7}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (vp >= lo) & (vp < hi)
            if m.sum() == 0:
                continue
            pp = vp[m].mean(); pa = yv[m].mean()
            print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>6}  "
                  f"{pp:>7.3f}  {pa:>7.3f}  {pa-pp:>+6.3f}")
        print()


if __name__ == "__main__":
    main()
