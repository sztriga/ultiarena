"""Parameterized trainer: python train.py <contract>.

Trains a small MLP on the .npz data for one contract. Saves weights
to <contract>_vnet.pt.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from _vlib import CONTRACT_CONFIGS, PickupNet, data_path, weights_path, input_dim

EPOCHS   = 200
BATCH    = 256
LR       = 1e-3
HIDDEN   = 64
VAL_FRAC = 0.1
SEED     = 0


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CONTRACT_CONFIGS:
        print(f"Usage: train.py <contract>  (one of {list(CONTRACT_CONFIGS)})")
        sys.exit(1)
    contract = sys.argv[1]
    cfg = CONTRACT_CONFIGS[contract]

    data = np.load(data_path(contract))
    X = data['X']; y = data['y']
    print(f"=== Train {contract} ===")
    print(f"  records: {X.shape[0]}  input dim: {X.shape[1]}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(X.shape[0])
    n_val = int(X.shape[0] * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train = torch.from_numpy(X[train_idx]); y_train = torch.from_numpy(y[train_idx])
    X_val   = torch.from_numpy(X[val_idx]);   y_val   = torch.from_numpy(y[val_idx])

    torch.manual_seed(SEED)
    model = PickupNet(in_dim=input_dim(cfg), hidden=HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n_train = X_train.shape[0]
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS+1):
        model.train()
        perm_ep = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm_ep[i:i+BATCH]
            pred = model(X_train[idx])
            loss = loss_fn(pred, y_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * idx.shape[0]
        epoch_loss /= n_train
        if epoch % 50 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                vp = model(X_val)
                vmae = (vp - y_val).abs().mean().item()
                brier = ((vp - y_val)**2).mean().item()
            print(f"  epoch {epoch:>3}  train_mse={epoch_loss:.4f}  "
                  f"val_mae={vmae:.4f}  brier={brier:.4f}")
    wall = time.perf_counter() - t0
    print(f"Trained in {wall:.1f}s")
    torch.save(model.state_dict(), weights_path(contract))
    print(f"Weights → {weights_path(contract)}")

    # Calibration table on val
    model.eval()
    with torch.no_grad():
        vp = model(X_val).numpy()
    yv = y_val.numpy()
    print()
    print("=== Calibration (val, bucket by v-net pred) ===")
    print(f"  {'bin':>14}  {'n':>5}  {'pred':>8}  {'actual':>8}  {'Δ':>7}")
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vp >= lo) & (vp < hi)
        if m.sum() == 0:
            continue
        p_avg = vp[m].mean(); a_avg = yv[m].mean()
        print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>5}  "
              f"{p_avg:>7.3f}  {a_avg:>7.3f}  {a_avg-p_avg:>+6.3f}")


if __name__ == "__main__":
    main()
