"""Train a small MLP on (32-dim hand vec → P_make_betli).

Tiny model, plain MSE loss, no GPU, no fancy schedulers. Smoke training
for the v-net pickup proof of concept.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA_PATH = Path(__file__).parent / "betli_data_10k.npz"
WEIGHTS   = Path(__file__).parent / "betli_vnet.pt"

EPOCHS     = 200
BATCH      = 256
LR         = 1e-3
HIDDEN     = 64
VAL_FRAC   = 0.1
SEED       = 0


class BetliNet(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    rng = np.random.default_rng(SEED)
    data = np.load(DATA_PATH)
    X = data['X']; y = data['y']
    print(f"Loaded {X.shape[0]} records (input dim {X.shape[1]})")

    # train/val split
    perm = rng.permutation(X.shape[0])
    n_val = int(X.shape[0] * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val   = torch.from_numpy(X[val_idx])
    y_val   = torch.from_numpy(y[val_idx])

    torch.manual_seed(SEED)
    model = BetliNet()
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
            xb, yb = X_train[idx], y_train[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n_train

        if epoch % 20 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                vp = model(X_val)
                vloss = loss_fn(vp, y_val).item()
                vmae  = (vp - y_val).abs().mean().item()
                brier = ((vp - y_val)**2).mean().item()
            print(f"  epoch {epoch:>3}  train_mse={epoch_loss:.4f}  "
                  f"val_mse={vloss:.4f}  val_mae={vmae:.4f}  brier={brier:.4f}")

    wall = time.perf_counter() - t0
    print(f"Trained in {wall:.1f}s")

    torch.save(model.state_dict(), WEIGHTS)
    print(f"Weights → {WEIGHTS}")

    # Final calibration: bucket val predictions by P bin, report actual
    model.eval()
    with torch.no_grad():
        vp = model(X_val).numpy()
    print()
    print("=== Calibration (val set, bucket by prediction) ===")
    print(f"  {'pred bin':>14}  {'n':>5}  {'mean_pred':>9}  {'mean_actual':>11}")
    edges = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    yv = y_val.numpy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (vp >= lo) & (vp < hi)
        n = mask.sum()
        if n == 0:
            continue
        print(f"  [{lo:.2f},{hi:.2f})   {n:>5}  {vp[mask].mean():>9.3f}  {yv[mask].mean():>11.3f}")


if __name__ == "__main__":
    main()
