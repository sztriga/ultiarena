"""V2 trainer: god-labeled binary BCE + suit-perm augmentation.

Usage: python train_v2.py <contract> [n_records_tag]
  contract: one of betli, ulti, parti, durchmars
  n_records_tag: filename tag for the input npz (default '250k')

Loads <contract>_god_<tag>.npz, trains PickupNetV2 (256→128) with BCE
(pos_weight for class imbalance) and on-the-fly suit-permutation
augmentation. Saves weights to <contract>_vnet_v2.pt.
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
    CONTRACT_CONFIGS, PickupNetV2, input_dim,
    random_suit_perm, apply_suit_perm,
)

EXP_DIR  = Path(__file__).parent
EPOCHS   = 40
BATCH    = 512
LR       = 1e-3
HIDDEN   = (256, 128)
DROPOUT  = 0.1
VAL_FRAC = 0.05
SEED     = 0


def god_data_path(contract: str, tag: str) -> Path:
    return EXP_DIR / f"{contract}_god_{tag}.npz"


def v2_weights_path(contract: str) -> Path:
    return EXP_DIR / f"{contract}_vnet_v2.pt"


def augment_batch(X: np.ndarray, has_trump: bool,
                  rng: np.random.Generator) -> np.ndarray:
    """Apply a fresh suit permutation per row."""
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        sigma = random_suit_perm(has_trump, X[i], rng)
        out[i] = apply_suit_perm(X[i], sigma, has_trump)
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CONTRACT_CONFIGS:
        print(f"Usage: train_v2.py <contract> [tag]  "
              f"(contracts: {list(CONTRACT_CONFIGS)})")
        sys.exit(1)
    contract = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) >= 3 else "250k"
    cfg = CONTRACT_CONFIGS[contract]
    in_dim = input_dim(cfg)

    data = np.load(god_data_path(contract, tag))
    X = data['X']; y = data['y']
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    # pos_weight=1 means plain BCE — better calibration; pos_weight=n_neg/n_pos
    # upweights positives (helps minority recall, hurts calibration). Env override
    # lets us flip per contract without code change.
    import os
    pos_weight = float(os.environ.get(
        "POS_WEIGHT", n_neg / max(1, n_pos)
    ))

    print(f"=== Train V2: {contract} ===")
    print(f"  records: {X.shape[0]}  input dim: {X.shape[1]}  has_trump: {cfg.has_trump}")
    print(f"  labels: pos={n_pos} neg={n_neg}  pos_weight={pos_weight:.3f}")
    print(f"  net: {in_dim} → {HIDDEN[0]} → {HIDDEN[1]} → 1  "
          f"(dropout={DROPOUT})")
    print(f"  epochs={EPOCHS} batch={BATCH} lr={LR} val_frac={VAL_FRAC}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(X.shape[0])
    n_val = int(X.shape[0] * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train_np = X[train_idx].copy(); y_train_np = y[train_idx].copy()
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    torch.manual_seed(SEED)
    model = PickupNetV2(in_dim=in_dim, hidden=HIDDEN, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pw = torch.tensor(pos_weight, dtype=torch.float32)
    # BCE with logits would be cleaner but the net uses Sigmoid head;
    # use BCELoss with manual weighting via reduction='none'.
    loss_fn = nn.BCELoss(reduction='none')

    n_train = X_train_np.shape[0]
    aug_rng = np.random.default_rng(SEED + 1)
    t0 = time.perf_counter()
    best_val_brier = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm_ep = np.random.permutation(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm_ep[i:i+BATCH]
            xb = augment_batch(X_train_np[idx], cfg.has_trump, aug_rng)
            xb_t = torch.from_numpy(xb)
            yb_t = torch.from_numpy(y_train_np[idx])
            pred = model(xb_t)
            per_sample = loss_fn(pred, yb_t)
            # weight pos labels by pos_weight, neg by 1
            w = torch.where(yb_t > 0.5, pw, torch.tensor(1.0))
            loss = (per_sample * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * idx.shape[0]
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            vp = model(X_val)
            v_brier = ((vp - y_val) ** 2).mean().item()
            v_acc = ((vp > 0.5).float() == y_val).float().mean().item()
        if v_brier < best_val_brier:
            best_val_brier = v_brier
            torch.save(model.state_dict(), v2_weights_path(contract))
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            wall = time.perf_counter() - t0
            print(f"  epoch {epoch:>3}  train_bce={epoch_loss:.4f}  "
                  f"val_brier={v_brier:.4f}  val_acc={v_acc:.3f}  "
                  f"wall={wall:.0f}s", flush=True)

    wall = time.perf_counter() - t0
    print(f"\nBest val_brier: {best_val_brier:.4f}")
    print(f"Trained in {wall:.1f}s")
    print(f"Weights → {v2_weights_path(contract)}")

    # Reload best and report calibration
    model.load_state_dict(torch.load(v2_weights_path(contract),
                                     weights_only=True))
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
