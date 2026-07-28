"""Train a baseline value head for ONE new base event (reach100_40 / reach100_20
/ duri_colored) on its god-labeled dataset.

Standalone (one small MLP per head) so the data's learnability + baseline quality
is measurable WITHOUT committing to a unified-multihead architecture — that fold
is milan's call (see PLAN.md design forks). Reports val AUC / Brier / calibration
(ECE), and saves <head>_baseline.pt.

Net: 36 (hand+trump) → 128 → 64 → 32 → 1 sigmoid  (matches the exp-17 head shape).

Env: HEAD, EPOCHS (25), BATCH (1024), LR (1e-3), VAL (0.1).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))

HEAD   = os.environ.get("HEAD", "reach100_40")
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BATCH  = int(os.environ.get("BATCH", "1024"))
LR     = float(os.environ.get("LR", "1e-3"))
VAL    = float(os.environ.get("VAL", "0.1"))
SEED   = 12345


class Head(nn.Module):
    def __init__(self, in_dim=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def auc(y, p):
    """Mann-Whitney AUC (rank-based), robust to ties."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    rows = []
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        e += (m.sum() / len(y)) * abs(conf - acc)
        rows.append((edges[b], edges[b + 1], int(m.sum()), conf, acc))
    return e, rows


def main():
    data_override = os.environ.get("DATA", "")
    if data_override:
        path = data_override
    else:
        files = sorted(glob.glob(os.path.join(_HERE, f"{HEAD}_god_a*_*.npz")))
        if not files:
            print(f"[train] no dataset for HEAD={HEAD} yet (waiting on datagen).")
            return 1
        path = files[-1]
    d = np.load(path)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)
    print(f"=== train {HEAD} ===  data={os.path.basename(path)}  "
          f"N={len(y)}  base_rate={y.mean():.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    nv = int(len(y) * VAL)
    vi, ti = perm[:nv], perm[nv:]
    Xtr, ytr = torch.from_numpy(X[ti]), torch.from_numpy(y[ti])
    Xva, yva = torch.from_numpy(X[vi]), torch.from_numpy(y[vi])

    torch.manual_seed(SEED)
    model = Head(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # class-weighted BCE (positives are rare for 20-100 / duri)
    pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    pos_w = min(pos_w, 20.0)
    print(f"  pos_weight={pos_w:.2f}  train={len(ti)}  val={len(vi)}", flush=True)

    n = len(ti)
    best = {"brier": float("inf")}
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = Xtr[idx], ytr[idx]
            p = model(xb).clamp(1e-6, 1 - 1e-6)
            w = torch.where(yb > 0.5, pos_w, 1.0)
            loss = (-(w * (yb * torch.log(p) + (1 - yb) * torch.log(1 - p)))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(Xva).numpy()
        yv = yva.numpy()
        br = float(((pv - yv) ** 2).mean())
        a = auc(yv, pv)
        if br < best["brier"]:
            best = {"brier": br, "auc": a, "ep": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:>3}  loss={tot/n:.4f}  val_brier={br:.4f}  val_auc={a:.4f}",
                  flush=True)

    model.load_state_dict(best["state"])
    with torch.no_grad():
        pv = model(Xva).numpy()
    yv = yva.numpy()
    e, rows = ece(yv, pv)
    out = os.path.join(_HERE, f"{HEAD}_baseline.pt")
    torch.save({"state_dict": best["state"], "in_dim": X.shape[1],
                "head": HEAD, "data": os.path.basename(path),
                "base_rate": float(y.mean())}, out)
    print(f"\n  BEST ep{best['ep']}  val_auc={best['auc']:.4f}  "
          f"val_brier={best['brier']:.4f}  ECE={e:.4f}", flush=True)
    print("  calibration (conf → acc):")
    for lo, hi, cnt, conf, acc in rows:
        print(f"    [{lo:.1f},{hi:.1f})  n={cnt:>6}  conf={conf:.3f}  acc={acc:.3f}")
    print(f"  saved {os.path.basename(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
