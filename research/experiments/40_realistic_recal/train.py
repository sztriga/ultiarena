"""exp40 — train a REALISTIC-defense make-prob head (any base event) + isotonic
calibration, and compare it head-to-head with the deployed GOD head on identical inputs.

Reuses `train_base_head.Head` and the same class-weighted BCE, so `<head>_real_baseline.pt`
is a byte-compatible drop-in for `NetProvider` (loads via `Head(in_dim)`). Saves
`<head>_real_isotonic.npz` (fit on the held-out val slice) — the RELIABILITY calibration.

The report answers "does modelling realistic defense change this head's valuation?":
  • realistic head: val AUC / Brier / ECE_raw / ECE_cal against the REALISTIC outcome y_real.
  • god head (deployed <head>_baseline.pt): its predictions vs the same y_real — it should
    systematically MIS-predict (it only sees perfect defense).
  • gap recovery: on dd-lost hands (y_god=0) that were realistically MADE (y_real=1), what
    probability does each head assign?  god → ~0 (blind); realistic → high.

Env: HEAD (required), DATA (<head>_real.npz), EPOCHS (40), BATCH (1024), LR (1e-3), VAL (0.1).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for _p in (_HERE, _EXP23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_base_head import Head, auc, ece   # noqa: E402

try:
    from sklearn.isotonic import IsotonicRegression
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

HEAD   = os.environ["HEAD"]
DATA   = os.environ.get("DATA") or os.path.join(_HERE, f"{HEAD}_real.npz")
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH  = int(os.environ.get("BATCH", "1024"))
LR     = float(os.environ.get("LR", "1e-3"))
VAL    = float(os.environ.get("VAL", "0.1"))
SEED   = 12345

# The deployed god head to compare against (provider head name → file).
_GOD_FILE = {
    "parti": "parti", "ulti": "ulti", "reach100_40": "reach100_40",
    "reach100_20": "reach100_20", "duri_colored": "duri_colored",
    "colorless_duri": "colorless_duri",
}


def _isotonic(pred, y):
    if _HAVE_SK:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pred, y)
        grid = np.linspace(0, 1, 201)
        return grid, np.clip(iso.predict(grid), 0, 1)
    edges = np.linspace(0, 1, 51); xs, ys = [], []
    for b in range(50):
        m = (pred >= edges[b]) & (pred <= edges[b + 1] if b == 49 else pred < edges[b + 1])
        if m.sum() >= 20:
            xs.append(pred[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.maximum.accumulate(np.array(ys))


def _reliability(pred, y, nb=10):
    """Reliability table: predicted-prob bucket → (mean_pred, empirical_rate, n)."""
    edges = np.linspace(0, 1, nb + 1)
    rows = []
    for b in range(nb):
        m = (pred >= edges[b]) & (pred < edges[b + 1] if b < nb - 1 else pred <= edges[b + 1])
        if m.sum():
            rows.append((edges[b], edges[b + 1], float(pred[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def main():
    d = np.load(DATA)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.float32)          # realistic outcome
    y_god = d["y_god"].astype(np.float32)
    print(f"=== exp40 train realistic head: {HEAD} ===  data={os.path.basename(DATA)}  N={len(y)}  dim={X.shape[1]}",
          flush=True)
    print(f"  realistic make-rate={y.mean():.4f}  dd-lost={1 - y_god.mean():.4f}  "
          f"dd-lost&made={float(((y == 1) & (y_god == 0)).mean()):.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y)); nv = int(len(y) * VAL)
    vi, ti = perm[:nv], perm[nv:]
    Xtr, ytr = torch.from_numpy(X[ti]), torch.from_numpy(y[ti])
    Xva, yva = torch.from_numpy(X[vi]), torch.from_numpy(y[vi])

    torch.manual_seed(SEED)
    model = Head(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_w = min(float((ytr == 0).sum() / max(1, (ytr == 1).sum())), 20.0)
    print(f"  pos_weight={pos_w:.2f}  train={len(ti)}  val={len(vi)}", flush=True)

    n = len(ti); best = {"brier": float("inf")}
    for ep in range(1, EPOCHS + 1):
        model.train(); order = torch.randperm(n); tot = 0.0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]; xb, yb = Xtr[idx], ytr[idx]
            p = model(xb).clamp(1e-6, 1 - 1e-6)
            w = torch.where(yb > 0.5, pos_w, 1.0)
            loss = (-(w * (yb * torch.log(p) + (1 - yb) * torch.log(1 - p)))).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = model(Xva).numpy()
        yv = yva.numpy(); br = float(((pv - yv) ** 2).mean()); a = auc(yv, pv)
        if br < best["brier"]:
            best = {"brier": br, "auc": a, "ep": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:>3}  loss={tot / n:.4f}  val_brier={br:.4f}  val_auc={a:.4f}", flush=True)

    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        pv = model(Xva).numpy()
    yv = yva.numpy(); yg = y_god[vi]
    e_raw, _ = ece(yv, pv)
    x_grid, y_cal = _isotonic(pv, yv)
    pv_cal = np.interp(pv, x_grid, y_cal)
    e_cal, _ = ece(yv, pv_cal)

    out = os.path.join(_HERE, f"{HEAD}_real_baseline.pt")
    torch.save({"state_dict": best["state"], "in_dim": X.shape[1], "head": f"{HEAD}_real",
                "data": os.path.basename(DATA), "base_rate": float(y.mean())}, out)
    np.savez(os.path.join(_HERE, f"{HEAD}_real_isotonic.npz"), x=x_grid, y=y_cal)
    print(f"\n  REALISTIC head  best ep{best['ep']}  val_auc={best['auc']:.4f}  "
          f"val_brier={best['brier']:.4f}  ECE_raw={e_raw:.4f}  ECE_cal={e_cal:.4f}", flush=True)

    # ── RELIABILITY (calibrated) — predicted vs empirical make-rate per bucket ──
    print("  reliability (calibrated):  [lo,hi)  mean_pred  emp_rate   n", flush=True)
    for lo, hi, mp, er, nn in _reliability(pv_cal, yv):
        flag = "" if abs(mp - er) < 0.06 else "  <-- off"
        print(f"    [{lo:.1f},{hi:.1f})   {mp:.3f}     {er:.3f}   {nn:5d}{flag}", flush=True)

    # ── compare to the deployed GOD head on the SAME val inputs ──
    gf = _GOD_FILE.get(HEAD)
    gpath = os.path.join(_EXP23, f"{gf}_baseline.pt")
    if gf and os.path.exists(gpath):
        gck = torch.load(gpath, weights_only=False)
        gm = Head(gck["in_dim"]); gm.load_state_dict(gck["state_dict"]); gm.eval()
        with torch.no_grad():
            pg = gm(Xva).numpy()
        gi = os.path.join(_EXP23, f"{gf}_isotonic.npz")
        if os.path.exists(gi):
            gd = np.load(gi); pg = np.interp(pg, gd["x"], gd["y"])
        br_g = float(((pg - yv) ** 2).mean()); a_g = auc(yv, pg)
        direc = "UNDER" if pg.mean() < yv.mean() else "over"
        print(f"\n  GOD head '{gf}' (vs realistic outcome)  val_auc={a_g:.4f}  val_brier={br_g:.4f}  "
              f"mean_pred={pg.mean():.4f} vs actual make-rate {yv.mean():.4f}  → {direc}-predicts by "
              f"{abs(pg.mean() - yv.mean()):.3f}", flush=True)
        mask = (yg == 0) & (yv == 1)
        if mask.sum():
            print(f"  GAP hands (dd-lost but realistically MADE): n={int(mask.sum())} in val", flush=True)
            print(f"    god head mean p      = {pg[mask].mean():.3f}", flush=True)
            print(f"    realistic head mean p= {pv_cal[mask].mean():.3f}", flush=True)
    else:
        print(f"\n  (no deployed god head '{gf}' found for comparison)", flush=True)
    print(f"\n  saved {HEAD}_real_baseline.pt + {HEAD}_real_isotonic.npz", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
