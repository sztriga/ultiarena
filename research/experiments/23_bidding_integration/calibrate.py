"""Fit per-head isotonic calibration to the DEPLOYMENT distribution (α=0).

The baseline heads are over-confident (class-weighting + biased-α training), so
the composer's combo EVs explode. Isotonic remaps each head's raw output → true
α=0 probability (monotone, so AUC is preserved; only the probabilities move).

Calibration data (held-out α=0):
  • parti/ulti/betli/colorless_duri → exp-17's α=0 1M file, the SAME val split the
    baseline held out (default_rng(12345), first 10%).
  • reach100_40/20, duri_colored     → fresh α=0 sets (gen_base_events, seed 900M).

Saves <head>_isotonic.npz (x grid, y calibrated) for NetProvider(calibrate=True).
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, "/Users/milansimity/Cuccok/kodok/oldtawer")

from train_base_head import Head, ece  # noqa: E402

try:
    from sklearn.isotonic import IsotonicRegression
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False

E17 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/17_clean_pickup_net"
# head → (exp17 source file | None for fresh α=0 set)
SRC17 = {"parti": "parti", "ulti": "ulti", "betli": "betli",
         "colorless_duri": "durchmars"}
NEW = ("reach100_40", "reach100_20", "duri_colored")


def _calib_data(head):
    """Return (X, y) held-out α=0 calibration data for `head`."""
    if head in SRC17:
        d = np.load(f"{E17}/{SRC17[head]}_god_alpha0_1M.npz")
        X, y = d["X"], d["y"]
        rng = np.random.default_rng(12345)            # same as train_base_head
        perm = rng.permutation(len(y))
        vi = perm[:int(len(y) * 0.1)]
        return X[vi], y[vi]
    files = sorted(glob.glob(os.path.join(_HERE, f"{head}_god_a0_*.npz")))
    d = np.load(files[-1])
    return d["X"], d["y"]


def _isotonic(pred, y):
    if _HAVE_SK:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(pred, y)
        grid = np.linspace(0, 1, 201)
        return grid, np.clip(iso.predict(grid), 0, 1)
    # fallback: 50-bin empirical mean (monotonic-ish via cummax)
    edges = np.linspace(0, 1, 51)
    xs, ys = [], []
    for b in range(50):
        m = (pred >= edges[b]) & (pred <= edges[b + 1] if b == 49 else pred < edges[b + 1])
        if m.sum() < 20:
            continue
        xs.append(pred[m].mean()); ys.append(y[m].mean())
    xs, ys = np.array(xs), np.maximum.accumulate(np.array(ys))
    return xs, ys


def main():
    print(f"=== calibrate (sklearn={_HAVE_SK}) ===", flush=True)
    print(f"{'head':<16}{'N':>8}{'ECE_raw':>9}{'ECE_cal':>9}")
    for head in tuple(SRC17) + NEW:
        ckpt = torch.load(os.path.join(_HERE, f"{head}_baseline.pt"), weights_only=False)
        m = Head(ckpt["in_dim"]); m.load_state_dict(ckpt["state_dict"]); m.eval()
        X, y = _calib_data(head)
        with torch.no_grad():
            pred = m(torch.from_numpy(X.astype(np.float32))).numpy()
        x_grid, y_cal = _isotonic(pred, y)
        cal_pred = np.interp(pred, x_grid, y_cal)
        e_raw, _ = ece(y, pred)
        e_cal, _ = ece(y, cal_pred)
        np.savez(os.path.join(_HERE, f"{head}_isotonic.npz"), x=x_grid, y=y_cal)
        print(f"{head:<16}{len(y):>8}{e_raw:>9.4f}{e_cal:>9.4f}", flush=True)
    print("saved <head>_isotonic.npz for all heads")


if __name__ == "__main__":
    main()
