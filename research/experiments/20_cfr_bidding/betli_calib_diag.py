"""Diagnose WHERE the betli/duri miscalibration lives before "fixing" it.

(1) Net calibration on RANDOM α=0 hands (held-out): is the raw net over- or
    under-confident? If under/calibrated, recalibrating it would BACKFIRE.
(2) For contrast we already know the auction's COMMITTED betlis win ~8% at
    decision-p ~0.3 — so a big (1)-vs-committed gap ⇒ the leak is SELECTION
    (optimizer's curse over discards×contracts), not the net.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from vnet.pickup.colorless import ColorlessPickup

EXP17 = Path(__file__).parent.parent / "17_clean_pickup_net"
EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent.parent / "19_colorless_split"


def diag(contract):
    pk = ColorlessPickup.load(EXP19 / f"colorless_{contract}.pt")
    # prefer the disjoint held-out (seeds 900M+); fall back to a slice of 1M
    held = EXP18 / f"tier1_eval_{contract}_50k.npz"
    if held.exists():
        d = np.load(held); tag = "held-out 50k (seeds 900M+)"
    else:
        d = np.load(EXP17 / f"{contract}_god_alpha0_1M.npz"); tag = "1M (in-sample)"
    X, y = d['X'], d['y'].astype(np.float32)
    p = pk.predict(X)
    print(f"\n=== {contract}  ({tag})  N={len(y)}  base-rate={y.mean()*100:.2f}% ===")
    print(f"  {'pred bin':>12}  {'n':>7}  {'mean pred':>9}  {'god-true':>8}  {'gap(t-p)':>8}")
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        pp, tt = p[m].mean(), y[m].mean()
        flag = ""
        if m.sum() >= 30:
            flag = "  <-- net OVERconfident" if tt < pp - 0.05 else (
                    "  <-- net underconfident" if tt > pp + 0.05 else "")
        print(f"  [{lo:.2f},{hi:.2f})  {m.sum():>7}  {pp:>9.3f}  {tt:>8.3f}  "
              f"{tt-pp:>+8.3f}{flag}")
    # decision-relevant region: hands the auction would consider (pred high)
    hi = p >= 0.3
    if hi.sum():
        print(f"  pred≥0.30: n={hi.sum()}  mean pred {p[hi].mean():.3f}  "
              f"god-true {y[hi].mean():.3f}  "
              f"(committed-in-auction win ≈ 0.08 → gap is SELECTION)")


for c in ('betli', 'durchmars'):
    diag(c)
