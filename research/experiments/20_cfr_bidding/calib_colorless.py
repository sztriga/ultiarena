"""Selection-aware calibration for betli/duri DECISIONS (not the net).

The net is fine (underconfident on random hands); the leak is that the auction's
argmax-over-discards selects the over-predicted tail (optimizer's curse). So we
calibrate the quantity the bidder actually decides on — the DEBIAS_PCTL quantile
of the 66 discard scores — against the true god outcome of the played (argmax)
discard, on the α=0 deployment distribution.

Output: calib_{betli,durchmars}.npz holding a monotone lookup (x grid → calibrated
p) that auction_h2h applies to the betli/duri decision-p so the −2 floor rejects
the garbage automatically.

Usage: N_GEN=80000 PCTL=0.80 python calib_colorless.py
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))

from _lib import deal_12_10_10                            # noqa: E402
from solvers import pis                                   # noqa: E402
from eval.pimc_matchup import god_says_soloist_wins       # noqa: E402
from vnet.pickup import featurize                         # noqa: E402
from vnet.pickup.colorless import ColorlessPickup         # noqa: E402
import itertools                                          # noqa: E402

EXP19 = Path(__file__).parent.parent / "19_colorless_split"

N_GEN = int(os.environ.get("N_GEN", 80000))
SEED_BASE = int(os.environ.get("SEED_BASE", 500_000_000))   # disjoint from train/eval
PCTL = float(os.environ.get("PCTL", 0.80))
N_WORKERS = int(os.environ.get("N_WORKERS", 8))

_NETS = None
_SOLVER = {'betli': 'betli', 'durchmars': 'durchmars'}


def _nets():
    global _NETS
    if _NETS is None:
        _NETS = {c: ColorlessPickup.load(EXP19 / f"colorless_{c}.pt")
                 for c in ('betli', 'durchmars')}
    return _NETS


def _one(seed):
    nets = _nets()
    sol12, d1, d2 = deal_12_10_10(seed)
    out = {}
    discards = list(itertools.combinations(sol12, 2))
    finals = [[c for c in sol12 if c not in dp] for dp in discards]
    X = np.stack([featurize(h, None, False) for h in finals])
    for c in ('betli', 'durchmars'):
        ps = nets[c].predict(X)
        dec_p = float(np.quantile(ps, PCTL))
        bi = int(ps.argmax())
        sol10 = finals[bi]
        pos = pis.build_position(hands=[sol10, list(d1), list(d2)], soloist=0,
                                 leader=0, contract=c, trump=None,
                                 talon=list(discards[bi]))
        win = god_says_soloist_wins(pos, contract=c)
        out[c] = (dec_p, 1.0 if win else 0.0)
    return out


def _pav(x, y):
    """Pool-adjacent-violators isotonic fit; returns (sorted x, fitted y)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float).copy()
    w = np.ones_like(ys)
    # PAV
    i = 0
    blocks = [[ys[k], 1.0] for k in range(len(ys))]
    val = ys.copy()
    # simple O(n) PAV
    stack = []  # (mean, weight, count_start)
    means, weights = [], []
    for v in ys:
        means.append(v); weights.append(1.0)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2 = means.pop(), weights.pop()
            m1, w1 = means.pop(), weights.pop()
            means.append((m1 * w1 + m2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
    # expand back
    fitted = np.empty_like(ys)
    idx = 0
    for m, wt in zip(means, weights):
        n = int(round(wt))
        fitted[idx:idx + n] = m
        idx += n
    return xs, fitted


def main():
    print(f"=== colorless decision calibration  N={N_GEN}  PCTL={PCTL} ===",
          flush=True)
    seeds = [SEED_BASE + i for i in range(N_GEN)]
    rows = {'betli': [], 'durchmars': []}
    t0 = time.perf_counter()
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_one, seeds, chunksize=32):
            for c in ('betli', 'durchmars'):
                rows[c].append(r[c])
            if sum(len(v) for v in rows.values()) % 40000 == 0:
                print(f"  {len(rows['betli'])}/{N_GEN}  "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    print(f"  gen wall {time.perf_counter()-t0:.0f}s\n")

    grid = np.linspace(0, 1, 51)
    for c in ('betli', 'durchmars'):
        arr = np.array(rows[c])
        x, y = arr[:, 0], arr[:, 1]
        xs, fit = _pav(x, y)
        cal = np.interp(grid, xs, fit)               # grid → calibrated p
        np.savez(Path(__file__).parent / f"calib_{c}.npz", grid=grid, cal=cal,
                 pctl=PCTL)
        print(f"--- {c}: N={len(y)}  base-win={y.mean()*100:.2f}% "
              f"(deal_12_10_10 best-discard) ---")
        # DIRECT binned calibration on the auction distribution
        edges = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 1.01]
        print(f"    {'decision_p bin':>16}  {'n':>6}  {'mean dp':>7}  "
              f"{'true-win':>8}  {'calibrated':>10}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() == 0:
                continue
            print(f"    [{lo:.2f},{hi:.2f})      {m.sum():>6}  {x[m].mean():>7.3f}"
                  f"  {y[m].mean():>8.3f}  {np.interp(x[m].mean(), grid, cal):>10.3f}")
        cross = np.interp(0.30, cal, grid) if cal.max() >= 0.30 else None
        print("    → need decision_p ≳ %.2f for calibrated p≥0.30 (clears −2 floor)"
              % cross if cross is not None else
              "    → calibrated never reaches 0.30 ⇒ betli/duri ~never bid")


if __name__ == "__main__":
    main()
