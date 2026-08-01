"""Refit each candidate's isotonic with TAIL RESOLUTION.

The query-only isotonic had ~2 positives in 25k samples: every bin's god-rate was ~0,
so the curve ceilinged the head at ~0.002 and duri could never clear FLOOR again
(over-correction — from always-wrong to always-mute).

Fix: fit on the UNION of (a) the uniform query holdout (owns the low-p mass) and
(b) a FRESH biased-dealer holdout, disjoint seeds (populates the high-p bins with
real god rates). Bins are FIXED EDGES on the predicted p, so each bin's estimate
conditions on the prediction — the proposal bias of (b) is second order there.

Env: HEAD. Writes candidates/{HEAD}_isotonic.npz (overwrites the mute one).
"""
import os, random, sys, time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from ulti.bidding.base_head import Head
from ulti.bidding.recipe import sol_marriages
from ulti.eval.dojo import deal_durchmars_colored, deal_ulti_biased
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.solvers import pis
from ulti.vnet.pickup import featurize
from ultisolver._solver_core import set_multi_weights

HEAD = os.environ.get("HEAD", "duri_colored")
N_BIAS = int(os.environ.get("N_BIAS", "8000"))
WORKERS = int(os.environ.get("WORKERS", "8"))
SEED0 = 970_000_000                      # disjoint from mix (910M/930M) + calib (950M)
HERE = Path(__file__).resolve().parent
EDGES = np.array([0, .02, .05, .10, .20, .30, .45, .60, .75, .90, 1.0])

CFG = {
    "duri_colored": dict(solve="durchmars", build="durchmars", weights=None, restrict=None),
    "reach100_20":  dict(solve="multi", build="parti",
                         weights={"score_geq_100": 1.0}, restrict="20"),
}[HEAD]


def b_worker(seed):
    """Fresh biased sample — same smart-discard proposal as gen_mix.b_worker."""
    if HEAD == "duri_colored":
        deal = deal_durchmars_colored(seed=seed, alpha=1.0)
        sol12 = list(deal.sol_hand) + list(deal.talon)
        off = sorted((c for c in sol12 if c.suit != deal.trump),
                     key=lambda c: (sum(1 for x in sol12 if x.suit == c.suit), c.rank_index))
        if len(off) < 2: return None
        discard = off[:2]
    else:
        deal = deal_ulti_biased(seed=seed, alpha=1.0)
        sol12 = list(deal.sol_hand) + list(deal.talon)
        cands = sorted((c for c in sol12
                        if c.suit != deal.trump and c.rank in ("7", "8", "9", "lower")),
                       key=lambda c: c.rank_index)
        if len(cands) < 2: return None
        discard = cands[:2]
    keep10 = [c for c in sol12 if c not in discard]
    if HEAD == "reach100_20" and not sol_marriages(keep10, deal.trump)[1]:
        return None
    pos = pis.build_position(hands=[keep10, list(deal.def1_hand), list(deal.def2_hand)],
                             soloist=0, leader=0, contract=CFG["build"], trump=deal.trump,
                             talon=discard, declare_marriages=True,
                             marriage_restrict=CFG["restrict"])
    if CFG["solve"] == "multi":
        set_multi_weights(**CFG["weights"])
        _mv, v = pis.solve_best(pos, contract="multi")
        y = int(v > 0.5)
    else:
        y = int(god_says_soloist_wins(pos, contract=CFG["solve"]))
    return featurize(keep10, deal.trump, True), y


def _init():
    if CFG["weights"] is not None:
        set_multi_weights(**CFG["weights"])


def main():
    ck = torch.load(HERE / "candidates" / f"{HEAD}_baseline.pt", weights_only=False)
    net = Head(ck["in_dim"]); net.load_state_dict(ck["state_dict"]); net.eval()

    Xs, ys = [], []
    t0 = time.time()
    with get_context("fork").Pool(WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(b_worker, (SEED0 + i for i in range(N_BIAS * 6)), chunksize=64):
            if r is not None:
                Xs.append(r[0]); ys.append(r[1])
            if len(Xs) >= N_BIAS:
                pool.terminate(); break
    Xb, yb = np.array(Xs, np.float32), np.array(ys, np.float32)
    print(f"fresh biased holdout: {len(yb)} samples pos={yb.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

    c = np.load(HERE / f"{HEAD}_calib_query.npz")
    Xq, yq = c["X"].astype(np.float32), c["y"].astype(np.float32)
    X = np.concatenate([Xq, Xb]); y = np.concatenate([yq, yb])
    with torch.no_grad():
        p = net(torch.from_numpy(X)).numpy()

    xs, cal = [], []
    print(f"{'bin':>12s} {'n':>7s} {'god rate':>9s}")
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        m = (p >= a) & (p < b)
        if m.sum() < 20:                  # too thin — let interp bridge it
            continue
        xs.append(float(p[m].mean())); cal.append(float(y[m].mean()))
        print(f"{a:.2f}-{b:.2f} {int(m.sum()):7d} {y[m].mean():9.3f}", flush=True)
    xs = np.array(xs); cal = np.maximum.accumulate(np.array(cal))
    np.savez(HERE / "candidates" / f"{HEAD}_isotonic.npz", x=xs, y=cal)
    np.savez(HERE / "candidate_full" / f"{HEAD}_isotonic.npz", x=xs, y=cal)

    pos_cal = np.interp(p[len(yq):][yb == 1], xs, cal)
    print(f"calibrated p on god-POSITIVE hands: mean={pos_cal.mean():.3f} "
          f">0.8: {(pos_cal > 0.8).mean():.2%}", flush=True)


if __name__ == "__main__":
    main()
