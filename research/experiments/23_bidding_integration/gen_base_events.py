"""God-labeled datagen for the 3 NEW base events the full ladder needs beyond
exp 17's existing parti/ulti/betli/durchmars heads:

  reach100_40   P(soloist forces ≥100 holding the trump 40)   → drives 40-100 rungs
  reach100_20   P(soloist forces ≥100 holding a non-trump 20) → drives 20-100 rungs
  duri_colored  P(soloist sweeps all 10 WITH trump)           → drives colored-duri combos

Methodology matches exp 17's god labels (solve-at-root, contract-objective
minimax — defenders play the same objective, so the label is true double-dummy
"can the soloist force it"). The reach100 heads use the multi solver weighted
purely on `score_geq_100`; the marriage is seeded at solve-root via
`marriage_restrict` so the `≥100` threshold is EXACT (mirrors exp 21 recipe +
`scoring/oracle`). Validated by `validate_labels.py` against the trusted oracle.

Features = the exp-17 pickup featurize (36-dim hand+trump) of the 10-card
post-discard hand, so these heads bolt onto the colored multi-head net.

DESIGN FORKS (flagged for milan — sensible defaults chosen, easy to revisit):
  • dealer: deal_ulti_biased for reach100 (fat trump → realistic 40-hold rate,
    matches the validated eval_bid_100 path); deal_durchmars_colored for the sweep.
  • ALPHA default 0.0 (deployment distribution, like exp 17). eval_bid_100 used 0.6.
  • defender model = pure contract-objective minimax (exp-17-consistent), NOT the
    eval_bid_100 god-parti-defender variant. validate_labels.py reports both.

Env: HEAD (reach100_40|reach100_20|duri_colored), N, WORKERS, ALPHA, SEED_BASE.
"""
from __future__ import annotations

import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/milansimity/Cuccok/kodok/oldtawer")

from solvers import pis
from eval.pimc_matchup import god_says_soloist_wins
from eval.dojo import deal_ulti_biased, deal_durchmars_colored
from trickster._solver_core import set_multi_weights
from vnet.pickup import featurize
from recipe_local import sol_marriages   # local copy of exp21 recipe.sol_marriages

EXP_DIR = Path(__file__).parent

HEAD      = os.environ.get("HEAD", "reach100_40")
N         = int(os.environ.get("N", "300000"))
WORKERS   = int(os.environ.get("WORKERS", "8"))
ALPHA     = float(os.environ.get("ALPHA", "0.0"))
SEED_BASE = int(os.environ.get("SEED_BASE", "810000000"))

# Per-head config -----------------------------------------------------------
_CFG = {
    "reach100_40": dict(dealer="ulti", need="has_40", restrict="40",
                        solver="multi", weights={"score_geq_100": 1.0},
                        keep_marriage=True),
    "reach100_20": dict(dealer="ulti", need="has_20", restrict="20",
                        solver="multi", weights={"score_geq_100": 1.0},
                        keep_marriage=True),
    "duri_colored": dict(dealer="duri", need=None, restrict=None,
                         solver="durchmars", weights=None, keep_marriage=False),
}


def _deal(kind, seed):
    if kind == "ulti":
        return deal_ulti_biased(seed=seed, alpha=ALPHA)
    return deal_durchmars_colored(seed=seed, alpha=ALPHA)


def _discardable(sol12, keep_marriage):
    """Indices into sol12 that may be discarded. Never discard a king/upper
    (preserves every held marriage so the declared 100 survives)."""
    if not keep_marriage:
        return list(range(12))
    return [i for i, c in enumerate(sol12)
            if c.rank not in ("king", "upper")]


def worker(seed):
    cfg = _CFG[HEAD]
    deal = _deal(cfg["dealer"], seed)
    has_40, has_20 = sol_marriages(deal.sol_hand, deal.trump)
    if cfg["need"] == "has_40" and not has_40:
        return None
    if cfg["need"] == "has_20" and not has_20:
        return None

    sol12 = list(deal.sol_hand) + list(deal.talon)
    if len(sol12) != 12:
        return None
    d1 = list(deal.def1_hand)
    d2 = list(deal.def2_hand)
    trump = deal.trump

    rng = random.Random(seed ^ 0xA5A5A5A5)
    pool = _discardable(sol12, cfg["keep_marriage"])
    if len(pool) < 2:
        return None
    idx = set(rng.sample(pool, 2))
    discard = [sol12[i] for i in idx]
    remaining = [sol12[i] for i in range(12) if i not in idx]

    pos = pis.build_position(
        hands=[remaining, d1, d2], soloist=0, leader=0,
        contract="parti" if cfg["solver"] == "multi" else "durchmars",
        trump=trump, talon=discard, declare_marriages=True,
        marriage_restrict=cfg["restrict"],
    )
    label = 1 if god_says_soloist_wins(pos, contract=cfg["solver"]) else 0
    return featurize(remaining, trump, True), label


def _init_worker():
    cfg = _CFG[HEAD]
    if cfg["weights"] is not None:
        set_multi_weights(**cfg["weights"])


def out_path():
    tag = f"{N//1000}k" if N % 1000 == 0 and N < 1_000_000 else (
        f"{N//1_000_000}M" if N % 1_000_000 == 0 else str(N))
    return EXP_DIR / f"{HEAD}_god_a{ALPHA:g}_{tag}.npz"


def main():
    cfg = _CFG[HEAD]
    op = out_path()
    print(f"=== base-event datagen: HEAD={HEAD} ===", flush=True)
    print(f"  dealer={cfg['dealer']}  need={cfg['need']}  restrict={cfg['restrict']}"
          f"  solver={cfg['solver']}  weights={cfg['weights']}", flush=True)
    print(f"  N(kept target via scan)={N}  ALPHA={ALPHA}  workers={WORKERS}", flush=True)
    print(f"  out={op.name}", flush=True)

    # scan up to a generous cap to KEEP N records (reach100 heads filter by hold)
    cap = N * 10 if cfg["need"] else N
    seeds = (SEED_BASE + i for i in range(cap))
    Xs = []
    ys = []
    scanned = 0
    t0 = time.perf_counter()
    report_every = max(2000, N // 40)
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(worker, seeds, chunksize=128):
            scanned += 1
            if r is not None:
                Xs.append(r[0])
                ys.append(r[1])
            if len(ys) and len(ys) % report_every == 0 and r is not None:
                wall = time.perf_counter() - t0
                rate = len(ys) / wall
                eta = (N - len(ys)) / rate if rate else 0
                pos = sum(ys) / len(ys)
                print(f"  kept {len(ys)}/{N}  scanned {scanned}  keep={len(ys)/scanned:.2f}"
                      f"  pos={pos:.3f}  wall={wall:.0f}s  eta={eta:.0f}s", flush=True)
            if len(ys) >= N:
                break

    X = np.stack(Xs)
    y = np.array(ys, dtype=np.float32)
    np.savez_compressed(op, X=X, y=y)
    wall = time.perf_counter() - t0
    print(f"\n  saved {X.shape[0]} records  scanned {scanned}  "
          f"keep={X.shape[0]/scanned:.3f}  wall={wall:.0f}s", flush=True)
    print(f"  positive rate: {y.mean():.4f}  ({int(y.sum())} pos)", flush=True)


if __name__ == "__main__":
    main()
