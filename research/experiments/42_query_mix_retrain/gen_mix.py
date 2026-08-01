"""Mixture datagen for the exp41-convicted heads (duri_colored, reach100_20).

The fix for argmax sickness: train ON THE QUERY DISTRIBUTION (random deal x random
keep-10 x random legal trump — what the auction sweep actually asks), with the old
biased dealer kept ONLY as positive-class oversampling (the query distribution has
~0-2% positives; the net still needs to see what a make looks like).

Labels: god (exp40 verdict — realistic labels lose vs near-perfect opponents).
Output: {HEAD}_mix.npz (X, y, src 0=query/1=biased) + {HEAD}_calib_query.npz
(query-distribution holdout, disjoint seeds — isotonic must be fitted on THIS).

Env: HEAD, N_QUERY, N_BIAS, N_CALIB, WORKERS, SEED_BASE.
"""
import os, random, sys, time  # noqa
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from ulti.bidding.deal import deal_12_10_10
from ulti.bidding.recipe import sol_marriages
from ulti.eval.dojo import deal_durchmars_colored, deal_ulti_biased
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.solvers import pis
from ulti.vnet.pickup import featurize
from ultisolver._solver_core import set_multi_weights

HEAD = os.environ.get("HEAD", "duri_colored")
N_QUERY = int(os.environ.get("N_QUERY", "120000"))
N_BIAS = int(os.environ.get("N_BIAS", "60000"))
N_CALIB = int(os.environ.get("N_CALIB", "25000"))
WORKERS = int(os.environ.get("WORKERS", "8"))
SEED_BASE = int(os.environ.get("SEED_BASE", "910000000"))
TRUMPS = ("hearts", "acorns", "leaves", "bells")
OUT = Path(__file__).resolve().parent

CFG = {
    "duri_colored": dict(solve="durchmars", build="durchmars", weights=None, restrict=None),
    "reach100_20":  dict(solve="multi", build="parti",
                         weights={"score_geq_100": 1.0}, restrict="20"),
}[HEAD]


def _label(keep10, d1, d2, discard, trump) -> int:
    pos = pis.build_position(
        hands=[keep10, d1, d2], soloist=0, leader=0, contract=CFG["build"],
        trump=trump, talon=discard, declare_marriages=True,
        marriage_restrict=CFG["restrict"])
    if CFG["solve"] == "multi":
        _mv, v = pis.solve_best(pos, contract="multi")
        return int(v > 0.5)
    return int(god_says_soloist_wins(pos, contract=CFG["solve"]))


def q_worker(seed):
    """One QUERY-distribution sample: random deal, random keep-10, random legal trump."""
    rng = random.Random(seed)
    sol12, d1, d2 = deal_12_10_10(seed)
    drop = rng.sample(range(12), 2)
    keep10 = [c for i, c in enumerate(sol12) if i not in drop]
    discard = [sol12[i] for i in drop]
    if HEAD == "reach100_20":
        legal = [t for t in TRUMPS if sol_marriages(keep10, t)[1]]
        if not legal:
            return None
        trump = rng.choice(legal)
    else:
        trump = rng.choice(TRUMPS)
    y = _label(keep10, list(d1), list(d2), discard, trump)
    return featurize(keep10, trump, True), y


def b_worker(seed):
    """One BIASED (positive-oversampling) sample: strong dealer (alpha=1.0) + a SMART
    discard, because random discards kill the positive rate (probed 2026-08-01:
    duri random 5.5% vs smart 23.5%; r20 random 1-3% vs smart 18%). The label is
    still god — only the PROPOSAL distribution is biased."""
    if HEAD == "duri_colored":
        deal = deal_durchmars_colored(seed=seed, alpha=1.0)
        sol12 = list(deal.sol_hand) + list(deal.talon)
        # bury the two weakest off-trump cards (shortest suit first, lowest rank)
        off = sorted((c for c in sol12 if c.suit != deal.trump),
                     key=lambda c: (sum(1 for x in sol12 if x.suit == c.suit), c.rank_index))
        if len(off) < 2:
            return None
        discard = off[:2]
    else:
        deal = deal_ulti_biased(seed=seed, alpha=1.0)
        sol12 = list(deal.sol_hand) + list(deal.talon)
        # bury the two lowest no-point cards outside trump — preserves marriages,
        # tens/aces and trump length (what a real 20-100 bidder buries)
        cands = sorted((c for c in sol12
                        if c.suit != deal.trump and c.rank in ("7", "8", "9", "lower")),
                       key=lambda c: c.rank_index)
        if len(cands) < 2:
            return None
        discard = cands[:2]
    keep10 = [c for c in sol12 if c not in discard]
    trump = deal.trump
    if HEAD == "reach100_20" and not sol_marriages(keep10, trump)[1]:
        return None
    y = _label(keep10, list(deal.def1_hand), list(deal.def2_hand), discard, trump)
    return featurize(keep10, trump, True), y


def _init():
    if CFG["weights"] is not None:
        set_multi_weights(**CFG["weights"])


def run(worker, n, seed0, tag):
    Xs, ys = [], []
    t0 = time.time()
    scanned = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(worker, (seed0 + i for i in range(n * 8)), chunksize=64):
            scanned += 1
            if r is not None:
                Xs.append(r[0]); ys.append(r[1])
            if len(Xs) % max(1000, n // 30) == 0 and r is not None:
                el = time.time() - t0
                print(f"  [{tag}] {len(Xs)}/{n} kept (scanned {scanned}) "
                      f"{el:.0f}s pos={np.mean(ys):.4f} eta={el/max(1,len(Xs))*(n-len(Xs)):.0f}s",
                      flush=True)
            if len(Xs) >= n:
                pool.terminate()
                break
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int8)


def main():
    print(f"=== mixture datagen HEAD={HEAD} query={N_QUERY} bias={N_BIAS} calib={N_CALIB} ===",
          flush=True)
    Xq, yq = run(q_worker, N_QUERY, SEED_BASE, "query")
    Xb, yb = run(b_worker, N_BIAS, SEED_BASE + 20_000_000, "bias")
    X = np.concatenate([Xq, Xb]); y = np.concatenate([yq, yb])
    src = np.concatenate([np.zeros(len(yq), np.int8), np.ones(len(yb), np.int8)])
    np.savez_compressed(OUT / f"{HEAD}_mix.npz", X=X, y=y, src=src)
    print(f"mix: {len(y)} samples, positives {y.mean():.4f} "
          f"(query {yq.mean():.4f}, biased {yb.mean():.4f})", flush=True)
    Xc, yc = run(q_worker, N_CALIB, SEED_BASE + 40_000_000, "calib")
    np.savez_compressed(OUT / f"{HEAD}_calib_query.npz", X=Xc, y=yc)
    print(f"calib(query-only): {len(yc)} samples, positives {yc.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
