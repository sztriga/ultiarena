"""exp43 — is a kontra signal worth anything? Scored in GP, against a counterparty.

The payoff algebra makes the bet identical for every unit: kontra is +EV for the
defenders iff P(soloist makes) < 0.5. Even bid-ulti, whose bukott is the asymmetric
−(2^L+1)·4, breaks even at exactly one half. So a candidate signal is just a
probability estimate, and this module prices it.

WHY AUC IS NOT THE ANSWER. Kontra is not a bet against nature — the SOLOIST RESPONDS.
After trick 1 he rekontras with information the defender never had, so an edge measured
against the raw make-rate is an illusion: you get doubled precisely when you were wrong.
That is adverse selection, and it is the most likely reason exp27's aggressive parti
rule lost money even though parti was only making 36%.

We cannot know the deployed rekontra without re-solving, so every policy is priced in
BOTH bounding worlds and reported as a band:

    no-rekontra   soloist never re-doubles      → the defender's raw edge (optimistic)
    god-rekontra  soloist re-doubles iff he WILL make it → maximal adverse selection

The truth sits between. A signal that still pays under god-rekontra is real alpha; one
that only pays under no-rekontra is a mirage and must not be deployed.

Kontra bookkeeping follows the engine exactly: colored units are SHARED (együtt sírunk —
one defender's kontra sets the level for both), colourless units keep separate
per-defender counters. `iso[L]` from the corpus is the unit's isolated soloist
per-defender GP at level L, so a policy's value is a pure table lookup — no replay.

Run:  python3 evaluate.py            # baseline + single-feature hunt + GBM, per unit
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from features import UNITS, assert_no_leak, build_table   # noqa: E402

BREAKEVEN = 0.5      # kontra iff P(make) < this — exact for every unit, see docstring


# ── policy pricing ──────────────────────────────────────────────────────────────

def _order(tab: dict) -> np.ndarray:
    """Rows sorted (seed, viewer) so each deal's two defender rows are adjacent."""
    return np.lexsort((tab["viewer"], tab["seed"]))


def defender_gp(tab: dict, decide: np.ndarray, rekontra: str = "none") -> np.ndarray:
    """Per-row defender GP under `decide` (bool per defender row).

    Returns one value per row: what THAT defender earns from this unit. Colored units
    share a level across both defenders; colourless units do not.
    """
    idx = _order(tab)
    iso = tab["iso"][idx]                       # (n, 3) soloist per-def GP at level 0/1/2
    made = tab["y"][idx].astype(bool)
    colorless = tab["X"][idx, tab["names"].index("v_colorless")] > 0.5
    dec = decide[idx]
    n = len(idx)
    lvl = np.zeros(n, dtype=int)

    # deals are contiguous pairs after the lexsort
    for a in range(0, n - 1, 2):
        b = a + 1
        if colorless[a]:
            lvl[a] = 1 if dec[a] else 0
            lvl[b] = 1 if dec[b] else 0
        else:
            shared = 1 if (dec[a] or dec[b]) else 0
            lvl[a] = lvl[b] = shared
    if n % 2:                                   # unpaired tail row (shouldn't happen)
        lvl[-1] = 1 if dec[-1] else 0

    if rekontra == "god":
        # The soloist re-doubles exactly when he is going to make it: the defender is
        # only ever doubled on the deals where the kontra was wrong.
        lvl = np.where((lvl > 0) & made, 2, lvl)
    elif rekontra != "none":
        raise ValueError(rekontra)

    sol = iso[np.arange(n), lvl]
    out = np.empty(n, dtype=np.float64)
    out[idx] = -sol                             # defender earns the negative of soloist
    return out


def price(tab: dict, decide: np.ndarray, base: np.ndarray) -> Dict[str, float]:
    """Defender GP/row for a policy and its delta vs `base`, in both rekontra worlds."""
    out = {"rate": float(decide.mean())}
    for world in ("none", "god"):
        g = defender_gp(tab, decide, world)
        b = defender_gp(tab, base, world)
        d = g - b
        out[f"gp_{world}"] = float(g.mean())
        out[f"d_{world}"] = float(d.mean())
        # paired t over deals (the two defender rows of a deal are not independent)
        per_deal = d.reshape(-1, 2).sum(axis=1) if len(d) % 2 == 0 else d
        s = per_deal.std(ddof=1)
        out[f"t_{world}"] = float(per_deal.mean() / (s / np.sqrt(len(per_deal)))) if s > 0 else 0.0
    return out


# ── the incumbent ───────────────────────────────────────────────────────────────

def deployed_decide(unit: str, tab: dict) -> Tuple[np.ndarray, bool]:
    """apps/api/kontra_flow._ai_defender_kontras_unit, expressed on the feature table.
    Second return value = whether the rule is faithfully reproducible here (parti's is
    not: it calls PIMC makeability, which the corpus does not carry)."""
    X, names = tab["X"], tab["names"]
    col = lambda n: X[:, names.index(n)]
    if unit == "ulti":
        return col("t_n") >= 4, True
    if unit == "durchmars":
        return (col("v_colorless") < 0.5) & (col("t_n") >= 3), True
    if unit == "40_100":
        return col("p_trump_marr_card") > 0.5, True
    if unit == "parti":
        return np.zeros(len(X), dtype=bool), False       # PIMC makeability — not in corpus
    return np.zeros(len(X), dtype=bool), True            # 20_100 / betli → abstain


# ── candidate 1: the best SINGLE feature (directly comparable to the incumbent) ──

def best_single(tab: dict, base: np.ndarray, world: str = "god",
                min_rate: float = 0.02) -> List[dict]:
    """Sweep every feature × threshold × direction; rank by defender GP delta.

    Scored in the adverse-selection world by default — a rule that only wins when the
    soloist never re-doubles is not a rule we would ship.
    """
    X, names = tab["X"], tab["names"]
    found = []
    for j, name in enumerate(names):
        v = X[:, j]
        qs = np.unique(np.quantile(v, np.linspace(0.05, 0.95, 19)))
        for thr in qs:
            for ge in (True, False):
                dec = (v >= thr) if ge else (v <= thr)
                if dec.mean() < min_rate or dec.mean() > 1 - min_rate / 4:
                    continue
                p = price(tab, dec, base)
                found.append({"feat": name, "thr": float(thr), "ge": ge, **p})
    found.sort(key=lambda r: -r[f"d_{world}"])
    return found


# ── candidate 2: a learned model over the whole information set ─────────────────

def gbm_oof(tab: dict, seed: int = 43) -> Optional[np.ndarray]:
    """Out-of-fold P(make), grouped by DEAL so a deal's two defender rows never split
    across folds (they share the outcome — splitting them leaks the label)."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupKFold
    except ImportError:
        return None
    X, y, g = tab["X"], tab["y"], tab["seed"]
    if len(np.unique(y)) < 2 or len(X) < 200:
        return None
    oof = np.zeros(len(y), dtype=np.float64)
    n_splits = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


# ── report ──────────────────────────────────────────────────────────────────────

def _fmt(r: dict) -> str:
    return (f"rate {100*r['rate']:5.1f}%  "
            f"no-rk {r['gp_none']:+6.2f} (Δ{r['d_none']:+5.2f} t={r['t_none']:+5.1f})  "
            f"god-rk {r['gp_god']:+6.2f} (Δ{r['d_god']:+5.2f} t={r['t_god']:+5.1f})")


def report(tables: Dict[str, dict], top_k: int = 4, log=print) -> None:
    for unit in UNITS:
        if unit not in tables:
            continue
        tab = tables[unit]
        n = len(tab["y"])
        base, faithful = deployed_decide(unit, tab)
        log(f"\n{'='*100}\n{unit.upper()}   rows={n}  deals={n//2}  "
              f"make={100*tab['y'].mean():.1f}%   "
              f"(breakeven {100*BREAKEVEN:.0f}% → {'KONTRA-ABLE' if tab['y'].mean() < BREAKEVEN else 'usually makes'})")
        if not faithful:
            log("  ! deployed baseline shown as ABSTAIN — the live rule uses PIMC "
                  "makeability, which this corpus does not carry")
        log(f"  deployed:      {_fmt(price(tab, base, base))}")
        allk = np.ones(n, dtype=bool)
        log(f"  always-kontra: {_fmt(price(tab, allk, base))}")

        singles = best_single(tab, base)
        if singles:
            log("  best single features (ranked by Δ under god-rekontra):")
            seen = set()
            shown = 0
            for r in singles:
                if r["feat"] in seen:
                    continue
                seen.add(r["feat"])
                op = ">=" if r["ge"] else "<="
                print(f"    {r['feat']:18s} {op} {r['thr']:6.2f}   {_fmt(r)}")
                shown += 1
                if shown >= top_k:
                    break

        p = gbm_oof(tab)
        if p is not None:
            dec = p < BREAKEVEN
            log(f"  GBM (all features, grouped OOF)  AUC={auc(tab['y'], p):.3f}")
            log(f"    p<0.50:      {_fmt(price(tab, dec, base))}")
            best = None
            for thr in np.linspace(0.1, 0.9, 33):
                r = price(tab, p < thr, base)
                if best is None or r["d_god"] > best[1]["d_god"]:
                    best = (thr, r)
            if best:
                print(f"    p<{best[0]:.2f} (swept, mildly optimistic): {_fmt(best[1])}")
            # is def2's extra information worth anything?
            v2 = tab["viewer"] == 2
            if v2.any() and (~v2).any():
                print(f"    AUC def1={auc(tab['y'][~v2], p[~v2]):.3f}  "
                      f"def2={auc(tab['y'][v2], p[v2]):.3f}")


def main():
    tables = build_table()
    assert_no_leak(tables)
    print("exp43 — kontra signal pricing.  GP is PER DEFENDER PER DEAL, defender-positive.")
    print("Δ is vs the deployed rule. A signal must survive GOD-REKONTRA to be real.")
    report(tables)


if __name__ == "__main__":
    main()
