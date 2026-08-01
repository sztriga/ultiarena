"""Per-head report card: is each bidding head calibrated WHERE IT IS QUERIED?

The duri lesson (exp30 leak → 2026-08-01 diagnosis): a head can be perfectly trained on
its dealer's distribution and still be poison in the auction, because the bidder sweeps
EVERY hand × every discard × every trump through it and then takes an argmax — errors in
regions the training never covered don't average out, they get SELECTED. duri_colored was
trained only on 5–8-trump hands; ~97% of its real queries hold 0–4 trumps, where it said
0.85 on contracts god calls unmakeable.

This audit treats every head identically, so it generalizes to future heads:

  1. QUERY-DISTRIBUTION SAMPLES — random deals, random keep-10 (a uniform draw from the
     66 discards the sweep enumerates), random trump; god-label each sampled (hand,
     trump) for the head's own objective. → sliced reliability: predicted vs god-truth
     by prediction bin and by trump count.
  2. ARGMAX SAMPLES — per deal, the head's OWN best (discard, trump) by predicted p,
     god-checked. This measures calibration under selection pressure — the number that
     actually decides bids.

Labels are exact (god double-dummy via the solver); solving runs on a process pool.
betli_real is NOT audited here: its target is P(make vs REALISTIC defense), so god
labels would measure the wrong thing.

Run:  python -m ulti.eval.head_audit [N_UNIFORM] [N_ARGMAX]
Outputs a table to stdout + JSON under research/experiments/41_head_audit/.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── head registry: name → (colored?, marriage gate, god objective) ───────────────
# objective: (build_contract, trump_required, multi_weights, marriage_restrict, solve_contract)
HEADS: Dict[str, dict] = {
    "parti":          dict(colored=True,  gate=None, solve="parti",     build="parti",     weights=None,                    restrict=None),
    "ulti":           dict(colored=True,  gate=None, solve="ulti",      build="ulti",      weights=None,                    restrict=None),
    "reach100_40":    dict(colored=True,  gate="40", solve="multi",     build="parti",     weights={"score_geq_100": 1.0}, restrict="40"),
    "reach100_20":    dict(colored=True,  gate="20", solve="multi",     build="parti",     weights={"score_geq_100": 1.0}, restrict="20"),
    "duri_colored":   dict(colored=True,  gate=None, solve="durchmars", build="durchmars", weights=None,                    restrict=None),
    "betli":          dict(colored=False, gate=None, solve="betli",     build="betli",     weights=None,                    restrict=None),
    "colorless_duri": dict(colored=False, gate=None, solve="durchmars", build="durchmars", weights=None,                    restrict=None),
}
TRUMPS = ("hearts", "acorns", "leaves", "bells")


def _god_label(job: dict) -> int:
    """Worker: god-solve one (hand, defenders, talon) for a head's objective → 0/1."""
    from ulti.card import card_from_id
    from ulti.eval.pimc_matchup import god_says_soloist_wins
    from ulti.solvers import pis
    from ultisolver._solver_core import set_multi_weights

    set_multi_weights(**(job["weights"] or {}))
    pos = pis.build_position(
        hands=[[card_from_id(c) for c in h] for h in job["hands"]],
        soloist=0, leader=0, contract=job["build"],
        trump=job["trump"],
        talon=[card_from_id(c) for c in job["talon"]],
        declare_marriages=(job["trump"] is not None),
        marriage_restrict=job["restrict"],
    )
    if job["solve"] == "multi":
        _mv, v = pis.solve_best(pos, contract="multi")
        return int(v > 0.5)
    return int(god_says_soloist_wins(pos, contract=job["solve"]))


def _mk_job(cfg: dict, keep10, d1, d2, discard, trump) -> dict:
    return {
        "hands": [[c.id for c in keep10], [c.id for c in d1], [c.id for c in d2]],
        "talon": [c.id for c in discard],
        "build": cfg["build"], "solve": cfg["solve"],
        "trump": trump if cfg["colored"] else None,
        "weights": cfg["weights"], "restrict": cfg["restrict"],
    }


def _trump_count(hand, trump) -> int:
    return sum(1 for c in hand if c.suit == trump)


def audit(n_uniform: int = 600, n_argmax: int = 250, workers: int = 4,
          seed: int = 20260801, log=print, weights_dir: str = None,
          calibrate: bool = False, only: list = None) -> dict:
    from ulti.bidding.deal import deal_12_10_10
    from ulti.bidding.provider import NetProvider
    from ulti.bidding.recipe import sol_marriages

    prov = (NetProvider(weights_dir=weights_dir, calibrate=calibrate)
            if weights_dir else NetProvider(calibrate=calibrate))
    rng = random.Random(seed)
    results: Dict[str, dict] = {}
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
    t0 = time.time()

    def _p_of(head: str, hand10, trump: Optional[str]) -> Optional[float]:
        """The head's prediction for (hand, trump), honouring the marriage gates the
        provider applies (a gated-off query never reaches the net → not a query)."""
        cfg = HEADS[head]
        if cfg["gate"] is not None:
            has40, has20 = sol_marriages(hand10, trump)
            if cfg["gate"] == "40" and not has40: return None
            if cfg["gate"] == "20" and not has20: return None
        bp = prov.base_probs(list(hand10), trump if cfg["colored"] else "hearts")
        return {"parti": bp.p_parti, "ulti": bp.p_ulti, "reach100_40": bp.p_reach100_40,
                "reach100_20": bp.p_reach100_20, "duri_colored": bp.p_duri_colored,
                "betli": bp.p_betli, "colorless_duri": bp.p_duri_colorless}[head]

    for head, cfg in HEADS.items():
        if only and head not in only:
            continue
        h_t0 = time.time()
        # ── 1. uniform query-distribution samples ──
        samples: List[Tuple[float, int, dict]] = []   # (pred, trump_count, job)
        tries = 0
        while len(samples) < n_uniform and tries < n_uniform * 60:
            tries += 1
            sol12, d1, d2 = deal_12_10_10(rng.getrandbits(31))
            drop = rng.sample(range(12), 2)
            keep10 = [c for i, c in enumerate(sol12) if i not in drop]
            discard = [sol12[i] for i in drop]
            trump = rng.choice(TRUMPS) if cfg["colored"] else None
            p = _p_of(head, keep10, trump)
            if p is None:
                continue
            samples.append((p, _trump_count(keep10, trump) if trump else -1,
                            _mk_job(cfg, keep10, d1, d2, discard, trump)))
        futs = [pool.submit(_god_label, s[2]) for s in samples]
        labels = [f.result() for f in futs]

        # ── 2. argmax samples: the head's own best (discard, trump) per deal ──
        arg: List[Tuple[float, int, dict]] = []
        deals_tried = 0
        while len(arg) < n_argmax and deals_tried < n_argmax * 25:
            deals_tried += 1
            sol12, d1, d2 = deal_12_10_10(rng.getrandbits(31))
            best = None
            for i, j in itertools.combinations(range(12), 2):
                keep10 = [c for k, c in enumerate(sol12) if k not in (i, j)]
                for trump in (TRUMPS if cfg["colored"] else (None,)):
                    p = _p_of(head, keep10, trump)
                    if p is not None and (best is None or p > best[0]):
                        best = (p, keep10, [sol12[i], sol12[j]], trump)
            if best is None or best[0] < 0.05:        # sweep would never surface this head
                continue
            p, keep10, discard, trump = best
            arg.append((p, _trump_count(keep10, trump) if trump else -1,
                        _mk_job(cfg, keep10, d1, d2, discard, trump)))
        afuts = [pool.submit(_god_label, a[2]) for a in arg]
        alabels = [f.result() for f in afuts]

        # ── aggregate ──
        def _bins(pairs):
            out = {}
            for p, y in pairs:
                b = min(int(p * 5), 4)                # 5 bins of 0.2
                out.setdefault(b, [0, 0])
                out[b][0] += 1; out[b][1] += y
            return {f"{b*0.2:.1f}-{(b+1)*0.2:.1f}": {"n": n, "god": round(w / n, 3)}
                    for b, (n, w) in sorted(out.items())}

        def _by_trumps(triples, labels):
            out = {}
            for (p, tc, _j), y in zip(triples, labels):
                if tc < 0: continue
                out.setdefault(tc, [0, 0.0, 0])
                out[tc][0] += 1; out[tc][1] += p; out[tc][2] += y
            return {tc: {"n": n, "pred": round(sp / n, 3), "god": round(sy / n, 3)}
                    for tc, (n, sp, sy) in sorted(out.items())}

        uni_pairs = [(p, y) for (p, _tc, _j), y in zip(samples, labels)]
        arg_pairs = [(p, y) for (p, _tc, _j), y in zip(arg, alabels)]
        results[head] = {
            "n_uniform": len(samples), "n_argmax": len(arg),
            "uniform_mean_pred": round(sum(p for p, _ in uni_pairs) / max(1, len(uni_pairs)), 4),
            "uniform_mean_god": round(sum(y for _, y in uni_pairs) / max(1, len(uni_pairs)), 4),
            "reliability": _bins(uni_pairs),
            "by_trump_count": _by_trumps(samples, labels),
            "argmax_mean_pred": round(sum(p for p, _ in arg_pairs) / max(1, len(arg_pairs)), 4),
            "argmax_mean_god": round(sum(y for _, y in arg_pairs) / max(1, len(arg_pairs)), 4),
            "argmax_by_trump_count": _by_trumps(arg, alabels),
            "seconds": round(time.time() - h_t0, 1),
        }
        r = results[head]
        log(f"[{time.time()-t0:7.0f}s] {head:15s} uniform n={r['n_uniform']} "
            f"pred={r['uniform_mean_pred']:.3f} god={r['uniform_mean_god']:.3f} | "
            f"ARGMAX n={r['n_argmax']} pred={r['argmax_mean_pred']:.3f} "
            f"god={r['argmax_mean_god']:.3f}")

    pool.shutdown()
    return results


def main():
    n_uni = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    out_dir = Path(__file__).resolve().parents[2] / "research" / "experiments" / "41_head_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    log(f"head audit: n_uniform={n_uni} n_argmax={n_arg} (per head)")
    results = audit(n_uniform=n_uni, n_argmax=n_arg, log=log)
    out = out_dir / "audit.json"
    json.dump(results, open(out, "w"), indent=1)
    log(f"\nwritten: {out}")

    # verdict table
    log(f"\n{'head':15s} {'argmax pred':>11s} {'argmax god':>10s} {'gap':>6s}  verdict")
    for head, r in results.items():
        gap = r["argmax_mean_pred"] - r["argmax_mean_god"]
        verdict = ("HEALTHY" if abs(gap) < 0.10 else
                   "MISCALIBRATED" if abs(gap) < 0.30 else "SICK")
        log(f"{head:15s} {r['argmax_mean_pred']:11.3f} {r['argmax_mean_god']:10.3f} "
            f"{gap:+6.2f}  {verdict}")


if __name__ == "__main__":
    main()
