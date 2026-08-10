"""Rotation gate — the one harness both overnight tracks use to price a change in GP.

DESIGN (exp42's, and the reasons matter):

  * Each deal is played THREE times, with the candidate in seat 0, then 1, then 2, and the
    incumbent in the other two. The candidate occupies every seat exactly once, so
    positional effects — the forehand's structural passz tax above all — cancel.
  * Ulti is zero-sum, so the candidate's mean GP per seat-deal IS the delta. **0.000 means
    parity**, and two identical configs produce a literal 0.000. If they don't, the harness
    is broken, not the candidate.
  * The AGGREGATE needs no control run. A PER-SEAT number does: it still contains the
    positional baseline, which is large and noisy (2026-08-02: it moved a full GP between
    n=316 and n=600 on the same seeds, and reading it as a delta produced a confident and
    completely wrong conclusion). `run()` therefore refuses to report per-seat figures
    unless given a control, and `control_baseline()` produces one on the SAME seeds.

Both the bidder and the play stack can be varied per seat. Play-side knobs are process
globals read at import, so they travel as a per-seat `play_cfg` through
exp43's `_play_deployed` rather than as environment variables.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import get_context
from typing import Callable, Optional

import numpy as np

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPS = os.path.dirname(_HERE)
for _p in (os.path.join(_EXPS, "44_frontier_table"), os.path.join(_EXPS, "43_kontra_signals")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS_PENALTY = 2.0

# Set by the pool initializer: (bid_fn, play_cfg) for each arm.
_ARMS: dict = {}


def _init(spec: dict):
    """Build both arms inside the worker. `spec` is plain data (picklable): each arm is
    {"bid": <kwargs for frontier_bid_fn>, "play": <per-seat play cfg or None>}."""
    global _ARMS
    from ulti.bidding.frontier import frontier_bid_fn, frontier_provider
    prov = frontier_provider()
    _ARMS = {}
    for name, arm in spec.items():
        _ARMS[name] = {
            "bid": frontier_bid_fn(prov, **(arm.get("bid") or {})),
            "play": arm.get("play"),
            "opens": arm.get("opens"),
        }


def _one_deal(args):
    seed, spec_names = args
    from selfplay import _full_auction, play_and_score
    inc, cand = _ARMS[spec_names[0]], _ARMS[spec_names[1]]
    out = []
    for cand_seat in (0, 1, 2):
        try:
            fns = [inc["bid"]] * 3
            fns[cand_seat] = cand["bid"]
            opens = None
            if cand["opens"] is not None:
                opens = [-PASS_PENALTY] * 3
                opens[cand_seat] = cand["opens"]
            a = _full_auction(seed, fns, open_thresholds=opens)
            if a is None:
                rec = {"contract": "passz", "seat_gp": [-4.0, 2.0, 2.0], "winner": None}
            else:
                # play config travels by PLAY index (0 = soloist), so map the candidate's
                # SEAT to its play index for this deal
                play_cfg = None
                if cand["play"] is not None or inc["play"] is not None:
                    base = inc["play"] or {}
                    cp = cand["play"] or {}
                    ci = (cand_seat - a["winner"]) % 3
                    play_cfg = [dict(base) for _ in range(3)]
                    play_cfg[ci] = dict(cp)
                rec = play_and_score(a, seed, play_cfg=play_cfg)
            out.append({"seed": seed, "cand_seat": cand_seat,
                        "contract": rec["contract"],
                        "cand_gp": float(rec["seat_gp"][cand_seat]),
                        "cand_is_soloist": rec.get("winner") == cand_seat})
        except Exception as e:
            out.append({"seed": seed, "cand_seat": cand_seat,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def run(name: str, incumbent: dict, candidate: dict, n_deals: int, seed_base: int,
        out_path: str, workers: int = 8, log=print, spawn: bool = False) -> dict:
    """Play `n_deals` × 3 rotations and return the aggregate delta. Resumable."""
    seen = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["seed"])
                except Exception:
                    pass
    todo = [s for s in (seed_base + i for i in range(n_deals)) if s not in seen]
    spec = {"inc": incumbent, "cand": candidate}
    log(f"[{name}] {len(todo)} new deals x 3 rotations, {workers} workers")
    t0 = time.perf_counter()
    ctx = get_context("spawn" if spawn else "fork")
    rows, errs = [], 0
    with ctx.Pool(workers, initializer=_init, initargs=(spec,)) as pool, \
            open(out_path, "a") as o:
        args = [(s, ("inc", "cand")) for s in todo]
        for i, res in enumerate(pool.imap_unordered(_one_deal, args, chunksize=2), 1):
            for r in res:
                o.write(json.dumps(r) + "\n")
                if r.get("error"):
                    errs += 1
                    if errs <= 3:
                        log(f"  ! {name}: {r['error']}")
                else:
                    rows.append(r)
            o.flush()
            if i % 50 == 0:
                el = time.perf_counter() - t0
                g = np.array([x["cand_gp"] for x in rows])
                log(f"[{name}] {i}/{len(todo)}  {el:.0f}s  eta {(len(todo)-i)/(i/el)/60:.1f}m"
                    f"  delta {g.mean():+.3f}  err={errs}")
    return summarize(out_path, name=name, log=log)


def summarize(out_path: str, name: str = "", control: Optional[str] = None, log=print) -> dict:
    rows = []
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                rows.append(r)
    if not rows:
        return {"n": 0}
    g = np.array([r["cand_gp"] for r in rows])
    n = len(g)
    se = g.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    res = {"name": name, "n_seatdeals": n, "n_deals": n // 3,
           "delta": float(g.mean()), "se": float(se),
           "t": float(g.mean() / se) if se else 0.0,
           "declared": float(np.mean([bool(r.get("cand_is_soloist")) for r in rows])),
           "passz": float(np.mean([r["contract"] == "passz" for r in rows]))}
    log(f"[{name}] n={n//3} deals   delta {res['delta']:+.3f} +- {res['se']:.3f} "
        f"(t={res['t']:+.2f})   declares {100*res['declared']:.0f}%  passz {100*res['passz']:.0f}%")
    if control and os.path.exists(control):
        crows = [json.loads(l) for l in open(control) if l.strip()]
        crows = [r for r in crows if "error" not in r]
        base = {s: np.mean([r["cand_gp"] for r in crows if r["cand_seat"] == s])
                for s in (0, 1, 2)}
        shared = {r["seed"] for r in crows}
        sub = [r for r in rows if r["seed"] in shared]
        res["per_seat"] = {}
        for s in (0, 1, 2):
            m = [r["cand_gp"] for r in sub if r["cand_seat"] == s]
            if m:
                res["per_seat"][s] = float(np.mean(m) - base[s])
        log(f"    per-seat delta vs control: " +
            "  ".join(f"s{s} {v:+.2f}" for s, v in res["per_seat"].items()))
    return res


def control_baseline(n_deals: int, seed_base: int, out_path: str, workers: int = 8,
                     log=print) -> dict:
    """Incumbent against itself on the same seeds. Must return EXACTLY 0.000 in aggregate;
    its value is the per-seat positional baseline that any per-seat claim needs."""
    arm = {"bid": {}, "play": None}
    return run("control", arm, arm, n_deals, seed_base, out_path, workers, log)
