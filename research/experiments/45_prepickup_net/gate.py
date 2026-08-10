"""exp45 gate — does the pre-pickup model actually earn GP, at a real table?

ROTATION DESIGN (exp42's, kept because of the control it gives for free): each deal is
played three times, with the CANDIDATE in seat 0, then 1, then 2, and the incumbent in
the other two. The candidate therefore occupies every seat exactly once, so positional
effects — the forehand's structural pass tax above all — cancel exactly. Ulti is
zero-sum, so the candidate's mean GP per seat-deal IS the signal: **0.00 means parity**,
and two identical configs produce a literal 0.00, not a small number. If a run of
identical configs does not come out at exactly zero, the harness is broken, not the
bidder — which is the point of building it this way.

Everything downstream of the auction is the deployed stack (exp43's `_play_deployed` and
exp44's kontra + oracle scoring), so only the pickup decision differs between the arms.

Run:  WORKERS=6 python3 gate.py 800
      WORKERS=6 CONTROL=1 python3 gate.py 200      # incumbent vs itself → must be 0.00
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP44 = os.path.join(os.path.dirname(_HERE), "44_frontier_table")
for p in (_HERE, _EXP44, os.path.join(os.path.dirname(_HERE), "43_kontra_signals")):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED_BASE = 452_000_000
WORKERS = int(os.environ.get("WORKERS", "6"))
CONTROL = os.environ.get("CONTROL", "0") == "1"
# MODE=both      candidate uses the model for openings AND overcalls
# MODE=open_only candidate uses the model to OPEN, and the incumbent blind rule to
#                overcall. The first gate split hard by seat (+2.26 at forehand, −3.3 and
#                −2.8 behind it): opening more is worth it because the alternative is the
#                −4 passz tax, while overcalling more means taking contracts you would
#                rather have defended. This arm tests exactly that reading.
# MODE=threshold  no model at all — both arms use the incumbent blind rule, and the ONLY
#                 difference is the value the candidate seat must beat to OPEN when it is
#                 not the forehand (OPEN_THRESHOLD_NONFOREHAND).
MODE = os.environ.get("MODE", "both")
OPEN_THRESHOLD_NONFOREHAND = float(os.environ.get("OPEN_THRESHOLD_NONFOREHAND", "0.0"))
OUT = os.path.join(_HERE, f"gate_{MODE}.jsonl" if MODE != "both" else "gate.jsonl")

_INC = _CAND = None


def _init():
    global _INC, _CAND
    from predictor import blind_ev_model, load_pickup_model
    from ulti.bidding.auction import net_bid_fn
    from ulti.bidding.frontier import frontier_provider
    prov = frontier_provider()
    blind = blind_ev_model(prov)
    _INC = net_bid_fn(prov, pickup_model=blind)
    if CONTROL or MODE == "threshold":
        _CAND = _INC                      # threshold arm varies the auction, not the bidder
    else:
        model = load_pickup_model(provider=prov)
        if MODE == "open_only":
            def model(hand10, current, _m=model, _b=blind):
                return _m(hand10, current) if current is None else _b(hand10, current)
        _CAND = net_bid_fn(prov, pickup_model=model)


def _worker(seed):
    from selfplay import _full_auction, play_and_score
    out = []
    for cand_seat in (0, 1, 2):
        fns = [_INC, _INC, _INC]
        fns[cand_seat] = _CAND
        opens = None
        if MODE == "threshold":
            # No model on either side — the ONLY difference is what the candidate seat
            # must beat to OPEN. Isolates the auction's economics from the pickup model.
            opens = [-2.0, -2.0, -2.0]
            if cand_seat != 0:
                opens[cand_seat] = OPEN_THRESHOLD_NONFOREHAND
        try:
            a = _full_auction(seed, fns, open_thresholds=opens)
            if a is None:
                gp = [-4.0, 2.0, 2.0]          # forehand pays the pass penalty
                rec = {"contract": "passz", "seat_gp": gp}
            else:
                rec = play_and_score(a, seed)
            out.append({"seed": seed, "cand_seat": cand_seat,
                        "contract": rec["contract"],
                        "cand_gp": float(rec["seat_gp"][cand_seat]),
                        "cand_is_soloist": rec.get("winner") == cand_seat})
        except Exception as e:
            out.append({"seed": seed, "cand_seat": cand_seat,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def run(n):
    print(f"exp45 gate: {n} deals x 3 rotations, {WORKERS} workers"
          f"{'  [CONTROL: incumbent vs itself]' if CONTROL else ''}", flush=True)
    seeds = [SEED_BASE + i for i in range(n)]
    t0 = time.perf_counter()
    rows, errs = [], 0
    # SPAWN, not fork: the candidate arm loads a joblib/sklearn model, and sklearn has
    # already started threads by the time the pool forks — the macOS Objective-C runtime
    # refuses to continue in that child. Spawn re-imports cleanly in each worker.
    with get_context("spawn").Pool(WORKERS, initializer=_init) as pool, open(OUT, "w") as o:
        for i, res in enumerate(pool.imap_unordered(_worker, seeds, chunksize=2), 1):
            for r in res:
                o.write(json.dumps(r) + "\n")
                if r.get("error"):
                    errs += 1
                    if errs <= 3:
                        print(f"  ! {r['error']}", flush=True)
                else:
                    rows.append(r)
            o.flush()
            if i % 25 == 0:
                el = time.perf_counter() - t0
                g = np.array([x["cand_gp"] for x in rows])
                print(f"[gate] {i}/{n}  {el:.0f}s  eta {(n-i)/(i/el)/60:.1f}m  "
                      f"cand {g.mean():+.3f} GP/seat-deal  err={errs}", flush=True)
    report(rows)


def report(rows=None):
    if rows is None:
        rows = [json.loads(l) for l in open(OUT)]
        rows = [r for r in rows if "error" not in r]
    g = np.array([r["cand_gp"] for r in rows])
    n = len(g)
    se = g.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    print(f"\nexp45 gate — {n} seat-deals ({n//3} deals x 3 rotations)")
    print(f"  candidate: {g.mean():+.3f} GP/seat-deal   se {se:.3f}   "
          f"t {g.mean()/se if se else 0:+.2f}")
    print(f"  (0.000 = parity; the rotation makes identical configs cancel exactly)")
    for s in (0, 1, 2):
        m = np.array([r["cand_gp"] for r in rows if r["cand_seat"] == s])
        if len(m):
            print(f"    seat {s}: {m.mean():+.3f}  (n={len(m)})")
    sol = [r for r in rows if r.get("cand_is_soloist")]
    passz = [r for r in rows if r["contract"] == "passz"]
    print(f"  candidate declared: {len(sol)}/{n} ({100*len(sol)/n:.0f}%)   "
          f"passz deals: {100*len(passz)/n:.0f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        run(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
