"""Track A, phase 5 — the piros parti puzzle.

On the honest frontier (exp44, 6000 deals) piros parti is **15.7% of all deals**, gets
kontra'd **81%** of the time, makes only **56%** — and still returns **+2.92** to the
soloist. Three things about that are odd enough to be worth a night:

  1. It is the most-exercised kontra decision in the engine by a wide margin, and the only
     one whose rule is not structural: `parti -> kontra iff blind PIMC makeability < 0.10`.
     If that threshold is wrong it is wrong on one deal in six.
  2. A contract that makes 56% is barely above the 0.5 breakeven, so kontra should be close
     to a coin flip in value — yet the rule fires on 81% of them. Either the blind estimate
     is systematically low (exp26 found exactly that: it sampled RANDOM soloist hands and
     ignored that the soloist BID, reading 6-11% where the truth was ~80%), or 81% is right
     and the defenders are leaving money on the table by not kontra-ing the other 19%.
  3. Making 56% while collecting +2.92 means the párti component is NOT where the GP comes
     from. Something else is riding that unit.

This module answers all three from the corpus, with no new play-outs:

  * DECOMPOSE — re-score each deal and split the párti unit's GP by scoring component.
    Silent riders (silent durchmars, silent 100s, and the defenders' versions) REPLACE
    párti and ride its kontra level, so "kontra párti" doubles them too. That is the first
    place to look for the missing +2.92.
  * RELIABILITY — blind makeability vs what actually happened, binned. Is the signal honest
    at the threshold it is used at?
  * GP SWEEP — for every threshold, the realised soloist GP under `kontra iff mk < tau`,
    computed exactly from the recorded per-level payoffs. Includes never-kontra and
    always-kontra as bounds, and the deployed 0.10 as the incumbent.
  * ALTERNATIVES — structural signals (own card points, aces, trump length) scored the same
    way, because exp27 found structure beat makeability for every OTHER unit and parti was
    the one exception. Worth re-testing now that the bidder no longer cheats.

Run:  WORKERS=8 python3 parti.py run   [corpus.jsonl ...]
      python3 parti.py report
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
_EXP43 = os.path.join(os.path.dirname(_HERE), "43_kontra_signals")
for _p in (_EXP43, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(_HERE, "parti.jsonl")
DEPLOYED_TAU = 0.10


def _worker(rec):
    """One deal -> one row per defender: blind makeability, structure, outcome, payoffs."""
    from apps.api import ai_worker
    from ulti.bidding.scorers import resolve_bidset, _play_weights
    from ulti.card import RANK_POINTS, card_from_id
    from ulti.scoring.oracle import score as score_oracle
    from ulti.scoring.units import unit_of
    from ulti.solvers import pis
    from ulti.bidding.ladder import LADDER
    out = []
    try:
        if "parti" not in rec.get("units", {}):
            return []
        rung = next(r for r in LADDER if r.index == rec["rung_index"])
        trump = rec["trump"]
        sol = [card_from_id(i) for i in rec["sol"]]
        bid = resolve_bidset(rung, sol, trump)

        # ── decompose the parti unit's GP by component ──
        hands = [[card_from_id(i) for i in h] for h in (rec["sol"], rec["d1"], rec["d2"])]
        talon = [card_from_id(i) for i in rec["talon"]]
        n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
        if bid.betli:
            build_c, t, restrict = "betli", None, None
        elif bid.durchmars and rung.colorless and n_trick == 1:
            build_c, t, restrict = "durchmars", None, None
        else:
            build_c, t = "parti", trump
            restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        pos = pis.build_position(hands=[list(h) for h in hands], soloist=0, leader=0,
                                 contract=build_c, trump=t, talon=list(talon),
                                 declare_marriages=(t is not None),
                                 marriage_restrict=restrict, has_ulti=bool(bid.ulti))
        for _pid, cid in rec["hist"]:
            pis.apply_move(pos, card_from_id(cid))
        pv = score_oracle(final_pos=pos, bid=bid)
        comps = {k: int(v) for k, v in pv.components.items() if unit_of(k) == "parti"}

        for viewer in (1, 2):
            mk = ai_worker.op_unit_makeability({
                "hands0": [rec["sol"], rec["d1"], rec["d2"]], "talon": rec["talon"],
                "trump": trump, "unit": "parti", "viewer": viewer,
                "seed": rec["seed"] + 100 + viewer})
            own = hands[viewer]
            out.append({
                "seed": rec["seed"], "viewer": viewer, "contract": rec["contract"],
                "src": rec.get("src", "auction"),
                "mk": float(mk),
                "made": int(bool(rec["units"]["parti"]["made"])),
                "iso": rec["units"]["parti"]["iso"],
                "components": comps,
                # structural alternatives, from the defender's own hand only
                "own_pts": sum(RANK_POINTS[c.rank] for c in own),
                "own_ace": sum(1 for c in own if c.rank == "ace"),
                "own_ten": sum(1 for c in own if c.rank == "10"),
                "own_trumps": sum(1 for c in own if trump and c.suit == trump),
                "own_marr": sum(1 for s in {c.suit for c in own}
                                if any(c.suit == s and c.rank == "king" for c in own)
                                and any(c.suit == s and c.rank == "upper" for c in own)),
            })
    except Exception as e:
        return [{"seed": rec.get("seed"), "error": f"{type(e).__name__}: {e}"}]
    return out


def run(corpora, workers=8, log=print, limit=None):
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["seed"])
                except Exception:
                    pass
    deals = []
    for path in corpora:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("kept") and "parti" in r.get("units", {}) and r["seed"] not in seen:
                    deals.append(r)
    if limit:
        deals = deals[:limit]
    log(f"[parti] {len(deals)} deals with a live parti unit, {workers} workers")
    t0 = time.perf_counter()
    rows = errs = 0
    with get_context("fork").Pool(workers) as pool, open(OUT, "a") as o:
        for i, res in enumerate(pool.imap_unordered(_worker, deals, chunksize=4), 1):
            for r in res:
                o.write(json.dumps(r) + "\n")
                errs += int("error" in r)
                rows += int("error" not in r)
            o.flush()
            if i % 100 == 0:
                el = time.perf_counter() - t0
                log(f"[parti] {i}/{len(deals)}  {el:.0f}s  "
                    f"eta {(len(deals)-i)/(i/el)/60:.1f}m  rows={rows} err={errs}")
    log(f"[parti] done: {rows} rows, {errs} errors")


def _sweep(rows, key, lower_is_kontra, taus, log=print, label=""):
    """Realised soloist GP under `kontra iff signal <(or >) tau`, exactly, from iso[]."""
    v = np.array([r[key] for r in rows], dtype=float)
    iso = np.array([r["iso"] for r in rows], dtype=float)
    best = None
    for tau in taus:
        fire = (v < tau) if lower_is_kontra else (v > tau)
        gp = float(np.where(fire, iso[:, 1], iso[:, 0]).mean())
        if best is None or gp < best[1]:            # defenders want soloist GP LOW
            best = (float(tau), gp, float(fire.mean()))
    return best


def report(log=print):
    rows = [r for r in (json.loads(l) for l in open(OUT)) if "error" not in r]
    if not rows:
        log("no parti rows")
        return
    mk = np.array([r["mk"] for r in rows])
    made = np.array([r["made"] for r in rows])
    iso = np.array([r["iso"] for r in rows], dtype=float)
    log(f"\n=== PARTI DEEP-DIVE — {len(rows)} defender-positions "
        f"({len(set(r['seed'] for r in rows))} deals) ===\n")
    log(f"parti made: {100*made.mean():.1f}%   (breakeven for kontra is 50%)")

    # 1. where does the GP come from?
    agg = {}
    for r in rows[::2]:                       # one row per deal
        for k, v in r["components"].items():
            agg[k] = agg.get(k, 0) + v
    n_deals = len(rows) // 2
    log("\nGP on the PARTI unit, by scoring component (soloist per-def, kontra level 0):")
    for k, v in sorted(agg.items(), key=lambda kv: -abs(kv[1])):
        log(f"    {k:26s} {v/max(1,n_deals):+7.3f} /deal")
    log(f"    {'TOTAL':26s} {sum(agg.values())/max(1,n_deals):+7.3f} /deal")

    # 2. is the blind estimate honest?
    log("\nblind makeability vs reality:")
    log(f"    mean estimate {mk.mean():.3f}   actual make {made.mean():.3f}   "
        f"BIAS {mk.mean()-made.mean():+.3f}")
    for lo, hi in ((0, .05), (.05, .10), (.10, .20), (.20, .40), (.40, .70), (.70, 1.01)):
        m = (mk >= lo) & (mk < hi)
        if m.sum() >= 10:
            log(f"    mk [{lo:.2f},{hi:.2f})  n={int(m.sum()):5d}  actual make {made[m].mean():.3f}")

    # 3. what threshold would a defender actually want?
    log("\nsoloist GP per defender-position under `kontra iff mk < tau` "
        "(LOWER is better for the defenders):")
    log(f"    {'never kontra':>22s}  {iso[:,0].mean():+7.3f}")
    log(f"    {'always kontra':>22s}  {iso[:,1].mean():+7.3f}")
    dep = mk < DEPLOYED_TAU
    log(f"    {'DEPLOYED mk<0.10':>22s}  "
        f"{float(np.where(dep, iso[:,1], iso[:,0]).mean()):+7.3f}   fires {100*dep.mean():.0f}%")
    b = _sweep(rows, "mk", True, np.linspace(0.0, 1.0, 51))
    log(f"    {'swept optimum':>22s}  {b[1]:+7.3f}   at mk<{b[0]:.2f}, fires {100*b[2]:.0f}%")

    # 4. does structure beat it? (exp27 said no for parti — retest on the honest bidder)
    log("\nstructural alternatives, same sweep:")
    for key, lower in (("own_pts", False), ("own_ace", False), ("own_ten", False),
                       ("own_trumps", False), ("own_marr", False)):
        vals = np.array([r[key] for r in rows], dtype=float)
        taus = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, 25)))
        b2 = _sweep(rows, key, lower, taus)
        log(f"    {key:14s} best {b2[1]:+7.3f}  at {key} > {b2[0]:.1f}, fires {100*b2[2]:.0f}%")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "run":
        corpora = sys.argv[2:] or [os.path.join(_EXP43, "played.jsonl")]
        run(corpora, workers=int(os.environ.get("WORKERS", "8")))
    else:
        report()


if __name__ == "__main__":
    main()
