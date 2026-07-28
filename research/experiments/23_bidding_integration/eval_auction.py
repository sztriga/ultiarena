"""Eval the full-ladder auction: run it on N deals, god-check the winning contract,
score realized GP. Reports bid distribution, soundness P(make|bid), GP/seat, P0
deficit, pass rate.

Soundness is COMPONENT-WISE god double-dummy (each bid component checked by its own
solver; combos AND their components). Exact for the simple contracts (the vast
majority after calibration); a slight over-estimate for combos (joint play not
modelled — the deferred play-out piece). Flagged.

Env: N, WORKERS, CALIBRATE (1), SEED_BASE.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from multiprocessing import get_context

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from ladder import GPTable, BidSet           # noqa: E402
from recipe_local import sol_marriages       # noqa: E402

N         = int(os.environ.get("N", "400"))
WORKERS   = int(os.environ.get("WORKERS", "8"))
CALIBRATE = os.environ.get("CALIBRATE", "1") not in ("0", "false", "no")
SEED_BASE = int(os.environ.get("SEED_BASE", "500000000"))

_PROV = None
GP = GPTable()


def _init():
    global _PROV
    from provider import NetProvider
    _PROV = NetProvider(calibrate=CALIBRATE)


def resolve_bidset(rung, sol10, trump):
    """Which interchangeable contract of the rung the soloist actually declares
    (for the equivalent pairs: the one whose marriage it holds)."""
    if len(rung.bids) == 1:
        return rung.bids[0]
    has40, has20 = sol_marriages(sol10, trump)
    for b in rung.bids:
        if b.forty_hundred and has40:
            return b
        if b.twenty_hundred and has20:
            return b
    for b in rung.bids:                       # fall back to a non-100 rep (ulti-duri)
        if not (b.forty_hundred or b.twenty_hundred):
            return b
    return rung.bids[0]


def _god(contract, sol10, d1, d2, trump, talon, restrict=None, multi100=False):
    from solvers import pis
    from eval.pimc_matchup import god_says_soloist_wins
    from trickster._solver_core import set_multi_weights
    if multi100:
        set_multi_weights(score_geq_100=1.0)
    pos = pis.build_position(
        hands=[list(sol10), list(d1), list(d2)], soloist=0, leader=0,
        contract=("parti" if multi100 else contract), trump=trump,
        talon=list(talon), declare_marriages=(multi100 or contract == "parti"),
        marriage_restrict=restrict,
    )
    return bool(god_says_soloist_wins(pos, contract=("multi" if multi100 else contract)))


def god_outcome(rung, trump, sol10, d1, d2, talon):
    """Component-wise god makeability → (realized GP/def, primary_made)."""
    bid = resolve_bidset(rung, sol10, trump)
    points_based = not (bid.durchmars or bid.betli)
    bid_a_100 = bid.forty_hundred or bid.twenty_hundred
    terit = 1
    gp = 0.0
    makes = []
    if bid.ulti:
        m = _god("ulti", sol10, d1, d2, trump, talon); makes.append(m)
        gp += GP.ulti_bid if m else -GP.ulti_bid_bukott
    if bid.durchmars:
        dt = None if rung.colorless else trump
        m = _god("durchmars", sol10, d1, d2, dt, talon); makes.append(m)
        gp += (GP.durchmars_bid if m else -GP.durchmars_bid) * terit
    if bid.betli:
        m = _god("betli", sol10, d1, d2, None, talon); makes.append(m)
        gp += (GP.betli if m else -GP.betli) * terit
    if bid.forty_hundred:
        m = _god(None, sol10, d1, d2, trump, talon, restrict="40", multi100=True)
        makes.append(m); gp += GP.forty_hundred_bid if m else -GP.forty_hundred_bid
    if bid.twenty_hundred:
        m = _god(None, sol10, d1, d2, trump, talon, restrict="20", multi100=True)
        makes.append(m); gp += GP.twenty_hundred_bid if m else -GP.twenty_hundred_bid
    if points_based and not bid_a_100:
        m = _god("parti", sol10, d1, d2, trump, talon); makes.append(m)
        gp += GP.parti if m else -GP.parti
    if bid.piros:
        gp *= 2
    return gp, all(makes)


def worker(seed):
    from auction import run_auction, net_bid_fn
    r = run_auction(seed, net_bid_fn(_PROV, GP), GP)
    if r["winner"] is None:
        return {"seed": seed, "contract": "PASS", "winner": None,
                "gps": [2 * -2.0, 2.0, 2.0], "made": None, "n_bids": 0}
    gp, made = god_outcome(r["rung"], r["trump"], r["sol"], r["def1"],
                           r["def2"], r["talon"])
    w = r["winner"]
    gps = [-gp, -gp, -gp]
    gps[w] = 2 * gp
    return {"seed": seed, "contract": r["contract"], "winner": w,
            "gps": gps, "made": made, "n_bids": r["n_bids"], "ev": r["ev"]}


def main():
    print(f"=== eval auction | N={N} CALIBRATE={CALIBRATE} workers={WORKERS} ===",
          flush=True)
    seeds = [SEED_BASE + i for i in range(N)]
    rows = []
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(worker, seeds, chunksize=4):
            rows.append(r)
            if len(rows) % 50 == 0:
                wall = time.perf_counter() - t0
                print(f"  {len(rows)}/{N}  {wall:.0f}s  "
                      f"{len(rows)/wall:.1f}/s", flush=True)
    wall = time.perf_counter() - t0

    from ladder import LADDER
    npass = sum(1 for r in rows if r["winner"] is None)
    played = [r for r in rows if r["winner"] is not None]
    made_rate = (sum(1 for r in played if r["made"]) / len(played)) if played else 0
    seat_gp = [sum(r["gps"][s] for r in rows) / len(rows) for s in range(3)]
    overall = sum(sum(r["gps"]) for r in rows) / len(rows)
    avg_bids = sum(r["n_bids"] for r in played) / len(played) if played else 0

    # per-contract: count, made-count, soloist-GP sum (gps[winner]) for the game
    stats = {}
    for r in rows:
        name = r["contract"]
        s = stats.setdefault(name, {"n": 0, "made": 0, "gp_sol": 0.0})
        s["n"] += 1
        if r["winner"] is not None:
            s["made"] += 1 if r["made"] else 0
            s["gp_sol"] += r["gps"][r["winner"]]
        else:                                   # PASS → opener (P0) forfeits
            s["gp_sol"] += r["gps"][0]

    print(f"\n=== results | N={N}  wall={wall:.0f}s  DEBIAS_PCTL (env) ===")
    print(f"pass rate {npass/N:.3f} | soundness(make|bid) {made_rate:.3f} "
          f"(component-wise god) | avg bids {avg_bids:.2f}")
    print(f"GP/seat-deal  P0 {seat_gp[0]:+.3f}  P1 {seat_gp[1]:+.3f}  "
          f"P2 {seat_gp[2]:+.3f}  (Σ {overall:+.3f})")

    # full table over EVERY ladder rung (+ PASS), in ladder order
    print(f"\n{'#':>3} {'contract':<34}{'bids':>6}{'freq':>7}{'winrate':>8}"
          f"{'GP/game':>9}")
    order = [("PASS", -1)] + [(r.name, r.index) for r in LADDER]
    for name, idx in order:
        s = stats.get(name)
        if not s:
            print(f"{('-' if idx<0 else idx):>3} {name:<34}{0:>6}{'0.0%':>7}"
                  f"{'—':>8}{'—':>9}")
            continue
        n = s["n"]
        freq = n / N
        wr = (s["made"] / n) if name != "PASS" else float("nan")
        gpg = s["gp_sol"] / n
        wr_s = "—" if name == "PASS" else f"{wr:.2f}"
        print(f"{('-' if idx<0 else idx):>3} {name:<34}{n:>6}{freq:>6.1%}"
              f"{wr_s:>8}{gpg:>+9.2f}")
    print(f"\nwinrate = P(soloist makes the bid | this contract won the auction); "
          f"god double-dummy defenders.\nGP/game = mean soloist net GP for games won "
          f"on this contract (= 2 × GP/def).")


if __name__ == "__main__":
    main()
