"""Testing harness for the bidding loop.

evaluate(make_bid_fn, ...) runs N fixed-seed deals through the architecture-
agnostic auction (self-play: all 3 seats use the candidate bidder), scores each
won contract with a scorer (god by default), and returns a reproducible report:

  • METRIC  — mean soloist GP/game vs god defenders (the scalar the loop maximizes)
  • GP/seat — P0/P1/P2 (P0 = forced opener; its deficit is the never-pass tax)
  • per-contract table — bids / freq / winrate / GP/game  (the bleed view, milan judges)

`make_bid_fn` is a MODULE-LEVEL factory `() -> bid_fn` (so each worker builds its
own provider). Swap it to test a different architecture; nothing else changes.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import get_context

_HERE = os.path.dirname(os.path.abspath(__file__))
_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for p in (_HERE, _E23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction import run_auction              # noqa: E402
from ladder import LADDER                    # noqa: E402
from scorers import god_outcome              # noqa: E402

_BID_FN = None
_SCORER = god_outcome


def _init(make_bid_fn, scorer):
    global _BID_FN, _SCORER
    _BID_FN = make_bid_fn()
    _SCORER = scorer


def _worker(seed):
    r = run_auction(seed, _BID_FN)
    if r["winner"] is None:
        return {"contract": "PASS", "winner": None, "gps": [-4.0, 2.0, 2.0],
                "made": None, "nb": 0}
    gp, made = _SCORER(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"],
                       r["talon"], seed=seed)
    w = r["winner"]
    gps = [-gp, -gp, -gp]
    gps[w] = 2 * gp
    return {"contract": r["contract"], "winner": w, "gps": gps,
            "made": made, "nb": r["n_bids"]}


def evaluate(make_bid_fn, n=2000, seed_base=500_000_000, workers=8,
             scorer=god_outcome):
    seeds = [seed_base + i for i in range(n)]
    rows = []
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(make_bid_fn, scorer)) as pool:
        for r in pool.imap_unordered(_worker, seeds, chunksize=4):
            rows.append(r)
    wall = time.perf_counter() - t0

    played = [r for r in rows if r["winner"] is not None]
    npass = len(rows) - len(played)
    # METRIC: mean GP of the seat that holds the contract (P0 forfeit on PASS)
    metric = sum((r["gps"][r["winner"]] if r["winner"] is not None else r["gps"][0])
                 for r in rows) / len(rows)
    # non-floor metric: soloist GP on games escalated ABOVE the piros-parti floor —
    # the discriminating signal (the floor tax is structural pre-kontra, same for all)
    nf = [r["gps"][r["winner"]] for r in played if r["contract"] != "piros parti"]
    nonfloor = (sum(nf) / len(nf)) if nf else 0.0
    seat_gp = [sum(r["gps"][s] for r in rows) / len(rows) for s in range(3)]
    avg_bids = (sum(r["nb"] for r in played) / len(played)) if played else 0.0

    stats = {}
    for r in rows:
        s = stats.setdefault(r["contract"], {"n": 0, "made": 0, "gp": 0.0})
        s["n"] += 1
        if r["winner"] is not None:
            s["made"] += 1 if r["made"] else 0
            s["gp"] += r["gps"][r["winner"]]
        else:
            s["gp"] += r["gps"][0]
    return {"n": n, "wall": wall, "metric": metric, "nonfloor": nonfloor,
            "n_nonfloor": len(nf), "seat_gp": seat_gp,
            "pass_rate": npass / len(rows), "avg_bids": avg_bids, "stats": stats}


def print_report(res, label=""):
    print(f"\n=== {label} | N={res['n']} wall={res['wall']:.0f}s ===")
    print(f"METRIC mean soloist GP/game {res['metric']:+.3f}   "
          f"(floor-dominated; vs god defenders)")
    print(f"NON-FLOOR GP/game {res['nonfloor']:+.3f}  (n={res['n_nonfloor']}) "
          f"— escalations above piros parti; the discriminating signal")
    sg = res["seat_gp"]
    print(f"GP/seat  P0 {sg[0]:+.3f}  P1 {sg[1]:+.3f}  P2 {sg[2]:+.3f}  | "
          f"pass {res['pass_rate']:.3f}  avg bids {res['avg_bids']:.2f}")
    print(f"\n{'#':>3} {'contract':<36}{'bids':>6}{'freq':>7}{'wr':>6}{'GP/game':>9}")
    st = res["stats"]
    rows = [("PASS", -1)] + [(r.name, r.index) for r in LADDER]
    for name, idx in rows:
        s = st.get(name)
        tag = "-" if idx < 0 else str(idx)
        if not s:
            print(f"{tag:>3} {name:<36}{0:>6}{'0.0%':>7}{'—':>6}{'—':>9}")
            continue
        n = s["n"]
        wr = "—" if name == "PASS" else f"{s['made']/n:.2f}"
        print(f"{tag:>3} {name:<36}{n:>6}{n/res['n']:>6.1%}{wr:>6}{s['gp']/n:>+9.2f}")


if __name__ == "__main__":
    from net_bidder import make_net_bid_fn
    from scorers import (god_outcome, pimc_outcome, kontra_full_outcome,
                         kontra_pimc_outcome)
    which = os.environ.get("SCORER", "god")
    scorer = {"pimc": pimc_outcome, "kontra": kontra_full_outcome,
              "kpimc": kontra_pimc_outcome}.get(which, god_outcome)
    res = evaluate(make_net_bid_fn, n=int(os.environ.get("N", "1500")),
                   scorer=scorer)
    print_report(res, f"net+composer | scorer={which}")
