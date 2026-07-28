"""exp32 — TRUE head-to-head: RECOMMENDED config vs CURRENT frontier config, at the SAME table.

Per-bidder config (bidder.py/auction.py now thread floor/pctl/duri_mult), so both configs compete
in one auction. For each deal we run 3 seatings (R at seat 0/1/2, C at the other two) to
neutralise position; kontra + PIMC play are config-independent so this ISOLATES the bidding config.
Metric: R's mean GP/game as the lone R among 2 C (zero-sum → 0 if equal; positive = R beats C).

CURRENT     = FLOOR 0.70, DEBIAS 0.80, DURI_TERIT_MULT 1.0  (deployed)
RECOMMENDED = FLOOR 0.80, DEBIAS 0.85, DURI_TERIT_MULT 0.3
Env: N (deals), WORKERS.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import get_context

os.environ["KONTRA"] = "1"
os.environ.setdefault("FLOOR", "0.7"); os.environ.setdefault("DEBIAS_PCTL", "0.80")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/29_frontier_table", f"{_REPO}/experiments/27_kontra_revamp",
           f"{_REPO}/experiments/24_bidding_loop", f"{_REPO}/experiments/23_bidding_integration",
           f"{_REPO}/experiments/14_minigame_bid_eval", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(_HERE, "h2h.jsonl")
SEED_BASE = 480_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
PASS_PEN = 2.0
CUR = dict(pctl=0.80, floor=0.70, duri_mult=1.0)
REC = dict(pctl=0.85, floor=0.80, duri_mult=0.3)

_R = _C = None


def _init():
    global _R, _C
    from provider import NetProvider
    from auction import net_bid_fn
    prov = NetProvider(calibrate=True)            # config-independent; shared
    _R = net_bid_fn(prov, **REC)
    _C = net_bid_fn(prov, **CUR)


def _full_auction(seed, bid_fns):
    from _lib import deal_12_10_10
    from frontier_selfplay import _weakest_two
    sol12, d1, d2 = deal_12_10_10(seed)
    hands = [list(sol12[:10]), list(d1), list(d2)]; talon = list(sol12[10:])
    current = None; passes = 0; turn = 0; n_bids = 0
    while passes < 3:
        if current is not None and turn == current["pid"]:
            passes += 1; turn = (turn + 1) % 3; continue
        cards = list(hands[turn]) + list(talon)
        pick = bid_fns[turn](cards, current["rung"] if current else None, None)
        thresh = -PASS_PEN if current is None else -current["ev"]
        if pick is not None and pick[0] > thresh:
            ev, rung, trump, discard, hand10 = pick
            hands[turn] = hand10; talon = discard
            current = {"pid": turn, "rung": rung, "trump": trump, "ev": ev}; n_bids += 1; passes = 0
        else:
            if current is None and n_bids == 0 and turn == 0:
                disc, hand10 = _weakest_two(list(hands[turn]) + list(talon), None)
                hands[turn] = hand10; talon = disc
            passes += 1
        turn = (turn + 1) % 3
    if current is None:
        return {"winner": None}
    w = current["pid"]
    return {"winner": w, "rung": current["rung"], "contract": current["rung"].name,
            "trump": current["trump"], "sol": hands[w], "def1": hands[(w + 1) % 3],
            "def2": hands[(w + 2) % 3], "talon": talon}


def _play_seat_gp(r, seed):
    """Play one seating (r = R's seat, C at the others); return (seat_gp[3], winner_config)."""
    import harness27 as h
    from frontier_selfplay import _kontra_decision
    from scorers import resolve_bidset
    from scoring.oracle import score as osc
    bid_fns = [_R if s == r else _C for s in (0, 1, 2)]
    rr = _full_auction(seed, bid_fns)
    if rr["winner"] is None:
        return [-2 * PASS_PEN, PASS_PEN, PASS_PEN], "pass"      # opener seat 0 pays
    pos, bid = h._play_terminal(rr["rung"], rr["trump"], rr["sol"], rr["def1"], rr["def2"],
                                rr["talon"], seed)
    kontras, _ = _kontra_decision(bid, rr["trump"], rr["sol"], rr["def1"], rr["def2"],
                                  rr["talon"], seed)
    pv = osc(final_pos=pos, bid=bid, kontras=kontras)
    w = rr["winner"]
    seat_gp = [0.0, 0.0, 0.0]
    seat_gp[w] = float(pv.total_sol)
    seat_gp[(w + 1) % 3] = -float(pv.gp_vs(0)); seat_gp[(w + 2) % 3] = -float(pv.gp_vs(1))
    return seat_gp, ("R" if w == r else "C")


def _worker(seed):
    rows = []
    for r in (0, 1, 2):                            # rotate R through every seat
        seat_gp, winner = _play_seat_gp(r, seed)
        rows.append({"r_seat": r, "r_gp": seat_gp[r], "winner": winner})
    return {"seed": seed, "rot": rows}


def build(n):
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(l)["seed"] for l in open(OUT)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp32 config head-to-head: {len(seeds)} deals × 3 seatings (RECOMMENDED vs CURRENT)", flush=True)
    t0 = time.perf_counter(); done = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(OUT, "a") as o:
        for rec in pool.imap_unordered(_worker, seeds, chunksize=2):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            if done % 25 == 0:
                el = time.perf_counter() - t0
                print(f"[h2h] {done}/{len(seeds)} {el:.0f}s eta {(len(seeds)-done)/(done/el)/60:.0f}m", flush=True)
    print("done", flush=True)


def analyze():
    import collections
    rows = [r for l in open(OUT) for r in json.loads(l)["rot"]]
    n = len(rows)
    r_gp = sum(x["r_gp"] for x in rows)
    wins = collections.Counter(x["winner"] for x in rows)
    # per-seat R edge
    print(f"\nexp32 CONFIG HEAD-TO-HEAD — {n} games ({n//3} deals × 3 seatings)\n")
    print(f"  RECOMMENDED (FLOOR .80/DEBIAS .85/DURI .3) net GP/game as the lone R vs 2×CURRENT:")
    print(f"    R mean GP/game = {r_gp/n:+.3f}   (zero-sum → 0 if equal; POSITIVE = recommended beats current)")
    print(f"  bid winner: R {wins['R']} ({100*wins['R']/n:.0f}%)  C {wins['C']} ({100*wins['C']/n:.0f}%)  "
          f"pass {wins['pass']} ({100*wins['pass']/n:.0f}%)")
    for s in (0, 1, 2):
        sub = [x for x in rows if x["r_seat"] == s]
        print(f"    R at seat {s}: mean GP {sum(x['r_gp'] for x in sub)/len(sub):+.3f} (n={len(sub)})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "2000")))
    elif cmd == "analyze":
        analyze()
