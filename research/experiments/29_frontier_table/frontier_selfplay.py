"""exp29 — frontier self-play table: 3 frontier models bid + play, over many deals.

Faithful to the DEPLOYED engine: KONTRA=1 kontra-aware bidder (opener PASSES weak hands),
FULL auction (any seat may open after a forehand pass — not just run_auction's forehand-opens
model), PIMC play both sides, and the promoted per-unit frontier kontra (ulti trumps>=4,
colored duri trumps>=3, parti makeability<0.10, else abstain; rekontra unchanged). Oracle
scored incl. kontra + silents. Seat 0 = forehand/opener (fixed frame → positional analytics).

Records per deal: contract (or "passz"), winner seat, per-seat GP (zero-sum), soloist GP,
made, kontra level, auction length. → selfplay.jsonl (resumable). Then `analyze` → the table.
Env: N (seeds), WORKERS.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import get_context

os.environ["KONTRA"] = "1"                       # kontra-aware bidder (deployed config)
os.environ.setdefault("FLOOR", "0.7")
os.environ.setdefault("DEBIAS_PCTL", "0.80")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/27_kontra_revamp",
           f"{_REPO}/experiments/24_bidding_loop",
           f"{_REPO}/experiments/23_bidding_integration", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.path.join(_HERE, "selfplay.jsonl")
SEED_BASE = 550_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
PASS_PEN = 2.0

_BID_FN = None


def _init():
    global _BID_FN
    from net_bidder import make_net_bid_fn
    _BID_FN = make_net_bid_fn()


def _weakest_two(cards, trump):
    """The 2 junk cards a passing player buries (keep 7s; shed low non-trump)."""
    def key(c):
        return (1 if (trump is not None and c.suit == trump) else 0,
                c.points, 1 if c.rank == "7" else 0, c.rank_index)
    s = sorted(cards, key=key)
    return s[:2], s[2:]


def _full_auction(seed):
    """Faithful full auction (mirrors play.py::_advance_auction): any seat may open;
    the forehand buries 2 on an opening pass; a full round of 3 passes ends it."""
    from _lib import deal_12_10_10
    sol12, d1, d2 = deal_12_10_10(seed)
    hands = [list(sol12[:10]), list(d1), list(d2)]
    talon = list(sol12[10:])
    current = None; passes = 0; turn = 0; n_bids = 0
    while passes < 3:
        if current is not None and turn == current["pid"]:
            passes += 1; turn = (turn + 1) % 3; continue     # holder: AI never re-raises
        cards = list(hands[turn]) + list(talon)
        cur_rung = current["rung"] if current else None
        pick = _BID_FN(cards, cur_rung, None)
        thresh = -PASS_PEN if current is None else -current["ev"]
        if pick is not None and pick[0] > thresh:
            ev, rung, trump, discard, hand10 = pick
            hands[turn] = hand10; talon = discard
            current = {"pid": turn, "rung": rung, "trump": trump, "ev": ev}
            n_bids += 1; passes = 0
        else:
            if current is None and n_bids == 0 and turn == 0:    # forehand buries on opening pass
                disc, hand10 = _weakest_two(list(hands[turn]) + list(talon), None)
                hands[turn] = hand10; talon = disc
            passes += 1
        turn = (turn + 1) % 3
    if current is None:
        return {"winner": None}
    w = current["pid"]
    return {"winner": w, "rung": current["rung"], "contract": current["rung"].name,
            "trump": current["trump"], "n_bids": n_bids,
            "sol": hands[w], "def1": hands[(w + 1) % 3], "def2": hands[(w + 2) % 3], "talon": talon}


def _kontra_primary(bid):
    from bidder import _is_simple
    if not _is_simple(bid):
        return None
    if bid.betli:     return "betli"
    if bid.ulti:      return "ulti"
    if bid.durchmars: return "durchmars"
    return "parti"


def _kontra_decision(bid, trump, sol, d1, d2, talon, seed):
    """The deployed frontier kontra (per-unit gates) → (kontras dict, level)."""
    from scorers import _hand_makeability
    from kontra import _sol_ev
    prim = _kontra_primary(bid)
    if prim is None:
        return {}, 0
    def nt(hand):
        return sum(1 for c in hand if trump is not None and c.suit == trump)
    if prim == "ulti":
        kontra = max(nt(d1), nt(d2)) >= 4
    elif prim == "durchmars" and trump is not None:
        kontra = max(nt(d1), nt(d2)) >= 3
    elif prim == "parti":
        p1 = _hand_makeability(sol, d1, d2, trump, talon, "parti", 1, 6, seed + 11)
        p2 = _hand_makeability(sol, d1, d2, trump, talon, "parti", 2, 6, seed + 12)
        kontra = min(p1, p2) < 0.10
    else:
        kontra = False
    if not kontra:
        return {}, 0
    ps = _hand_makeability(sol, d1, d2, trump, talon, prim, 0, 6, seed + 23)
    lvl = 2 if _sol_ev(ps, bid, 0) > 0 else 1
    if trump is None:
        return {prim: (lvl, lvl)}, lvl
    return {prim: lvl}, lvl


def _worker(seed):
    import harness27 as h
    from scorers import resolve_bidset, _primary_made
    from ulti.scoring.oracle import score as osc
    r = _full_auction(seed)
    if r["winner"] is None:
        return {"seed": seed, "pass": True, "contract": "passz",
                "seat_gp": [-2 * PASS_PEN, PASS_PEN, PASS_PEN], "winner": 0, "n_bids": 0}
    pos, bid = h._play_terminal(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"],
                                r["talon"], seed)
    kontras, klvl = _kontra_decision(bid, r["trump"], r["sol"], r["def1"], r["def2"],
                                     r["talon"], seed)
    pv = osc(final_pos=pos, bid=bid, kontras=kontras)
    w = r["winner"]
    seat_gp = [0.0, 0.0, 0.0]
    seat_gp[w] = float(pv.total_sol)
    seat_gp[(w + 1) % 3] = -float(pv.gp_vs(0))
    seat_gp[(w + 2) % 3] = -float(pv.gp_vs(1))
    return {"seed": seed, "pass": False, "contract": r["contract"], "winner": w,
            "seat_gp": seat_gp, "soloist_gp": float(pv.total_sol),
            "per_def": float(pv.total_per_def), "made": bool(_primary_made(bid, pv)),
            "kontra": klvl, "n_bids": r["n_bids"]}


def build(n):
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(l)["seed"] for l in open(OUT)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp29 frontier self-play: {len(seeds)} deals (KONTRA=1, full auction, PIMC play)", flush=True)
    t0 = time.perf_counter(); done = 0; passes = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(OUT, "a") as o:
        for rec in pool.imap_unordered(_worker, seeds, chunksize=2):
            o.write(json.dumps(rec) + "\n"); o.flush()
            done += 1
            if rec.get("pass"):
                passes += 1
            if done % 25 == 0:
                el = time.perf_counter() - t0
                eta = (len(seeds) - done) / (done / el) if done else 0
                print(f"[selfplay] {done}/{len(seeds)} {el:.0f}s eta {eta/60:.1f}m  "
                      f"passz {100*passes/done:.0f}%", flush=True)
    print(f"done: {done} deals, {passes} passz", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "6000")))
    else:
        print(f"unknown cmd {cmd}", flush=True)
