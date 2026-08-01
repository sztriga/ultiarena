"""exp42 GATE: query-mixture duri_colored + reach100_20 heads vs the deployed frontier.

Port of the exp40 gate to the consolidated repo. Two configs, IDENTICAL except the
bidder's provider: FRONTIER = deployed heads; CAND = deployed heads with duri_colored +
reach100_20 swapped for the exp42 mixture retrains (query-distribution training,
tail-enriched isotonic). Same engine both sides (exp36 net for hidden betli, terített
reveal, PIMC play, exp27 kontra gates), so any GP delta is the two heads alone.

MATCHUP env: h2h:CAND:FRONTIER (the gate; 3 seatings per deal) | self:CAND | self:FRONTIER
Env: N, WORKERS, PIMC_N.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ["KONTRA"] = "1"
os.environ.setdefault("FLOOR", "0.80")
os.environ.setdefault("DEBIAS_PCTL", "0.85")
os.environ.setdefault("DURI_TERIT_MULT", "0.3")
os.environ.setdefault("REBETLI_FLOOR", "0.90")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO))

from ulti.solvers import pis, determinize as _det          # noqa: E402
from ulti.eval.pimc_matchup import pimc_pick               # noqa: E402
from ulti.scoring.oracle import score as oracle_score      # noqa: E402
from ulti.bidding.scorers import resolve_bidset, _play_weights, _hand_makeability  # noqa: E402
from ulti.bidding.kontra import _sol_ev                    # noqa: E402
from ulti.bidding.deal import deal_12_10_10                # noqa: E402
from ulti.bidding.ladder import contract_name              # noqa: E402
from ultisolver._solver_core import set_multi_weights      # noqa: E402
from ulti.betli import defense as _b36                     # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "6"))
PIMC_N = int(os.environ.get("PIMC_N", "16"))
PASS_PEN = 2.0
SEED_BASE = 820_000_000
CFG = dict(pctl=0.85, floor=0.80, duri_mult=0.3)
MATCHUP = os.environ.get("MATCHUP", "h2h:CAND:FRONTIER")
_BID = {}


def _weakest_two(cards12, trump):
    def junk_key(c):
        is_trump = 1 if (trump is not None and c.suit == trump) else 0
        is_seven = 1 if c.rank == "7" else 0
        return (is_trump, c.points, is_seven, c.rank_index)
    ordered = sorted(cards12, key=junk_key)
    return ordered[:2], ordered[2:]


def _kontra_primary(bid):
    """Simple-contract primary (exp29 gate semantics — combined games don't kontra here)."""
    if bid.forty_hundred or bid.twenty_hundred or bid.teritett:
        return None
    n = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if n > 1:
        return None
    if bid.betli: return "betli"
    if bid.ulti: return "ulti"
    if bid.durchmars: return "durchmars"
    return "parti"


def _kontra_decision(bid, trump, sol, d1, d2, talon, seed):
    """The deployed frontier kontra (exp27 per-unit gates) → (kontras dict, level)."""
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


def _init():
    global _BID
    from ulti.bidding.auction import net_bid_fn
    from ulti.bidding.provider import NetProvider
    b37 = str(_REPO / "models" / "ulti" / "betli")
    prov_f = NetProvider(calibrate=True, betli_real_dir=b37)
    prov_c = NetProvider(weights_dir=str(_HERE / "candidate_full"),
                         calibrate=True, betli_real_dir=b37)
    _BID = {"FRONTIER": net_bid_fn(prov_f, betli_real=True, rebetli_real=True, **CFG),
            "CAND":     net_bid_fn(prov_c, betli_real=True, rebetli_real=True, **CFG)}


def _seatings(matchup):
    p = matchup.split(":")
    if p[0] == "self":
        return [(p[1], p[1], p[1])]
    X, Y = p[1], p[2]
    return [(X, Y, Y), (Y, X, Y), (Y, Y, X)]


def _full_auction(seed, bid_fns):
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


def _play(rung, trump, sol, d1, d2, talon, seed):
    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        build_c, solve_c, restrict, weights = "betli", "betli", None, None
    elif bid.durchmars and trump is None and n_trick == 1:
        build_c, solve_c, restrict, weights = "durchmars", "durchmars", None, None
    else:
        build_c, solve_c = "parti", "multi"
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        weights = _play_weights(bid, sol, trump)
    is_terit = bool(getattr(bid, "teritett", False))
    if weights is not None:
        set_multi_weights(**weights)
    pos = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
                             contract=build_c, trump=trump, talon=list(talon),
                             declare_marriages=(trump is not None), marriage_restrict=restrict,
                             has_ulti=bool(bid.ulti))
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos); s = seed * 131 + mi
        if weights is not None:
            set_multi_weights(**weights)
        if p == 0:
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd)
        elif is_terit and mi >= 3:
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd,
                           must_hold={0: list(pis.hands_by_player(pos)[0])})
        elif solve_c == "betli" and not is_terit and _b36.available():
            mv = _b36.betli_defense_pick(pos, p)
            if mv is None:
                mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd)
        else:
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd)
        if mv is None:
            mv = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, mv); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, mv); mi += 1
    return pos, bid


def _worker(seed):
    games = []
    for seat_cfgs in _seatings(MATCHUP):
        rr = _full_auction(seed, [_BID[c] for c in seat_cfgs])
        if rr["winner"] is None:
            games.append({"pass": True, "seat_cfgs": list(seat_cfgs)}); continue
        pos, bid = _play(rr["rung"], rr["trump"], rr["sol"], rr["def1"], rr["def2"],
                         rr["talon"], seed)
        kontras, _lvl = _kontra_decision(bid, rr["trump"], rr["sol"], rr["def1"],
                                         rr["def2"], rr["talon"], seed)
        pv = oracle_score(final_pos=pos, bid=bid, kontras=kontras)
        w = rr["winner"]; sg = [0.0, 0.0, 0.0]
        sg[w] = float(pv.total_sol)
        sg[(w + 1) % 3] = -float(pv.gp_vs(0)); sg[(w + 2) % 3] = -float(pv.gp_vs(1))
        games.append({"pass": False, "seat_cfgs": list(seat_cfgs), "winner": w,
                      "contract": contract_name(bid), "seat_gp": sg})
    return {"seed": seed, "games": games}


def main():
    n = int(os.environ.get("N", "600"))
    out = _HERE / ("mu_" + MATCHUP.replace(":", "_") + ".jsonl")
    seen = set()
    if out.exists():
        seen = {json.loads(l)["seed"] for l in open(out)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp42 {MATCHUP}: {len(seeds)} deals × {len(_seatings(MATCHUP))} seatings "
          f"PIMC_N={PIMC_N} workers={WORKERS}", flush=True)
    t0 = time.perf_counter(); done = 0
    with get_context("fork").Pool(WORKERS, initializer=_init) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(_worker, seeds, chunksize=4):
            f.write(json.dumps(r) + "\n"); f.flush()
            done += 1
            if done % 20 == 0:
                el = time.perf_counter() - t0
                print(f"  {done}/{len(seeds)} deals  {el:.0f}s  eta={el/done*(len(seeds)-done):.0f}s",
                      flush=True)
    print("TOURNAMENT DONE", flush=True)


if __name__ == "__main__":
    main()
