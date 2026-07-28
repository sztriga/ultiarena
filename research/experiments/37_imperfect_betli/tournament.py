"""exp37 — does bidding REALISTIC / imperfect betli gain GP? Seat-isolated paired tournament.

R = the exp37 bidder (BETLI_REAL on: plain betli scored by `p_betli_real`).
C = the deployed baseline bidder (betli_real off: plain betli scored by the god `p_betli`).
Both share the promoted frontier config (FLOOR .80 / DEBIAS .85 / DURI .3) — the ONLY difference is
how plain betli is valued, so any GP delta is attributable to imperfect-betli bidding.

Per deal we take the ALL-C table (every seat baseline) as the reference, then for each seat s replace
JUST that seat's bidder with R (opponents stay baseline C) and re-run the auction. If switching seat s
changes the auction (only happens when a betli decision flips), we play both tables out with the SAME
realistic engine (harness27 PIMC play, config-independent kontra, oracle) and record
    diff_s = GP(seat s | s=R, others=C) − GP(seat s | all C).
Averaged over seat-deals this is the GP/seat-deal gain from adopting betli_real vs the deployed baseline,
against a fixed baseline table that DEFENDS with PIMC and KONTRAS realistically (bluffs can be punished).

Also records, per seat-deal, whether R bid a plain betli C wouldn't (the divergence) and whether it was
made — so we see the bid rate, realised make-rate, and contract mix of the newly-unlocked betlis.

Env: N (deals), WORKERS, PIMC_N (16). MODE=main (default). DEF=pimc|god (robustness; god defenders punish
every dd-lost bluff → the exploitative-vs-robust check).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

os.environ["KONTRA"] = "1"
os.environ.setdefault("FLOOR", "0.80")
os.environ.setdefault("DEBIAS_PCTL", "0.85")
os.environ.setdefault("DURI_TERIT_MULT", "0.3")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/29_frontier_table", f"{_REPO}/experiments/27_kontra_revamp",
           f"{_REPO}/experiments/24_bidding_loop", f"{_REPO}/experiments/23_bidding_integration",
           f"{_REPO}/experiments/14_minigame_bid_eval", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = os.environ.get("OUT") or os.path.join(_HERE, "tournament.jsonl")
SEED_BASE = 505_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
PASS_PEN = 2.0
DEF = os.environ.get("DEF", "pimc")             # robustness: defender strength in the playout
FRONTIER = dict(pctl=0.85, floor=0.80, duri_mult=0.3)

_R = _C = None


def _init():
    global _R, _C
    from provider import NetProvider
    from auction import net_bid_fn
    prov = NetProvider(calibrate=True, betli_real_dir=_HERE)     # loads betli_real head
    if "betli_real" not in prov.heads:
        raise SystemExit("betli_real_baseline.pt not found — run datagen.py + train.py first")
    _R = net_bid_fn(prov, betli_real=True, **FRONTIER)
    _C = net_bid_fn(prov, betli_real=False, **FRONTIER)


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


def _akey(rr):
    if rr["winner"] is None:
        return ("pass",)
    return (rr["winner"], rr["contract"], rr["trump"], tuple(sorted(c.id for c in rr["sol"])))


def _play_def(rung, trump, sol, d1, d2, talon, seed):
    """Play to terminal. DEF=pimc → harness27 (PIMC everyone, = deployed & datagen). DEF=god →
    PIMC soloist but GOD defenders (perfect defense punishes every dd-lost bluff)."""
    if DEF == "pimc":
        import harness27 as h
        return h._play_terminal(rung, trump, sol, d1, d2, talon, seed)
    from ulti.solvers import pis, determinize as _det
    from ulti.eval.pimc_matchup import pimc_pick
    from trickster._solver_core import set_multi_weights
    from scorers import resolve_bidset, _play_weights
    bid = resolve_bidset(rung, sol, trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        build_c, restrict, weights, solve_c = "betli", None, None, "betli"
    elif bid.durchmars and trump is None and n_trick == 1:
        build_c, restrict, weights, solve_c = "durchmars", None, None, "durchmars"
    else:
        build_c, solve_c = "parti", "multi"
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        weights = _play_weights(bid, sol, trump)
    if weights is not None:
        set_multi_weights(**weights)
    pos = pis.build_position(hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
                             contract=build_c, trump=trump, talon=list(talon),
                             declare_marriages=(trump is not None), marriage_restrict=restrict)
    pimc_n = int(os.environ.get("PIMC_N", "16"))
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        if p == 0:
            ch = pimc_pick(pos=pos, contract=solve_c, n_samples=pimc_n, seed=seed * 31337 + mi, voids_dict=vd)
        else:
            ch, _ = pis.solve_best(pos, contract=solve_c)
        if ch is None:
            ch = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, ch); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, ch); mi += 1
    return pos, bid


def _seat_gp(rr, seed):
    """Play one auction result out (kontra + oracle); return seat_gp[3]."""
    from frontier_selfplay import _kontra_decision
    from ulti.scoring.oracle import score as osc
    if rr["winner"] is None:
        return [-2 * PASS_PEN, PASS_PEN, PASS_PEN]              # opener (seat 0) forfeits
    pos, bid = _play_def(rr["rung"], rr["trump"], rr["sol"], rr["def1"], rr["def2"], rr["talon"], seed)
    kontras, _ = _kontra_decision(bid, rr["trump"], rr["sol"], rr["def1"], rr["def2"], rr["talon"], seed)
    pv = osc(final_pos=pos, bid=bid, kontras=kontras)
    w = rr["winner"]; sg = [0.0, 0.0, 0.0]
    sg[w] = float(pv.total_sol)
    sg[(w + 1) % 3] = -float(pv.gp_vs(0)); sg[(w + 2) % 3] = -float(pv.gp_vs(1))
    return sg


def _is_plain_betli(rr, s):
    return rr["winner"] == s and rr["contract"] == "betli"


def _worker(seed):
    rrC = _full_auction(seed, [_C, _C, _C])                    # all-baseline reference table
    keyC = _akey(rrC)
    gpC = _seat_gp(rrC, seed)                                   # played once, shared reference
    rows = []
    for s in (0, 1, 2):
        fns = [_R if i == s else _C for i in (0, 1, 2)]
        rrR = _full_auction(seed, fns)
        divergent = _akey(rrR) != keyC
        row = {"s": s, "divergent": divergent,
               "c_contract": rrC["contract"] if rrC["winner"] is not None else "pass",
               "r_contract": rrR["contract"] if rrR["winner"] is not None else "pass",
               "r_betli": _is_plain_betli(rrR, s), "c_betli": _is_plain_betli(rrC, s)}
        if not divergent:
            row["diff"] = 0.0
        else:
            gpR = _seat_gp(rrR, seed)
            row["diff"] = gpR[s] - gpC[s]
            row["r_seat_gp"] = gpR[s]
            row["c_seat_gp"] = gpC[s]
        rows.append(row)
    return {"seed": seed, "rows": rows}


def build(n):
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(l)["seed"] for l in open(OUT)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp37 betli-bidding tournament: {len(seeds)} deals × 3 seats  (DEF={DEF}, "
          f"R=betli_real ON vs C=baseline)", flush=True)
    t0 = time.perf_counter(); done = 0; div = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(OUT, "a") as o:
        for rec in pool.imap_unordered(_worker, seeds, chunksize=2):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            div += sum(1 for r in rec["rows"] if r["divergent"])
            if done % 25 == 0:
                el = time.perf_counter() - t0
                print(f"[tour] {done}/{len(seeds)} {el:.0f}s eta {(len(seeds)-done)/(done/el)/60:.0f}m "
                      f"divergences {div}", flush=True)
    print("done", flush=True)


def analyze():
    import collections
    recs = [json.loads(l) for l in open(OUT)]
    rows = [r for rec in recs for r in rec["rows"]]
    n = len(rows)
    diffs = [r["diff"] for r in rows]
    mean = sum(diffs) / n
    div = [r for r in rows if r["divergent"]]
    nz = [d for d in diffs if abs(d) > 1e-9]
    # paired t on the per-seat diffs
    if n > 1:
        m = mean; var = sum((d - m) ** 2 for d in diffs) / (n - 1); se = (var / n) ** 0.5
        t = m / se if se > 0 else 0.0
    else:
        t = 0.0
    out = [f"# exp37 — imperfect/bluff betli bidding tournament  (DEF={DEF})\n",
           f"{len(recs)} deals × 3 seats = {n} seat-games. R = betli_real ON, C = deployed baseline "
           f"(same frontier config otherwise). diff = GP(seat=R, opp=C) − GP(seat=C, all-C table).\n",
           f"## Headline",
           f"- **mean diff = {mean:+.4f} GP/seat-deal   (t={t:+.1f}, n={n})**  → the GP gained by adopting "
           f"betli_real vs the deployed baseline",
           f"- auction diverged (R bid a betli C wouldn't, or plain-vs-terített) on {len(div)}/{n} "
           f"seat-deals ({100*len(div)/n:.1f}%)",
           f"- GP actually changed on {len(nz)} seat-deals; mean diff there = "
           f"{sum(nz)/len(nz) if nz else 0:+.3f}"]
    if div:
        made = [r for r in div if r.get("r_betli") and r.get("r_seat_gp", 0) > 0]
        rbet = [r for r in div if r.get("r_betli")]
        out += [f"\n## The newly-unlocked betlis (R bid plain betli, C didn't): n={len(rbet)}",
                f"- realised make-rate {100*len(made)/len(rbet):.0f}% ({len(made)}/{len(rbet)} made)"
                if rbet else "- (none)",
                f"- mean diff on those = {sum(r['diff'] for r in rbet)/len(rbet):+.3f} GP/seat-deal"
                if rbet else ""]
        cc = collections.Counter(r["c_contract"] for r in rbet)
        out.append(f"- what C bid instead: " + "  ".join(f"{k}={v}" for k, v in cc.most_common(6)))
    txt = "\n".join(x for x in out if x) + "\n"
    open(os.path.join(_HERE, f"TOURNAMENT_{DEF}.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "1500")))
    elif cmd == "analyze":
        analyze()
    else:
        print(f"unknown cmd {cmd}", flush=True)
