"""exp39 — REBETLI extension analysis: does letting the bidder escalate to rebetli help?

Two configs, IDENTICAL except the bidder:
  FRONTIER = the deployed frontier (betli_real ON, rebetli OFF) + exp36 net defense + terített reveal.
  REBETLI  = FRONTIER + rebetli_real ON (rebetli scored by the realistic prob, gated at REBETLI_FLOOR
             0.90 → only escalates to the hidden 10p doubling when the realistic make is near-certain).
Both play the CORRECTED engine (terített reveal), defend PLAIN betli/rebetli with the exp36 net, soloist
= PIMC (config-independent). Contracts recorded as the RESOLVED game (not the "X ≡ Y" rung label).

Matchups (`MATCHUP` env):
  self:FRONTIER  (large N) → baseline ladder + the NEGATIVE-GAMES investigation (are the −GP duri-combined
                 rows a real leak or low-count noise? — per-contract mean/SE/n).
  self:REBETLI   (large N) → does rebetli appear, at what freq/GP, and what does it displace (betli? ulti?).
  h2h:REBETLI:FRONTIER → the game-points impact (REBETLI's GP/game as the lone player vs 2×FRONTIER).
`report` reads them all → RESULTS.md.

Env: N, WORKERS, PIMC_N, MATCHUP.
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
os.environ.setdefault("REBETLI_FLOOR", "0.90")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/37_imperfect_betli", f"{_REPO}/experiments/36_betli_defense",
           f"{_REPO}/experiments/29_frontier_table", f"{_REPO}/experiments/27_kontra_revamp",
           f"{_REPO}/experiments/24_bidding_loop", f"{_REPO}/experiments/23_bidding_integration",
           f"{_REPO}/experiments/14_minigame_bid_eval", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ulti.solvers import pis, determinize as _det                       # noqa: E402
from ulti.eval.pimc_matchup import pimc_pick                            # noqa: E402
from ulti.scoring.oracle import score as oracle_score                  # noqa: E402

WORKERS = int(os.environ.get("WORKERS", "8"))
PIMC_N = int(os.environ.get("PIMC_N", "16"))
PASS_PEN = 2.0
SEED_BASE = 590_000_000
FRONTIER = dict(pctl=0.85, floor=0.80, duri_mult=0.3)
MATCHUP = os.environ.get("MATCHUP", "self:FRONTIER")
_BID = {}


def _outfile(m):
    return os.path.join(_HERE, "mu_" + m.replace(":", "_") + ".jsonl")


def _init():
    global _BID
    from provider import NetProvider
    from auction import net_bid_fn
    import infer  # noqa: F401
    prov = NetProvider(calibrate=True, betli_real_dir=f"{_REPO}/experiments/37_imperfect_betli")
    _BID = {"FRONTIER": net_bid_fn(prov, betli_real=True, rebetli_real=False, **FRONTIER),
            "REBETLI":  net_bid_fn(prov, betli_real=True, rebetli_real=True,  **FRONTIER)}


def _seatings(matchup):
    p = matchup.split(":")
    if p[0] == "self":
        return [(p[1], p[1], p[1])]
    X, Y = p[1], p[2]
    return [(X, Y, Y), (Y, X, Y), (Y, Y, X)]


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


def _play(rung, trump, sol, d1, d2, talon, seed):
    """Corrected engine: terített reveal + exp36 net for hidden betli/rebetli; PIMC else. Soloist PIMC."""
    from scorers import resolve_bidset, _play_weights
    from trickster._solver_core import set_multi_weights
    import infer
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
                             declare_marriages=(trump is not None), marriage_restrict=restrict)
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos); s = seed * 131 + mi
        if p == 0:
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd)
        elif is_terit and mi >= 3:                                 # terített reveal
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd,
                           must_hold={0: list(pis.hands_by_player(pos)[0])})
        elif solve_c == "betli" and not is_terit:                  # exp36 net (hidden betli/rebetli)
            mv = infer.betli_defense_pick(pos, p)
            if mv is None:
                mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd)
        else:
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd)
        if mv is None:
            mv = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, mv); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, mv); mi += 1
    return pos, bid


def _seat_gp(rr, seed):
    from frontier_selfplay import _kontra_decision
    from ladder import contract_name
    pos, bid = _play(rr["rung"], rr["trump"], rr["sol"], rr["def1"], rr["def2"], rr["talon"], seed)
    kontras, _ = _kontra_decision(bid, rr["trump"], rr["sol"], rr["def1"], rr["def2"], rr["talon"], seed)
    pv = oracle_score(final_pos=pos, bid=bid, kontras=kontras)
    w = rr["winner"]; sg = [0.0, 0.0, 0.0]
    sg[w] = float(pv.total_sol)
    sg[(w + 1) % 3] = -float(pv.gp_vs(0)); sg[(w + 2) % 3] = -float(pv.gp_vs(1))
    return sg, contract_name(bid)


def _worker(seed):
    games = []
    for seat_cfgs in _seatings(MATCHUP):
        rr = _full_auction(seed, [_BID[c] for c in seat_cfgs])
        if rr["winner"] is None:
            games.append({"pass": True}); continue
        w = rr["winner"]
        sg, resolved = _seat_gp(rr, seed)
        games.append({"pass": False, "seat_cfgs": list(seat_cfgs), "winner": w,
                      "contract": resolved, "seat_gp": sg})
    return {"seed": seed, "games": games}


def build(n):
    out = _outfile(MATCHUP)
    seen = set()
    if os.path.exists(out):
        seen = {json.loads(l)["seed"] for l in open(out)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp39 {MATCHUP}: {len(seeds)} deals × {len(_seatings(MATCHUP))} seatings, PIMC_N={PIMC_N}", flush=True)
    t0 = time.perf_counter(); done = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init) as pool, open(out, "a") as o:
        for rec in pool.imap_unordered(_worker, seeds, chunksize=2):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            if done % 50 == 0:
                el = time.perf_counter() - t0
                print(f"[{MATCHUP}] {done}/{len(seeds)} {el:.0f}s eta {(len(seeds)-done)/(done/el)/60:.0f}m", flush=True)
    print("done", flush=True)


def _stat(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1); se = (var / n) ** 0.5
    return m, se, n


def _selfplay_rows(matchup):
    out = _outfile(matchup)
    if not os.path.exists(out):
        return None
    rows = []; npass = 0
    for l in open(out):
        for g in json.loads(l)["games"]:
            if g.get("pass"):
                npass += 1
            else:
                rows.append(g)
    return rows, npass


def _ladder(matchup):
    import collections
    r = _selfplay_rows(matchup)
    if r is None:
        return None
    rows, npass = r
    byc = collections.defaultdict(list)
    for g in rows:
        byc[g["contract"]].append(g["seat_gp"][g["winner"]])
    return byc, len(rows), npass


def report():
    import collections
    L = ["# exp39 — rebetli extension: does escalating to rebetli help?\n",
         "FRONTIER (deployed: betli_real ON, rebetli OFF) vs REBETLI (+ rebetli_real, floor "
         f"{os.environ.get('REBETLI_FLOOR','0.90')}). Corrected engine (terített reveal + exp36 net "
         "defense); soloist PIMC.\n"]

    # (1) does rebetli appear + what it displaces
    fr = _ladder("self:FRONTIER"); rb = _ladder("self:REBETLI")
    L.append("## (1) Contract mix — FRONTIER vs REBETLI self-play (does rebetli appear? eat what?)")
    if fr and rb:
        (fb, ftot, _), (rbb, rtot, _) = fr, rb
        keys = ("betli", "rebetli", "teritett betli", "piros ulti", "ulti", "piros parti")
        L.append("| contract | FRONTIER %·GP | REBETLI %·GP |")
        L.append("|---|---|---|")
        for k in keys:
            fv = fb.get(k, []); rv = rbb.get(k, [])
            fs = f"{100*len(fv)/ftot:.1f}% · {sum(fv)/len(fv):+.2f}" if fv else "—"
            rs = f"{100*len(rv)/rtot:.1f}% · {sum(rv)/len(rv):+.2f}" if rv else "—"
            L.append(f"| {k} | {fs} | {rs} |")
        rv = rbb.get("rebetli", [])
        if rv:
            m, se, n = _stat(rv)
            L.append(f"\n- **rebetli appears in REBETLI: {n} bids ({100*n/rtot:.2f}% of games), "
                     f"GP/bid {m:+.2f} ± {1.96*se:.2f}**")
            L.append(f"- plain-betli share FRONTIER {100*len(fb.get('betli',[]))/ftot:.2f}% → "
                     f"REBETLI {100*len(rbb.get('betli',[]))/rtot:.2f}%  (did rebetli eat plain betli?)")
            L.append(f"- ulti share FRONTIER {100*len(fb.get('ulti',[]))/ftot:.2f}% → REBETLI "
                     f"{100*len(rbb.get('ulti',[]))/rtot:.2f}%  (did it eat ulti? — the v1 failure mode)")
        else:
            L.append("\n- rebetli did NOT appear even with the extension (floor too high for this N).")

    # (2) game-points impact — h2h
    L.append("\n## (2) GAME-POINTS IMPACT — REBETLI vs FRONTIER head-to-head")
    r = _selfplay_rows("h2h:REBETLI:FRONTIER")
    if r:
        rows, npass = r
        xg = [g["seat_gp"][g["seat_cfgs"].index("REBETLI")] for g in rows
              if "REBETLI" in g["seat_cfgs"] and g["seat_cfgs"].count("REBETLI") == 1]
        m, se, n = _stat(xg)
        L.append(f"- **REBETLI (lone) vs 2×FRONTIER: {m:+.4f} GP/game (±{1.96*se:.3f}, n={n}; {npass} redealt)** "
                 f"— 0 = tie, + = rebetli helps.")

    # (3) NEGATIVE-GAMES investigation (Task D) — from the large FRONTIER self-play
    L.append("\n## (3) The NEGATIVE contracts — real leak or low-count noise? (FRONTIER self-play)")
    if fr:
        fb, ftot, _ = fr
        neg = []
        for k, v in fb.items():
            m, se, n = _stat(v)
            if m < 0:
                z = abs(m) / se if se > 0 else 0.0
                neg.append((k, n, m, se, z))
        neg.sort(key=lambda x: x[2])
        L.append("| contract | n | GP/bid | 95% CI | |mean|/SE | verdict |")
        L.append("|---|---|---|---|---|---|")
        for k, n, m, se, z in neg:
            verdict = "**real −EV**" if z >= 2 else ("leaning −" if z >= 1 else "noise (CI spans 0)")
            L.append(f"| {k} | {n} | {m:+.2f} | ±{1.96*se:.1f} | {z:.1f} | {verdict} |")
        real = [k for k, n, m, se, z in neg if z >= 2]
        L.append(f"\n- **{len(real)} contract(s) are significantly −EV (|mean|/SE ≥ 2): "
                 f"{', '.join(real) if real else 'none'}** — the rest are consistent with noise at this N.")
    txt = "\n".join(L) + "\n"
    open(os.path.join(_HERE, "RESULTS.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "6000")))
    elif cmd == "report":
        report()
    else:
        print(f"unknown cmd {cmd}", flush=True)
