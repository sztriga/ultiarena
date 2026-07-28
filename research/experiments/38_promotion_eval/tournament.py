"""exp38 — PROMOTION EVAL: FRONTIER vs exp36 vs exp37, pairwise head-to-head + full-ladder tables.

Three agents:
  FRONTIER = deployed bidder (betli_real OFF) + PIMC defense.
  exp36    = SAME bidder, but the learned NET defends PLAIN betli (differs only on DEFENSE — fires only
             vs a plain-betli bidder, which only exp37 is).
  exp37    = betli_real bidder (bids imperfect PLAIN betli) + PIMC defense (differs only on BIDDING).
ALL run the CORRECTED engine (terített reveal ON). Soloist play = PIMC (config-independent, unbiased).

Why PAIRWISE, not a 3-way table: in a 3-way table exp36 & FRONTIER are always CO-defenders of exp37's
betli, share the outcome, and exp36 reads as exactly = FRONTIER (its value masked). "X alone vs 2×Y"
(exp32 style) makes exp37's betli face a PURE PIMC wall or a PURE net wall → clean isolation.

Matchups (`MATCHUP` env; each writes its own jsonl):
  h2h:exp37:FRONTIER  → exp37's raw bidding gain vs PIMC defenders.
  h2h:exp36:FRONTIER  → exp36's standalone value (≈0 — FRONTIER never bids plain betli to defend).
  h2h:exp37:exp36     → exp37's gain vs NET defenders; (h2h#1 − this) = exp36's defensive value.
  self:FRONTIER / self:exp37 → the full-ladder table (contract mix + soloist GP) per config.
                               (exp36's ladder ≡ FRONTIER's — same bidder — noted, not re-run.)
`report` reads them all → RESULTS.md.

Env: N (deals), WORKERS, PIMC_N, MATCHUP.
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
SEED_BASE = 560_000_000
FRONTIER = dict(pctl=0.85, floor=0.80, duri_mult=0.3)
MATCHUP = os.environ.get("MATCHUP", "h2h:exp37:FRONTIER")
_MATCHUPS = ["h2h:exp37:FRONTIER", "h2h:exp36:FRONTIER", "h2h:exp37:exp36", "self:FRONTIER", "self:exp37"]

_BID = {}


def _outfile(m):
    return os.path.join(_HERE, "mu_" + m.replace(":", "_") + ".jsonl")


def _init():
    global _BID
    from provider import NetProvider
    from auction import net_bid_fn
    import infer  # noqa: F401  (ensures the exp36 net is importable in the worker)
    prov = NetProvider(calibrate=True, betli_real_dir=f"{_REPO}/experiments/37_imperfect_betli")
    base = net_bid_fn(prov, betli_real=False, **FRONTIER)
    _BID = {"FRONTIER": base, "exp36": base,
            "exp37": net_bid_fn(prov, betli_real=True, **FRONTIER)}


def _seatings(matchup):
    p = matchup.split(":")
    if p[0] == "self":
        return [(p[1], p[1], p[1])]
    X, Y = p[1], p[2]
    return [(X, Y, Y), (Y, X, Y), (Y, Y, X)]      # X alone at seat 0/1/2


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


def _play_configs(rung, trump, sol, d1, d2, talon, play_configs, seed):
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
        elif is_terit and mi >= 3:                                 # terített reveal (all configs)
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=s, voids_dict=vd,
                           must_hold={0: list(pis.hands_by_player(pos)[0])})
        elif solve_c == "betli" and play_configs[p] == "exp36":    # exp36 net defends PLAIN betli
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


def _seat_gp(rr, play_configs, seed):
    from frontier_selfplay import _kontra_decision
    from ladder import contract_name
    pos, bid = _play_configs(rr["rung"], rr["trump"], rr["sol"], rr["def1"], rr["def2"],
                             rr["talon"], play_configs, seed)
    kontras, _ = _kontra_decision(bid, rr["trump"], rr["sol"], rr["def1"], rr["def2"], rr["talon"], seed)
    pv = oracle_score(final_pos=pos, bid=bid, kontras=kontras)
    w = rr["winner"]; sg = [0.0, 0.0, 0.0]
    sg[w] = float(pv.total_sol)
    sg[(w + 1) % 3] = -float(pv.gp_vs(0)); sg[(w + 2) % 3] = -float(pv.gp_vs(1))
    # the SPECIFIC game actually played (resolve_bidset picks one of a rung's interchangeable
    # contracts) — NOT the rung's "X ≡ Y" label, which conflates two distinct games/rules.
    return sg, contract_name(bid)


def _worker(seed):
    games = []
    for seat_cfgs in _seatings(MATCHUP):
        rr = _full_auction(seed, [_BID[c] for c in seat_cfgs])
        if rr["winner"] is None:
            games.append({"pass": True, "seat_cfgs": list(seat_cfgs)}); continue
        w = rr["winner"]
        play_configs = [seat_cfgs[(w + p) % 3] for p in range(3)]
        sg, resolved = _seat_gp(rr, play_configs, seed)
        games.append({"pass": False, "seat_cfgs": list(seat_cfgs), "winner": w,
                      "contract": resolved, "seat_gp": sg})
    return {"seed": seed, "games": games}


def build(n):
    out = _outfile(MATCHUP)
    seen = set()
    if os.path.exists(out):
        seen = {json.loads(l)["seed"] for l in open(out)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp38 {MATCHUP}: {len(seeds)} deals × {len(_seatings(MATCHUP))} seatings, PIMC_N={PIMC_N}", flush=True)
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
    return m, (m / se if se > 0 else 0.0), n


def _h2h_number(matchup):
    """X's mean GP/game as the lone X vs 2×Y (zero-sum → 0 if equal, + = X beats Y)."""
    p = matchup.split(":"); X, Y = p[1], p[2]
    out = _outfile(matchup)
    if not os.path.exists(out):
        return None
    xg = []
    npass = 0
    for l in open(out):
        for g in json.loads(l)["games"]:
            if g.get("pass"):
                npass += 1; continue
            xi = [i for i, c in enumerate(g["seat_cfgs"]) if c == X][0]  # X's seat (lone)
            xg.append(g["seat_gp"][xi])
    m, t, n = _stat(xg)
    return X, Y, m, t, n, npass


def _ladder(matchup):
    """From self-play: per contract, freq + mean soloist GP."""
    import collections
    out = _outfile(matchup)
    if not os.path.exists(out):
        return None
    byc = collections.defaultdict(list); total = 0; npass = 0
    for l in open(out):
        for g in json.loads(l)["games"]:
            if g.get("pass"):
                npass += 1; continue
            total += 1
            byc[g["contract"]].append(g["seat_gp"][g["winner"]])
    return byc, total, npass


def report():
    lines = ["# exp38 — promotion eval: FRONTIER vs exp36 vs exp37\n",
             "Pairwise head-to-head (X alone vs 2×Y) + full-ladder self-play. Corrected engine "
             "(terített reveal ON); soloist play = PIMC (config-independent).\n",
             "## (1) HEAD-TO-HEAD — X's mean GP/game as the lone X vs 2×Y  (0 = tie, + = X wins)"]
    nums = {}
    for m in ("h2h:exp37:FRONTIER", "h2h:exp36:FRONTIER", "h2h:exp37:exp36"):
        r = _h2h_number(m)
        if r is None:
            lines.append(f"- {m}: (no data)"); continue
        X, Y, mean, t, n, npass = r
        nums[m] = mean
        lines.append(f"- **{X} vs 2×{Y}:  {mean:+.4f} GP/game**  (t={t:+.1f}, n={n}; {npass} redealt)")
    if "h2h:exp37:FRONTIER" in nums and "h2h:exp37:exp36" in nums:
        dv = nums["h2h:exp37:FRONTIER"] - nums["h2h:exp37:exp36"]
        lines.append(f"\n- **exp36's defensive value** = (exp37 gain vs PIMC) − (exp37 gain vs net) = "
                     f"{dv:+.4f} GP/game — how much exp36's net defense claws back from exp37's betlis.")
        lines.append(f"- **exp36's standalone value** (vs FRONTIER, no plain-betli bidder present) = "
                     f"{nums.get('h2h:exp36:FRONTIER', 0):+.4f} — expected ≈0.")

    lines.append("\n## (2) FULL-LADDER TABLE — contract mix + soloist GP/bid, per config")
    for cfg, m in (("FRONTIER", "self:FRONTIER"), ("exp37", "self:exp37")):
        r = _ladder(m)
        if r is None:
            lines.append(f"\n### {cfg}: (no data)"); continue
        byc, total, npass = r
        lines.append(f"\n### {cfg}  (self-play, {total} contracts played, {npass} all-pass/redealt)")
        lines.append("| contract | count | % | soloist GP/bid |")
        lines.append("|---|---|---|---|")
        for k, v in sorted(byc.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {k} | {len(v)} | {100*len(v)/total:.1f}% | {sum(v)/len(v):+.2f} |")
        lines.append(f"| **ALL** | {total} | 100% | **{sum(sum(v) for v in byc.values())/total:+.2f}** |")
    lines.append("\n_(exp36's ladder ≡ FRONTIER's — identical bidder; exp36 differs only on defense.)_")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(_HERE, "RESULTS.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "4000")))
    elif cmd == "report":
        report()
    else:
        print(f"unknown cmd {cmd}", flush=True)
