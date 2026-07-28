"""exp40 — REALISTIC recalibration GP gate: does swapping the god base-event heads for the
realistic (PIMC-labeled) heads actually WIN games vs the current champion?

Two configs, IDENTICAL except the BIDDER'S PROVIDER:
  FRONTIER = the current champion — deployed god heads + betli_real (exp37) + rebetli (exp39).
  RECAL    = FRONTIER with the six god heads (parti, ulti, reach100_40, reach100_20, duri_colored,
             colorless_duri) SWAPPED for their exp40 `_real` heads. betli stays (exp37 already realistic).
Both play the SAME corrected engine (terített reveal + exp36 net for hidden betli/rebetli; soloist PIMC;
kontra-aware). So any GP delta is the recalibration alone. Contracts recorded as the RESOLVED game.

Matchups (`MATCHUP` env):
  self:FRONTIER  → baseline contract mix + negative-games (per-contract mean/SE).
  self:RECAL     → does the AI now RISK duri/ulti more (the god-vs-real gap)? does the 40-100-duri
                   family improve? new leaks?
  h2h:RECAL:FRONTIER → the GATE: RECAL's GP/game as the lone player vs 2×FRONTIER.
`report` reads all three → RESULTS.md.

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
SEED_BASE = 700_000_000
FRONTIER = dict(pctl=0.85, floor=0.80, duri_mult=0.3)
MATCHUP = os.environ.get("MATCHUP", "h2h:RECAL:FRONTIER")
# PIMC-32 matched recalibration: the 3 heads that bled (ulti + both duri), retrained AND evaluated vs
# the SAME strong opponent (PIMC-32) → no train≠test optimism. parti/reach keep god (no bleed).
_RECAL_HEADS = ("duri_colored", "colorless_duri", "ulti")
_BID = {}


def _outfile(m):
    return os.path.join(_HERE, "mu_" + m.replace(":", "_") + ".jsonl")


def _load_recal(prov):
    """Swap the six god heads for their exp40 realistic counterparts (drop-in: same featurize dim,
    same provider head-name keys → base_probs returns realistic p_* automatically)."""
    import numpy as np
    import torch
    from train_base_head import Head
    for h in _RECAL_HEADS:
        ck = torch.load(os.path.join(_HERE, f"{h}_real_baseline.pt"), weights_only=False)
        m = Head(ck["in_dim"]); m.load_state_dict(ck["state_dict"]); m.eval()
        prov.heads[h] = m
        cp = os.path.join(_HERE, f"{h}_real_isotonic.npz")
        if os.path.exists(cp):
            z = np.load(cp); prov.calib[h] = (z["x"], z["y"])
    return prov


def _init():
    global _BID
    from provider import NetProvider
    from auction import net_bid_fn
    import infer  # noqa: F401
    b37 = f"{_REPO}/experiments/37_imperfect_betli"
    prov_god = NetProvider(calibrate=True, betli_real_dir=b37)
    prov_recal = _load_recal(NetProvider(calibrate=True, betli_real_dir=b37))
    # both configs run the CURRENT champion bidder flags (betli_real + rebetli ON) — only the heads differ
    _BID = {"FRONTIER": net_bid_fn(prov_god, betli_real=True, rebetli_real=True, **FRONTIER),
            "RECAL":    net_bid_fn(prov_recal, betli_real=True, rebetli_real=True, **FRONTIER)}


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
    print(f"exp40 {MATCHUP}: {len(seeds)} deals × {len(_seatings(MATCHUP))} seatings, PIMC_N={PIMC_N}", flush=True)
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
    L = ["# exp40 — realistic recalibration GP gate: does swapping god→realistic heads win games?\n",
         f"FRONTIER (current champion: deployed god heads + betli_real + rebetli) vs RECAL (same, with these "
         f"heads swapped for exp40 realistic heads: **{', '.join(_RECAL_HEADS)}**). Identical corrected "
         f"engine; soloist PIMC. Any delta = the recalibration alone.\n"]

    fr = _ladder("self:FRONTIER"); rc = _ladder("self:RECAL")

    # (1) GP-GATE (h2h) FIRST — the headline
    L.append("## (1) GP GATE — RECAL vs FRONTIER head-to-head")
    r = _selfplay_rows("h2h:RECAL:FRONTIER")
    if r:
        rows, npass = r
        xg = [g["seat_gp"][g["seat_cfgs"].index("RECAL")] for g in rows
              if "RECAL" in g["seat_cfgs"] and g["seat_cfgs"].count("RECAL") == 1]
        m, se, n = _stat(xg)
        sig = "SIGNIFICANT +" if m - 1.96 * se > 0 else ("SIGNIFICANT −" if m + 1.96 * se < 0 else "within noise")
        L.append(f"- **RECAL (lone) vs 2×FRONTIER: {m:+.4f} GP/game (±{1.96*se:.3f}, n={n}; {npass} redealt) "
                 f"→ {sig}** — 0 = tie, + = recalibration helps.")
    else:
        L.append("- (h2h not run yet)")

    # (2) Contract mix — does the AI now RISK duri/ulti more?
    L.append("\n## (2) Contract mix — FRONTIER vs RECAL self-play (does it risk duri/ulti more?)")
    if fr and rc:
        (fb, ftot, _), (rb, rtot, _) = fr, rc
        keys = sorted(set(fb) | set(rb), key=lambda k: -(len(rb.get(k, [])) + len(fb.get(k, []))))
        L.append("| contract | FRONTIER %·GP | RECAL %·GP |")
        L.append("|---|---|---|")
        for k in keys:
            fv, rv = fb.get(k, []), rb.get(k, [])
            if len(fv) + len(rv) < 5:
                continue
            fs = f"{100*len(fv)/ftot:.1f}% · {sum(fv)/len(fv):+.2f}" if fv else "—"
            rs = f"{100*len(rv)/rtot:.1f}% · {sum(rv)/len(rv):+.2f}" if rv else "—"
            L.append(f"| {k} | {fs} | {rs} |")

        def _fam(byc, tot, needle):
            v = [g for k, gg in byc.items() if needle in k for g in gg]
            return 100 * len(v) / tot if tot else 0.0
        L.append("\n**family shares (any rung containing the word):**")
        for needle in ("duri", "ulti", "betli", "parti", "40-100"):
            L.append(f"- {needle}: FRONTIER {_fam(fb, ftot, needle):.1f}% → RECAL {_fam(rb, rtot, needle):.1f}%")
        L.append(f"- passz: FRONTIER {100*fr[2]/(ftot+fr[2]):.1f}% → RECAL {100*rc[2]/(rtot+rc[2]):.1f}%")

    # (3) Negative-games — did the 40-100-duri leak improve? new leaks?
    L.append("\n## (3) Negative contracts — FRONTIER vs RECAL (did the 40-100-duri leak close? new leaks?)")
    for tag, lad in (("FRONTIER", fr), ("RECAL", rc)):
        if not lad:
            continue
        byc, tot, _ = lad
        neg = []
        for k, v in byc.items():
            m, se, n = _stat(v)
            if m < 0 and n >= 3:
                z = abs(m) / se if se > 0 else 0.0
                neg.append((k, n, m, se, z))
        neg.sort(key=lambda x: x[2])
        L.append(f"\n### {tag}")
        L.append("| contract | n | GP/bid | 95% CI | |mean|/SE | verdict |")
        L.append("|---|---|---|---|---|---|")
        for k, n, m, se, z in neg:
            verdict = "**real −EV**" if z >= 2 else ("leaning −" if z >= 1 else "noise")
            L.append(f"| {k} | {n} | {m:+.2f} | ±{1.96*se:.1f} | {z:.1f} | {verdict} |")
        real = [k for k, n, m, se, z in neg if z >= 2]
        L.append(f"- **{tag}: {len(real)} significantly −EV: {', '.join(real) if real else 'none'}**")

    txt = "\n".join(L) + "\n"
    open(os.path.join(_HERE, "TOURNAMENT.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "8000")))
    elif cmd == "report":
        report()
    else:
        print(f"unknown cmd {cmd}", flush=True)
