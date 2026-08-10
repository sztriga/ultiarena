"""exp48 step 2 — does NOT feeding actually gain GP when the game is played on?

Step 1 established that 84.4% of fed point-cards are ties: a zero-point card scored
identically under the solver's objective. That is a statement about the OBJECTIVE, which
is a sum of binary indicators and cannot see card points. It is not yet a statement about
GP.

milan's objection is the right one: sometimes you keep the ten because it wins a critical
trick later, and a rule that always sheds it is wrong. So before building any policy, test
the counterfactual directly.

METHOD. For each position where the AI fed a ten while a zero-point card was tied-optimal,
replay the REAL deal from that ply twice — once with the card actually played, once with
the best zero-point tied alternative — continuing both with the DEPLOYED play stack and
scoring both with the oracle. Same seed for both branches, so the only difference is the
one card.

This answers the question with the real layout rather than a model: if shedding the ten
gains nothing once the hand is played out, milan's objection kills the idea and we stop.
If it gains, the follow-up question is whether an exploit-style rollout can PICK correctly
without knowing the layout — which is step 3.

Run:  WORKERS=8 python3 counterfactual.py [N]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

from ulti.config import apply_deploy_defaults

apply_deploy_defaults()

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP43 = os.path.join(os.path.dirname(_HERE), "43_kontra_signals")
CORPUS = os.path.join(os.path.dirname(_HERE), "47_overnight", "natural.jsonl")
if _EXP43 not in sys.path:
    sys.path.insert(0, _EXP43)
OUT = os.path.join(_HERE, "counterfactual.jsonl")


def _continue_and_score(job, first_card_id):
    """Play the deal on from `ply` after forcing `first_card_id`, with the deployed stack."""
    from apps.api import ai_worker
    from apps.api.engine import _BETLI_DEF, _EXPLOIT, _MIX_EQUIV, _exp36
    from ulti.card import card_from_id
    from ulti.scoring.oracle import score as score_oracle, soloist_points
    from ulti.solvers import determinize as _det, pis
    from ultisolver._solver_core import set_multi_weights

    set_multi_weights(**(job["weights"] or {}))
    pos = pis.build_position(
        hands=[[card_from_id(i) for i in h] for h in job["hands0"]],
        soloist=0, leader=0, contract=job["build_c"], trump=job["trump"],
        talon=[card_from_id(i) for i in job["talon"]],
        declare_marriages=(job["trump"] is not None),
        marriage_restrict=job["restrict"], has_ulti=job["has_ulti"])
    voids = _det.Voids()
    hist = []
    for p, cid in job["hist"][:job["ply"]]:
        c = card_from_id(cid)
        voids.observe(pos, p, c)
        pis.apply_move(pos, c)
        hist.append((p, cid))
    # the forced card
    c0 = card_from_id(first_card_id)
    voids.observe(pos, pis.current_player(pos), c0)
    pis.apply_move(pos, c0)
    hist.append((pis.current_player(pos), first_card_id))

    solve_c = job["solve_c"]
    is_terit = bool(job.get("teritett"))
    ctr = job["seed"] * 31337 + job["ply"] * 101
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        ctr += 1
        ch = None
        if (_BETLI_DEF and p != 0 and solve_c == "betli" and not is_terit
                and _exp36 is not None and _exp36.available()):
            ch = _exp36.betli_defense_pick(pos, p)
        if ch is None:
            mode = ("exploit" if (_EXPLOIT and p == 0 and not is_terit)
                    else ("pimc_pinned" if (p != 0 and is_terit) else "pimc"))
            cid = ai_worker.op_ai_pick({
                "hands0": job["hands0"], "talon": job["talon"],
                "build_c": job["build_c"], "solve_c": solve_c, "trump": job["trump"],
                "restrict": job["restrict"], "has_ulti": job["has_ulti"],
                "weights": job["weights"], "voids": voids.as_dict(),
                "history": [(a, b) for a, b in hist], "mode": mode,
                "seed": ctr, "bid": job["bid"]})
            ch = card_from_id(cid) if cid is not None else None
        if ch is None:
            ch = random.Random(ctr).choice(pis.legal_actions(pos))
        voids.observe(pos, p, ch)
        pis.apply_move(pos, ch)
        hist.append((p, ch.id))
    return (float(score_oracle(final_pos=pos, bid=job["bid"]).total_sol),
            int(soloist_points(pos)))


def _worker(job):
    try:
        played, pts_played = _continue_and_score(job, job["played_id"])
        banked, pts_banked = _continue_and_score(job, job["zero_id"])
        return {"seed": job["seed"], "ply": job["ply"], "mover": job["mover"],
                "sol_gp_played": played, "sol_gp_banked": banked,
                "delta_sol": banked - played,
                "sol_pts_played": pts_played, "sol_pts_banked": pts_banked,
                "delta_pts": pts_banked - pts_played}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def build_jobs(n):
    """Positions where a ten was fed to the opponent AND a zero-point card was tied."""
    from ulti.bidding.scorers import resolve_bidset, _play_weights
    from ulti.bidding.ladder import LADDER
    from ulti.card import RANK_POINTS, card_from_id
    from ulti.solvers import pis
    from ultisolver._solver_core import set_multi_weights
    rng = random.Random(482)
    deals = [json.loads(l) for l in open(CORPUS) if '"kept": true' in l]
    rng.shuffle(deals)
    jobs = []
    for rec in deals:
        try:
            rung = next(x for x in LADDER if x.index == rec["rung_index"])
            trump = rec["trump"]
            H = [[card_from_id(i) for i in h] for h in (rec["sol"], rec["d1"], rec["d2"])]
            bid = resolve_bidset(rung, H[0], trump)
            nt = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
            if bid.betli:
                bc, sc, t, rs, w = "betli", "betli", None, None, None
            elif bid.durchmars and rung.colorless and nt == 1:
                bc, sc, t, rs, w = "durchmars", "durchmars", None, None, None
            else:
                bc, sc, t = "parti", "multi", trump
                rs = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
                w = _play_weights(bid, H[0], trump)
            set_multi_weights(**(w or {}))
            pos = pis.build_position(hands=[list(x) for x in H], soloist=0, leader=0,
                                     contract=bc, trump=t,
                                     talon=[card_from_id(i) for i in rec["talon"]],
                                     declare_marriages=(t is not None),
                                     marriage_restrict=rs, has_ulti=bool(bid.ulti))
            hist = [(p, card_from_id(c)) for p, c in rec["hist"]]
            for k in range(10):
                trick = hist[k * 3:(k + 1) * 3]
                info = {}
                for j, (p, card) in enumerate(trick):
                    info[p] = (list(pis.legal_actions(pos)), k * 3 + j, pis.solve_all(pos, contract=sc))
                    pis.apply_move(pos, card)
                if k < 9:
                    winner = pis.current_player(pos)
                else:
                    led = trick[0][1].suit
                    best = None
                    for p, c in trick:
                        key = (2 if (trump and c.suit == trump) else (1 if c.suit == led else 0),
                               c.rank_index)
                        if best is None or key > best[0]:
                            best = (key, p)
                    winner = best[1]
                for p, card in trick:
                    if RANK_POINTS[card.rank] != 10:
                        continue
                    enemy = (winner == 0) if p != 0 else (winner != 0)
                    if not enemy:
                        continue
                    legal, ply, vals = info[p]
                    if not vals:
                        continue
                    bestv = max(vals.values()) if p == 0 else min(vals.values())
                    zeros = [c for c, v in vals.items()
                             if RANK_POINTS[c.rank] == 0 and abs(v - bestv) < 1e-6]
                    if abs(vals.get(card, bestv) - bestv) > 1e-6 or not zeros:
                        continue
                    jobs.append({"seed": rec["seed"], "ply": ply, "mover": p,
                                 "played_id": card.id, "zero_id": zeros[0].id,
                                 "hands0": [rec["sol"], rec["d1"], rec["d2"]],
                                 "talon": rec["talon"], "build_c": bc, "solve_c": sc,
                                 "trump": t, "restrict": rs, "weights": w,
                                 "has_ulti": bool(bid.ulti), "bid": bid,
                                 "teritett": bool(getattr(bid, "teritett", False)),
                                 "hist": [[a, b.id] for a, b in hist]})
        except Exception:
            pass
        if len(jobs) >= n:
            break
    return jobs[:n]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(os.environ.get("WORKERS", "8"))
    print(f"[exp48cf] collecting {n} fed-ten positions with a tied zero-point alternative",
          flush=True)
    jobs = build_jobs(n)
    print(f"[exp48cf] replaying {len(jobs)} positions x 2 branches, {workers} workers",
          flush=True)
    t0 = time.perf_counter()
    res = []
    with get_context("fork").Pool(workers) as pool, open(OUT, "w") as o:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            if r and "error" not in r:
                res.append(r)
                o.write(json.dumps(r) + "\n")
                o.flush()
            if i % 25 == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)}  {el:.0f}s  eta {(len(jobs)-i)/(i/el)/60:.1f}m",
                      flush=True)
    if not res:
        print("nothing judged")
        return
    d = np.array([r["delta_sol"] for r in res])          # change in SOLOIST GP from banking
    print()
    print(f"positions replayed: {len(res)}")
    for lab, sel in (("soloist banks", lambda r: r["mover"] == 0),
                     ("defender banks", lambda r: r["mover"] != 0)):
        s = np.array([r["delta_sol"] for r in res if sel(r)])
        if not len(s):
            continue
        se = s.std(ddof=1) / np.sqrt(len(s))
        # a defender wants soloist GP DOWN; the soloist wants it UP
        gain = -s.mean() if "defender" in lab else s.mean()
        print(f"  {lab:16s} n={len(s):4d}  change in soloist GP {s.mean():+.3f} +- {se:.3f}"
              f"   -> mover gains {gain:+.3f}   (t={s.mean()/se:+.2f})")
        print(f"      unchanged in {100*np.mean(np.abs(s) < 1e-9):.0f}% of positions")


if __name__ == "__main__":
    main()
