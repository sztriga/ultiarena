"""exp48 — is a fed point card a FREE error, or a necessary one?

exp47's corpus shows 8.6 card points per deal handed to the opponent at a moment when a
zero-point card was also legal (defenders 6.0, soloist 2.6). That is an upper bound on the
mistake, not the mistake: sometimes the ten MUST be played — it is the only card that wins
a trick you need, or the low card has to be kept to duck later.

The rigorous question is whether the solver is INDIFFERENT. For each such position, solve
every legal move and ask:

    is some zero-point card among the god-optimal moves?

If yes, the ten was given away for nothing — a pure tie-break error, and fixing it is free
by construction (the god value is unchanged, exactly the guarantee that makes the anti-tell
mixer safe). If no, the AI was right to play it and there is nothing to win.

Reported per role, because the soloist and the defenders have different objectives, and
the fix would be a single tie-break rule serving both.

Run:  WORKERS=8 python3 measure.py [N_POSITIONS]
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
CORPUS = os.path.join(os.path.dirname(_HERE), "47_overnight", "natural.jsonl")
OUT = os.path.join(_HERE, "positions.jsonl")


def _collect(rec):
    """Every ply in this deal where the mover held BOTH a 10-point and a 0-point legal
    card, and played the ten into a trick the OTHER side won."""
    from ulti.bidding.scorers import resolve_bidset, _play_weights
    from ulti.bidding.ladder import LADDER
    from ulti.card import RANK_POINTS, card_from_id
    from ulti.solvers import pis
    from ultisolver._solver_core import set_multi_weights
    out = []
    rung = next(x for x in LADDER if x.index == rec["rung_index"])
    trump = rec["trump"]
    H = [[card_from_id(i) for i in h] for h in (rec["sol"], rec["d1"], rec["d2"])]
    bid = resolve_bidset(rung, H[0], trump)
    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        build_c, solve_c, t, restrict, weights = "betli", "betli", None, None, None
    elif bid.durchmars and rung.colorless and n_trick == 1:
        build_c, solve_c, t, restrict, weights = "durchmars", "durchmars", None, None, None
    else:
        build_c, solve_c, t = "parti", "multi", trump
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        weights = _play_weights(bid, H[0], trump)
    set_multi_weights(**(weights or {}))
    pos = pis.build_position(hands=[list(x) for x in H], soloist=0, leader=0,
                             contract=build_c, trump=t,
                             talon=[card_from_id(i) for i in rec["talon"]],
                             declare_marriages=(t is not None), marriage_restrict=restrict,
                             has_ulti=bool(bid.ulti))
    hist = [(p, card_from_id(c)) for p, c in rec["hist"]]
    for k in range(10):
        trick = hist[k * 3:(k + 1) * 3]
        legal = {}
        ply_index = {}
        for j, (p, card) in enumerate(trick):
            legal[p] = list(pis.legal_actions(pos))
            ply_index[p] = k * 3 + j
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
            alts = legal[p]
            if RANK_POINTS[card.rank] != 10:
                continue
            if not any(RANK_POINTS[c.rank] == 0 for c in alts):
                continue
            enemy_won = (winner == 0) if p != 0 else (winner != 0)
            if not enemy_won:
                continue
            out.append({"seed": rec["seed"], "ply": ply_index[p], "mover": p,
                        "solve_c": solve_c, "build_c": build_c, "trump": t,
                        "restrict": restrict, "weights": weights,
                        "has_ulti": bool(bid.ulti),
                        "hands0": [rec["sol"], rec["d1"], rec["d2"]],
                        "talon": rec["talon"],
                        "hist": [[p2, c2.id] for p2, c2 in hist]})
    return out


def _judge(job):
    """Solve every legal move at this ply. Was a zero-point card also god-optimal?"""
    from ulti.card import RANK_POINTS, card_from_id
    from ulti.solvers import pis
    from ultisolver._solver_core import set_multi_weights
    try:
        set_multi_weights(**(job["weights"] or {}))
        pos = pis.build_position(
            hands=[[card_from_id(i) for i in h] for h in job["hands0"]],
            soloist=0, leader=0, contract=job["build_c"], trump=job["trump"],
            talon=[card_from_id(i) for i in job["talon"]],
            declare_marriages=(job["trump"] is not None),
            marriage_restrict=job["restrict"], has_ulti=job["has_ulti"])
        for p, cid in job["hist"][:job["ply"]]:
            pis.apply_move(pos, card_from_id(cid))
        vals = pis.solve_all(pos, contract=job["solve_c"])
        if not vals:
            return None
        played = card_from_id(job["hist"][job["ply"]][1])
        mover = job["mover"]
        # the soloist maximises the objective; defenders minimise it
        best = max(vals.values()) if mover == 0 else min(vals.values())
        eps = 1e-6
        opt = [c for c, v in vals.items() if abs(v - best) < eps]
        return {"seed": job["seed"], "ply": job["ply"], "mover": mover,
                "played_optimal": any(c.id == played.id for c in opt),
                "zero_optimal": any(RANK_POINTS[c.rank] == 0 for c in opt),
                "n_legal": len(vals), "n_optimal": len(opt)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    workers = int(os.environ.get("WORKERS", "8"))
    rng = random.Random(48)
    deals = []
    with open(CORPUS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kept"):
                deals.append(r)
    rng.shuffle(deals)
    print(f"[exp48] scanning deals for fed-point positions...", flush=True)
    jobs = []
    for rec in deals:
        try:
            jobs += _collect(rec)
        except Exception:
            pass
        if len(jobs) >= n_pos:
            break
    jobs = jobs[:n_pos]
    print(f"[exp48] judging {len(jobs)} positions with {workers} workers", flush=True)
    t0 = time.perf_counter()
    res = []
    with get_context("fork").Pool(workers) as pool, open(OUT, "w") as o:
        for i, r in enumerate(pool.imap_unordered(_judge, jobs, chunksize=2), 1):
            if r and "error" not in r:
                res.append(r)
                o.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}  {time.perf_counter()-t0:.0f}s", flush=True)
    if not res:
        print("no positions judged")
        return
    print()
    print(f"positions judged: {len(res)}")
    for lab, sel in (("ALL", lambda r: True),
                     ("soloist", lambda r: r["mover"] == 0),
                     ("defenders", lambda r: r["mover"] != 0)):
        s = [r for r in res if sel(r)]
        if not s:
            continue
        free = [r for r in s if r["zero_optimal"]]
        print(f"  {lab:10s} n={len(s):4d}   a 0-point card was ALSO god-optimal: "
              f"{100*len(free)/len(s):5.1f}%   "
              f"(the played ten was optimal in {100*np.mean([r['played_optimal'] for r in s]):.0f}%)")
    free_all = [r for r in res if r["zero_optimal"]]
    print()
    print(f"=> {100*len(free_all)/len(res):.1f}% of fed tens were given away for NOTHING "
          f"(a zero-point card scored identically).")


if __name__ == "__main__":
    main()
