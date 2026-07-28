"""exp35 — BELIEF-WEIGHTED DETERMINIZATION (Kermit/Skat-style, IJCAI'09) for Ulti.

Uniform PIMC samples worlds uniformly from the info set. BWD instead WEIGHTS each sampled world
by how consistent the OPPONENTS' observed plays are with strong play in that world: under an
ε-greedy-god opponent model, P(their observed card | world) = (1-ε)·[card == god-best in that
world] + ε/|legal|. Worlds where their actual plays were god-optimal get high weight; worlds where
their cards would have been blunders get downweighted. The PIMC value per move becomes a WEIGHTED
average → belief concentrates on the worlds consistent with how they've actually played. Cheat-clean
(only public actions). Windowed to the last W non-viewer plies to bound solver cost (recent plays =
most informative AND cheapest to solve, being near the current position).

TEST: the DEFENDER — who has the most to infer (the declarer's hidden hand + talon). Both defenders
play uniform-PIMC (arm U) or BWD-PIMC (arm B) against a strong cheat-clean PIMC soloist. Metric =
soloist GP (LOWER = better defense); diff = bwd − uniform → NEGATIVE = BWD defense held the soloist
to fewer points. Paired per deal. Env: N (deals), BELIEF_EPS, WINDOW, NW, PIMC_N, WORKERS.
"""
from __future__ import annotations

import collections
import json
import math
import os
import random
import sys
import time
from multiprocessing import get_context

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/31_exploit_play", f"{_REPO}/experiments/29_frontier_table",
           f"{_REPO}/experiments/24_bidding_loop", f"{_REPO}/experiments/23_bidding_integration",
           f"{_REPO}/experiments/14_minigame_bid_eval", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tournament as T                                            # reuse auction/config/helpers
from solvers import pis, determinize as _det                     # noqa: E402
from eval.pimc_matchup import pimc_pick                          # noqa: E402
from scoring.oracle import score as oracle_score                 # noqa: E402

OUT = os.environ.get("OUT") or os.path.join(_HERE, "bwd_tournament.jsonl")
SEED_BASE = 350_000_000
WORKERS = int(os.environ.get("WORKERS", "8"))
PIMC_N = int(os.environ.get("PIMC_N", "16"))
NW = int(os.environ.get("NW", "16"))
BELIEF_EPS = float(os.environ.get("BELIEF_EPS", "0.15"))
WINDOW = int(os.environ.get("WINDOW", "3"))                      # last W non-viewer plies to weight by
EPS_LIST = [float(x) for x in os.environ.get("EPS_LIST", "0.0").split(",")]  # soloist arm not used; placeholder


def pimc_pick_bwd(true_pos, solve_c, build_c, restrict, weights, trump, talon, viewer,
                  n_samples, seed, voids, history, belief_eps, window):
    """Belief-weighted PIMC move for `viewer`. `history` = [(player, Card), …] played so far."""
    rng = random.Random(seed)
    iset = _det.build_info_set(true_pos, viewer, solve_c, voids=voids)
    nonviewer = [i for i, (pl, _c) in enumerate(history) if pl != viewer]
    target = set(nonviewer[-window:]) if window > 0 else set()
    played_by = collections.defaultdict(list)
    for (pl, c) in history:
        played_by[pl].append(c)
    val_sum, wsum = {}, 0.0
    for _ in range(n_samples):
        try:
            rem, tal = _det.sample_world(iset, rng)
        except Exception:
            continue
        talon_w = list(tal) if tal else list(talon)
        hands_start = [list(rem[p]) + list(played_by[p]) for p in range(3)]
        if weights is not None:
            from trickster._solver_core import set_multi_weights
            set_multi_weights(**weights)
        try:
            pos = pis.build_position(hands=hands_start, soloist=0, leader=0, contract=build_c,
                                     trump=trump, talon=talon_w,
                                     declare_marriages=(trump is not None), marriage_restrict=restrict)
        except Exception:
            continue
        logw = 0.0
        ok = True
        for i, (pl, card) in enumerate(history):
            if i in target:
                legal = pis.legal_actions(pos)
                vals = pis.solve_all(pos, contract=solve_c)
                if not vals:
                    ok = False; break
                if pl == 0:                                   # soloist maximises
                    gb = max(vals.values()); is_best = vals.get(card, -1e18) >= gb - 1e-9
                else:                                         # defender minimises
                    gb = min(vals.values()); is_best = vals.get(card, 1e18) <= gb + 1e-9
                p = (1 - belief_eps) * (1.0 if is_best else 0.0) + belief_eps / max(1, len(legal))
                logw += math.log(max(p, 1e-12))
            if card not in pis.legal_actions(pos):
                ok = False; break
            pis.apply_move(pos, card)
        if not ok:
            continue
        w = math.exp(logw)
        vals = pis.solve_all(pos, contract=solve_c)           # value at the current position
        for a in pis.legal_actions(pos):
            val_sum[a] = val_sum.get(a, 0.0) + w * float(vals.get(a, 0.0))
        wsum += w
    if not val_sum or wsum <= 0:
        return None
    avg = {a: val_sum[a] / wsum for a in val_sum}
    return (max(avg, key=avg.get) if viewer == 0 else min(avg, key=avg.get))


def _play_arm(sol, d1, d2, talon, trump, build_c, restrict, weights, solve_c, bid, use_bwd, seed):
    """Soloist(0)=strong PIMC; defenders(1,2)=uniform PIMC or BWD. Return (soloist_gp, ms, n)."""
    if weights is not None:
        from trickster._solver_core import set_multi_weights
        set_multi_weights(**weights)
    pos = T._build_play_pos(sol, d1, d2, talon, trump, build_c, restrict)
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    history = []
    d_t = 0.0; n_d = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        if p == 0:
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=PIMC_N, seed=seed * 131 + mi, voids_dict=vd)
        elif use_bwd:
            t0 = time.perf_counter()
            mv = pimc_pick_bwd(pos, solve_c, build_c, restrict, weights, trump, talon, p, NW,
                               seed * 131 + mi, vd, history, BELIEF_EPS, WINDOW)
            d_t += time.perf_counter() - t0; n_d += 1
        else:
            mv = pimc_pick(pos=pos, contract=solve_c, n_samples=NW, seed=seed * 131 + mi, voids_dict=vd)
        if mv is None:
            mv = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, mv); vd.clear(); vd.update(voids.as_dict())
        history.append((p, mv))
        pis.apply_move(pos, mv); mi += 1
    return float(oracle_score(final_pos=pos, bid=bid).total_sol), d_t, n_d


def _worker(seed):
    r = T._full_auction(seed)
    if r["winner"] is None:
        return {"seed": seed, "pass": True}
    bid, solve_c, build_c, restrict, weights = T._play_cfg(r["rung"], r["trump"], r["sol"])
    sol, d1, d2, talon, trump = r["sol"], r["def1"], r["def2"], r["talon"], r["trump"]
    dd_ok, dd_val = T._dd_makeable(sol, d1, d2, talon, trump, build_c, restrict, weights, solve_c)
    gp_u, _, _ = _play_arm(sol, d1, d2, talon, trump, build_c, restrict, weights, solve_c, bid, False, seed)
    gp_b, tb, nb = _play_arm(sol, d1, d2, talon, trump, build_c, restrict, weights, solve_c, bid, True, seed)
    return {"seed": seed, "pass": False, "contract": r["contract"],
            "dd_makeable": bool(dd_ok), "dd_val": dd_val,
            "gp_uniform_def": gp_u, "gp_bwd_def": gp_b,
            "ms_bwd": (1000.0 * tb / nb) if nb else 0.0}


def build(n):
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(l)["seed"] for l in open(OUT)}
    seeds = [SEED_BASE + i for i in range(n) if SEED_BASE + i not in seen]
    print(f"exp35 BWD tournament: {len(seeds)} deals (BELIEF_EPS={BELIEF_EPS}, WINDOW={WINDOW}, "
          f"NW={NW}) — uniform-def vs BWD-def vs a strong PIMC soloist", flush=True)
    t0 = time.perf_counter(); done = 0; passes = 0
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=T._init) as pool, open(OUT, "a") as o:
        for rec in pool.imap_unordered(_worker, seeds, chunksize=1):
            o.write(json.dumps(rec) + "\n"); o.flush(); done += 1
            if rec.get("pass"):
                passes += 1
            if done % 20 == 0:
                el = time.perf_counter() - t0
                print(f"[bwd] {done}/{len(seeds)} {el:.0f}s eta {(len(seeds)-done)/(done/el)/60:.1f}m "
                      f"passz {100*passes/done:.0f}%", flush=True)
    print("done", flush=True)


def _stats(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = (var / n) ** 0.5
    return m, (m / se if se > 0 else 0.0), n


def analyze():
    recs = [json.loads(l) for l in open(OUT) if not json.loads(l).get("pass")]
    diffs = [r["gp_bwd_def"] - r["gp_uniform_def"] for r in recs]
    m, t, n = _stats(diffs)
    gu = sum(r["gp_uniform_def"] for r in recs) / n
    gb = sum(r["gp_bwd_def"] for r in recs) / n
    out = ["# exp35 — BELIEF-WEIGHTED DETERMINIZATION (defender) vs uniform PIMC\n",
           f"N={n} played · BELIEF_EPS={BELIEF_EPS} · WINDOW={WINDOW} · NW={NW}. Metric = soloist GP "
           "(LOWER = better defense). diff = BWD − uniform → NEGATIVE = BWD defense held the soloist "
           "to fewer points.\n",
           f"- soloist GP/deal:  uniform-def {gu:+.3f}   BWD-def {gb:+.3f}",
           f"- **diff (BWD − uniform) = {m:+.3f} GP/deal  (t={t:+.1f})**  [negative = BWD BETTER]"]
    nz = [d for d in diffs if abs(d) > 1e-9]
    out.append(f"- decisions differ on {len(nz)} deals ({100*len(nz)/n:.0f}%); when they differ "
               f"mean {sum(nz)/len(nz) if nz else 0:+.3f}")
    for cls, sub in [("dd-makeable (soloist favored — infer to break)", [r for r in recs if r["dd_makeable"]]),
                     ("dd-LOST (defenders favored)", [r for r in recs if not r["dd_makeable"]])]:
        if sub:
            mm, tt, _ = _stats([r["gp_bwd_def"] - r["gp_uniform_def"] for r in sub])
            out.append(f"    · {cls:<46} n={len(sub):<4} diff {mm:+.3f} (t={tt:+.1f})")
    byc = collections.defaultdict(list)
    for r in recs:
        byc[r["contract"]].append(r["gp_bwd_def"] - r["gp_uniform_def"])
    out.append("  by contract: " + "  ".join(f"{c}={sum(v)/len(v):+.2f}(n{len(v)})"
               for c, v in sorted(byc.items(), key=lambda kv: -len(kv[1]))[:7]))
    out.append(f"- BWD thinking time: {sum(r['ms_bwd'] for r in recs)/n:.0f} ms/move")
    txt = "\n".join(out) + "\n"
    open(os.path.join(_HERE, "BWD_RESULTS.md"), "w").write(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(int(os.environ.get("N", "500")))
    elif cmd == "analyze":
        analyze()
    else:
        print(f"unknown cmd {cmd}", flush=True)
