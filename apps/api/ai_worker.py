"""Solver-side AI compute, shaped to run in a WORKER PROCESS (or inline).

Why this module exists: the Cython solver keeps process-global state — the EV_MULTI
weight vector (set_multi_weights) and the transposition table — and it holds the GIL
for the whole search. One process can therefore only ever run ONE search at a time,
which used to be enforced by a global lock serializing every AI decision across every
live session. Scaling fix: ship each decision to a small pool of worker processes
(apps.api.ai_pool). Each worker owns its own solver globals, so weights can't be
trampled across sessions and N tables solve on N cores.

Contract: every job is a plain dict of primitives (card IDS, never Card objects), and
every result is primitives too — the pool boundary must stay picklable. Positions are
REBUILT here from the deal + move history, exactly the way /play/analysis always did;
build_position + apply_move are deterministic, so a rebuilt position is byte-equivalent
to the session's live one.

This module must stay TORCH-FREE — workers never load the nets (the betli-defense net
and the bidder run in the main process; they are microsecond forward passes).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ulti.card import card_from_id
from ulti.config import env_float, env_int
from ulti.eval.pimc_matchup import god_says_soloist_wins, pimc_pick
from ulti.scoring.oracle import score as score_oracle
from ulti.scoring.units import UNIT_OBJECTIVE as _UNIT_OBJ
from ulti.solvers import determinize as _det
from ulti.solvers import pis as pis_bridge
from ultisolver._solver_core import set_multi_weights

# exp31 exploit knobs — read from env here as well (workers inherit the parent env).
_EXPLOIT_EPS = env_float("EXPLOIT_EPS", 0.15)
_EXPLOIT_NW = env_int("EXPLOIT_NW", 16)
_EXPLOIT_FRAC = env_float("EXPLOIT_FRAC", 0.10)
_PIMC_N = env_int("PLAY_PIMC_N", 16)
_KONTRA_NDET = env_int("PLAY_KONTRA_NDET", 6)

_ZERO_WEIGHTS: Dict[str, float] = {}


def _apply_weights(weights: Optional[dict]) -> None:
    """Install the job's weight vector — ALWAYS, so a previous job's weights can
    never leak into this one (workers are reused across sessions)."""
    set_multi_weights(**(weights or _ZERO_WEIGHTS))


def _rebuild(recipe: dict):
    """Deal + history → the session's exact live position (deterministic rebuild)."""
    pos = pis_bridge.build_position(
        hands=[[card_from_id(c) for c in h] for h in recipe["hands0"]],
        soloist=0, leader=0,
        contract=recipe["build_c"],
        trump=recipe["trump"],
        talon=[card_from_id(c) for c in recipe["talon"]],
        declare_marriages=(recipe["trump"] is not None),
        marriage_restrict=recipe.get("restrict"),
        has_ulti=bool(recipe.get("has_ulti", False)),
    )
    for _pid, cid in recipe.get("history", ()):
        pis_bridge.apply_move(pos, card_from_id(cid))
    return pos


# ── exp31 safe-exploit soloist (moved verbatim from ai_play) ─────────────────────

def _god_move(pos, solve_c):
    mv, _ = pis_bridge.solve_best(pos, contract=solve_c)
    return mv


def _make_eps_god(solve_c: str, eps: float):
    """The AI's model of a fallible defender: god-argmin except an ε fraction of slips.

    VERBATIM from the pre-pool code — the rng.random() draw happens on EVERY call
    (even forced single-move positions). Reordering the draws shifts the shared RNG
    stream and changes every downstream world sample, i.e. the AI's actual picks.
    """
    def _pick(pos, rng):
        if eps > 0 and rng.random() < eps:
            return rng.choice(pis_bridge.legal_actions(pos))
        mv = _god_move(pos, solve_c)
        return mv if mv is not None else rng.choice(pis_bridge.legal_actions(pos))
    return _pick


def _exploit_rollout_gp(pos, soloist, def_model, solve_c, bid, rng) -> float:
    """Play `pos` to terminal (soloist god-best, defenders play the model); soloist GP."""
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        mv = _god_move(pos, solve_c) if p == soloist else def_model(pos, rng)
        if mv is None:
            mv = rng.choice(pis_bridge.legal_actions(pos))
        pis_bridge.apply_move(pos, mv)
    return float(score_oracle(final_pos=pos, bid=bid).total_sol)


def _safe_exploit_pick(pos, solve_c, bid, voids, seed: int):
    """SAFE exploitation for the AI soloist. Among moves within EXPLOIT_FRAC of the
    PIMC-optimal averaged-god value, pick the best EXPECTED GP vs the modeled defender.
    Cheat-clean: samples worlds from the soloist's OWN info set (+ observed voids)."""
    rng = random.Random(seed)
    iset = _det.build_info_set(pos, 0, solve_c, voids=voids)
    worlds, god_sum, cnt = [], {}, {}
    for _ in range(_EXPLOIT_NW):
        try:
            hands, talon = _det.sample_world(iset, rng)
        except Exception:
            continue
        world = (pis_bridge.clone_with_hands_and_talon(pos, hands, talon)
                 if iset.talon_known is None else pis_bridge.clone_with_hands(pos, hands))
        worlds.append(world)
        gvals = pis_bridge.solve_all(world, contract=solve_c)
        for a in pis_bridge.legal_actions(world):
            god_sum[a] = god_sum.get(a, 0.0) + float(gvals.get(a, 0.0))
            cnt[a] = cnt.get(a, 0) + 1
    if not cnt:
        return None
    god_avg = {a: god_sum[a] / cnt[a] for a in cnt}
    gmax, gmin = max(god_avg.values()), min(god_avg.values())
    tol = _EXPLOIT_FRAC * max(gmax - gmin, 1e-9)
    safe = [a for a in god_avg if god_avg[a] >= gmax - tol]
    if len(safe) == 1:                          # unique worst-case-optimal → = PIMC, no rollout
        return safe[0]
    model = _make_eps_god(solve_c, _EXPLOIT_EPS)
    exp_sum = {a: 0.0 for a in safe}
    for world in worlds:
        for a in safe:
            child = world.clone(); pis_bridge.apply_move(child, a)
            exp_sum[a] += _exploit_rollout_gp(child, 0, model, solve_c, bid, rng)
    return max(safe, key=lambda a: exp_sum[a])


# ── ops (the pool boundary) ──────────────────────────────────────────────────────

def op_ai_pick(job: dict) -> Optional[int]:
    """One AI card decision. mode: 'exploit' (soloist) | 'pimc_pinned' (terített
    defender, soloist hand pinned) | 'pimc'. Returns a card id (None = caller falls
    back to a random legal card, as before).

    `pimc_n` may override the deployed search width for THIS decision. It exists so a
    research harness can seat two play configurations at one table — the deployed value
    is a module global read at import, which makes a per-seat comparison impossible
    otherwise. Absent (the serving path), the global applies exactly as before."""
    _apply_weights(job.get("weights"))
    pos = _rebuild(job)
    solve_c, seed, voids = job["solve_c"], job["seed"], job.get("voids")
    n = int(job.get("pimc_n") or _PIMC_N)
    ch = None
    if job["mode"] == "god":
        # CHEATS: perfect information, plays the double-dummy best move. Never served —
        # research only, as the ceiling on what any play improvement could ever be worth.
        ch = _god_move(pos, solve_c)
    elif job["mode"] == "exploit":
        ch = _safe_exploit_pick(pos, solve_c, job["bid"], voids, seed)
    elif job["mode"] == "pimc_pinned":
        ch = pimc_pick(pos=pos, contract=solve_c, n_samples=n, seed=seed,
                       voids_dict=voids,
                       must_hold={0: list(pis_bridge.hands_by_player(pos)[0])})
    if ch is None:
        ch = pimc_pick(pos=pos, contract=solve_c, n_samples=n, seed=seed,
                       voids_dict=voids)
    return ch.id if ch is not None else None


def op_unit_makeability(job: dict) -> float:
    """P(soloist makes `unit` | viewer's own hand) at the PRE-PLAY root — cheat-clean
    own-hand sampling, god-solved for the unit's objective (body from kontra_flow)."""
    unit, viewer, seed = job["unit"], job["viewer"], job["seed"]
    solver, weights, restrict = _UNIT_OBJ[unit]
    build_c = "durchmars" if solver == "durchmars" else ("betli" if solver == "betli" else "parti")
    _apply_weights(weights)
    root = pis_bridge.build_position(
        hands=[[card_from_id(c) for c in h] for h in job["hands0"]],
        soloist=0, leader=0, contract=build_c, trump=job["trump"],
        talon=[card_from_id(c) for c in job["talon"]],
        declare_marriages=(job["trump"] is not None), marriage_restrict=restrict)
    iset = _det.build_info_set(root, viewer, solver, voids=None)
    rng = random.Random(seed)
    w = valid = 0
    for _ in range(_KONTRA_NDET):
        try:
            hands, tal = _det.sample_world(iset, rng)
            spos = (pis_bridge.clone_with_hands_and_talon(root, hands, tal)
                    if iset.talon_known is None else pis_bridge.clone_with_hands(root, hands))
            _apply_weights(weights)
            valid += 1
            if god_says_soloist_wins(spos, contract=solver):
                w += 1
        except Exception:
            continue
    return w / float(valid) if valid else 0.0


def op_unit_makeability_post1(job: dict) -> float:
    """Post-trick-1 unit makeability for the soloist's rekontra (body from kontra_flow):
    determinize the CURRENT position, rebuild each world's pre-play root, replay trick 1,
    god-check. Falls back to the pre-play signal when no world validates."""
    unit, seed = job["unit"], job["seed"]
    solver, weights, restrict = _UNIT_OBJ[unit]
    if solver == "multi":            # 100-games: no clean replay path → root signal
        return op_unit_makeability(job)
    _apply_weights(weights)
    pos = _rebuild(job)
    trump = job["trump"]
    plays = [(pid, card_from_id(cid)) for pid, cid in job["history"][:3]]
    iset = _det.build_info_set(pos, 0, job["solve_c"], voids=job.get("voids"))
    rng = random.Random(seed)
    w = valid = 0
    for _ in range(_KONTRA_NDET):
        try:
            rem, _tal = _det.sample_world(iset, rng)          # remaining hands at current pos
            init = [list(rem[p]) for p in range(3)]
            for pid, card in plays:
                init[pid].append(card)                         # rebuild the pre-trick-1 hands
            root = pis_bridge.build_position(
                hands=init, soloist=0, leader=0, contract=solver, trump=trump,
                talon=[card_from_id(c) for c in job["talon"]],
                declare_marriages=(trump is not None))
            for _pid, card in plays:
                pis_bridge.apply_move(root, card)              # replay trick 1
            valid += 1
            if god_says_soloist_wins(root, contract=solver):
                w += 1
        except Exception:
            continue
    if valid == 0:
        return op_unit_makeability(job)                        # pre-trick-1 fallback
    return w / float(valid)


# Severity bands, in GP LOST BY THE MOVER. PROVISIONAL — set from judgement, not data,
# because we have no corpus of human decisions yet. They live here as one table so that
# recalibrating them later (to, say, the worst 1% of real human decisions) is a one-line
# change rather than a hunt through the frontend.
_SEVERITY = ((8.0, "baklövés"), (3.0, "hiba"), (1.0, "pontatlanság"))


def _severity(gp_loss: float) -> str:
    for threshold, label in _SEVERITY:
        if gp_loss >= threshold:
            return label
    return "ok"


def _gp_if_played(pos, card, solve_c, bid):
    """Soloist GP if `card` is played here and BOTH sides then play double-dummy.

    This is the number the analysis board should show, and it is not what the solver
    optimises. The solver maximises a weighted sum of BINARY indicators (parti +-1,
    ulti 0/1, reaches-100 0/1); card points only reach it through the parti flag, which
    is usually settled early. Measured on 6,936 deals: 81% of mid-game positions have
    EVERY legal move tied in solver units, so a blunder detector built on them is silent
    four times in five and then fires on things that cost nothing. Scoring the terminal
    with the oracle instead gives GP — including the parti, the silent riders and the
    kontra multipliers — which is what the player actually pays.
    """
    child = pos.clone()
    pis_bridge.apply_move(child, card)
    while not pis_bridge.is_terminal(child):
        mv, _ = pis_bridge.solve_best(child, contract=solve_c)
        if mv is None:
            mv = pis_bridge.legal_actions(child)[0]
        pis_bridge.apply_move(child, mv)
    return float(score_oracle(final_pos=child, bid=bid).total_sol)


def _gp_if_played_blind(pos, card, viewer, solve_c, bid, n_worlds, seed):
    """The same, but from the MOVER's information set: sample worlds consistent with what
    they could see, play each out double-dummy, average the GP.

    The difference between this and `_gp_if_played` is the point. A move can be a
    disaster in hindsight and completely reasonable at the time — milan's piros ulti
    (game 2d26851c23a2) died on its opening card, yet from his seat almost every lead
    looked identical. Showing only the hindsight number would call that a blunder, which
    is both false and useless for learning. Showing both separates "you played badly"
    from "you got unlucky".
    """
    rng = random.Random(seed)
    iset = _det.build_info_set(pos, viewer, solve_c, voids=None)
    total = 0.0
    n = 0
    for _ in range(n_worlds):
        try:
            hands, talon = _det.sample_world(iset, rng)
        except Exception:
            continue
        world = (pis_bridge.clone_with_hands_and_talon(pos, hands, talon)
                 if iset.talon_known is None else pis_bridge.clone_with_hands(pos, hands))
        total += _gp_if_played(world, card, solve_c, bid)
        n += 1
    return (total / n) if n else None


def op_analysis(job: dict) -> List[dict]:
    """Rate every played ply. Body from the /play/analysis route; returns primitives
    (card IDS) for the pool boundary.

    Two verdicts per ply:
      * gp_loss          — GP the move cost against double-dummy hindsight
      * gp_loss_knowable — GP it cost given only what the mover could see
    plus the legacy solver-unit values, which the explore board still uses.
    """
    _apply_weights(job.get("weights"))
    solve_c, weights = job["solve_c"], job.get("weights")
    pos = _rebuild({**job, "history": ()})                     # root only; we replay below
    per_ply: List[dict] = []
    for i, (pid_recorded, cid) in enumerate(job["history"]):
        chosen = card_from_id(cid)
        legal_now = pis_bridge.legal_actions(pos)
        if pis_bridge.is_terminal(pos) or chosen not in legal_now:
            break
        pid = pis_bridge.current_player(pos)
        is_solo = (pid == 0)
        _apply_weights(weights)
        vals = pis_bridge.solve_all(pos, contract=solve_c)
        vmax, vmin = max(vals.values()), min(vals.values())
        if is_solo:
            best_val = vmax
            best_card = next(c for c, v in vals.items() if v >= best_val - 1e-6)
        else:                                 # defender minimises the soloist value
            best_val = vmin
            best_card = next(c for c, v in vals.items() if v <= best_val + 1e-6)
        chosen_val = float(vals.get(chosen, 0.0))
        loss = (best_val - chosen_val) if is_solo else (chosen_val - best_val)
        # Legacy flag: gave up at least half the swing available in SOLVER units. Kept so
        # the explore board keeps working, but it is not what the player is shown — see
        # _gp_if_played for why solver units mislead here.
        is_blunder = loss > 1e-6 and loss >= 0.5 * max(1e-9, vmax - vmin)

        row = {
            "ply_index": i, "player_id": pid,
            "chosen_card_id": chosen.id,
            "god_best_card_id": best_card.id,
            "god_best_value": float(best_val),
            "god_chosen_value": chosen_val,
            "is_blunder": bool(is_blunder),
            "legal_card_ids": [c.id for c in legal_now],
        }

        bid = job.get("bid")
        if bid is not None:
            # GP, from the mover's point of view: the soloist wants total_sol UP, a
            # defender wants it DOWN, so the sign flips for seats 1 and 2.
            sign = 1.0 if is_solo else -1.0
            gp = {c: sign * _gp_if_played(pos, c, solve_c, bid) for c in legal_now}
            gp_best = max(gp.values())
            gp_chosen = gp[chosen]
            gp_loss = max(0.0, gp_best - gp_chosen)
            gp_best_card = max(gp, key=lambda c: gp[c])
            row.update({
                # Soloist-perspective value AFTER this move. The client flips it once for
                # whoever is watching, so the board reads as one continuous evaluation
                # from your own seat rather than a number whose sign hops per row.
                # gp_chosen is already sign-flipped for the mover, so multiplying by the
                # same sign recovers the soloist's view (sign*sign == 1 either way).
                "gp_sol_after": round(sign * gp_chosen, 2),
                "gp_chosen": round(sign * gp_chosen, 2),
                "gp_best": round(sign * gp_best, 2),
                "gp_loss": round(gp_loss, 2),
                "gp_best_card_id": gp_best_card.id,
                "severity": _severity(gp_loss),
                "gp_swing": round(gp_best - min(gp.values()), 2),
            })
            nw = int(job.get("analysis_worlds") or 0)
            if nw > 0 and gp_loss > 0:
                # Only worth sampling where something was actually lost.
                blind = {c: _gp_if_played_blind(pos, c, pid, solve_c, bid, nw,
                                                job.get("seed", 0) + i * 7919 + c.id)
                         for c in legal_now}
                if all(v is not None for v in blind.values()):
                    b = {c: sign * v for c, v in blind.items()}
                    row["gp_loss_knowable"] = round(max(0.0, max(b.values()) - b[chosen]), 2)
        per_ply.append(row)
        pis_bridge.apply_move(pos, chosen)
    return per_ply


_OPS = {
    "ai_pick": op_ai_pick,
    "unit_makeability": op_unit_makeability,
    "unit_makeability_post1": op_unit_makeability_post1,
    "analysis": op_analysis,
}


def dispatch(op: str, job: dict) -> Any:
    """Entry point the pool submits (top-level → picklable under spawn)."""
    return _OPS[op](job)
