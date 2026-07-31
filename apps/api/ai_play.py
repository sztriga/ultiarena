"""AI card play: exploit soloist (exp31), betli-defense net (exp36), PIMC, the anti-tell mixer, scoring."""
from __future__ import annotations


import os
import random
import sys
import time
import uuid
from typing import Dict, List, Optional

from ulti.config import apply_deploy_defaults, env_bool, env_float, env_int
from ulti.bidding.ladder import GPTable, overcalls, contract_name
from ulti.bidding.auction import net_bid_fn, PASS_PENALTY
from ulti.bidding.scorers import resolve_bidset, _play_weights, _primary_made, _hand_makeability
from ulti.solvers import pis as pis_bridge
from ulti.solvers import determinize as _det
from ulti.solvers.blocks import equivalent_moves
from ulti.eval.pimc_matchup import pimc_pick
from ulti.scoring.oracle import score as score_oracle
from ulti.scoring.units import UNITS_ORDER as _UNITS_ORDER, \
    UNIT_OBJECTIVE as _UNIT_OBJ, kontra_units as _kontra_units
from ultisolver._solver_core import set_multi_weights
from ulti.card import card_from_id, sort_hand

from .serialize import card_to_dict
from .engine import Session, _BETLI_DEF, _EXPLOIT, _EXPLOIT_EPS, _EXPLOIT_FRAC, _EXPLOIT_NW, _MIX_EQUIV, _PIMC_N, _exp36, _play_lock  # noqa: E402
from .kontra_flow import _apply_kontra_ai, _kontra_dict, _next_kontra_offer  # noqa: E402


# ── Play ─────────────────────────────────────────────────────────────────────────

def _record_play(sess: Session, play_idx: int, card, by_ai: bool) -> None:
    plies = len(sess.p_history)
    sess.p_history.append({
        "player_id": play_idx,
        "card": card_to_dict(card),
        "trick_index": plies // 3,
        "trick_position": plies % 3,
        "by_ai": by_ai,
    })


def _god_move(pos, solve_c):
    mv, _ = pis_bridge.solve_best(pos, contract=solve_c)
    return mv


def _make_eps_god(solve_c: str, eps: float):
    """The AI's model of a fallible defender: god-argmin except an ε fraction of slips."""
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


def _safe_exploit_pick(sess: Session, seed: int):
    """SAFE exploitation for the AI soloist (soloist = play-index 0). Among moves within
    EXPLOIT_FRAC·(value-spread) of the PIMC-optimal averaged-god value, pick the best
    EXPECTED GP vs the modeled defender. Unique optimum → return it (= PIMC cost, no
    rollout). Cheat-clean: samples worlds from the soloist's OWN info set (+ observed voids)."""
    solve_c = sess.p_solve_contract
    rng = random.Random(seed)
    true_pos = sess.p_pos
    iset = _det.build_info_set(true_pos, 0, solve_c, voids=sess.voids.as_dict())
    worlds, god_sum, cnt = [], {}, {}
    for _ in range(_EXPLOIT_NW):
        try:
            hands, talon = _det.sample_world(iset, rng)
        except Exception:
            continue
        world = (pis_bridge.clone_with_hands_and_talon(true_pos, hands, talon)
                 if iset.talon_known is None else pis_bridge.clone_with_hands(true_pos, hands))
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
            exp_sum[a] += _exploit_rollout_gp(child, 0, model, solve_c, sess.bid, rng)
    return max(safe, key=lambda a: exp_sum[a])


def _terit_revealed(sess: Session) -> bool:
    """Terített (open-hand) games: once trick 1 is complete — which is also after the
    marriage declarations and the whole kontra round, both interleaved with trick 1 —
    the soloist lays their hand FACE-UP and the defenders see it for the rest of the
    game. Covers terített betli, terített durchmars, AND terített combined games
    (ulti / 40-100 / …), since `bid.teritett` is set on all of them (milan 2026-07-24).

    Reveal ONLY once the whole first round is settled — marriage declarations, trick 1,
    AND the full kontra round (defender kontra + soloist rekontra). `k_rk_off` flips true
    when the rekontra decision is made (or immediately if nothing was kontra'd); a game
    with no kontra-able unit reveals as soon as trick 1 completes. (milan 2026-07-29)"""
    if not bool(getattr(sess.bid, "teritett", False)) or len(sess.p_history) < 3:
        return False
    return sess.k_rk_off or not sess.k_units


def _mix_equivalent(sess: Session, play_idx: int, card):
    """Swap ``card`` for a random card that plays identically — an anti-tell.

    Whatever chose the card (PIMC, the exp36 betli-defense net, the exploit soloist)
    tends to return the SAME member of an equivalent run every time — normally the
    highest. That is a tell: leading the top of a run says "I hold nothing above this".
    A human picks arbitrarily inside a run, so the engine should too.

    Only cards in one equivalence block are considered, so this is free by construction:
    the block members lead to literally the same game (proved by
    tests/ulti/test_block_equivalence.py). Blocks never span suits, so void inference is
    unaffected too.

    NOT applied to the betli-family soloist: there ``_legal`` has already applied a
    DOMINANCE cull (highest card per suit), and the cards it dropped are worse, not
    equal. Randomising over those would throw away real value.
    """
    if not _MIX_EQUIV or card is None:
        return card
    colorless = sess.trump is None
    if colorless and play_idx == 0:
        return card                       # betli / colorless duri soloist → dominance
    try:
        block = equivalent_moves(sess.p_pos, play_idx, card,
                                 colorless=colorless, trump=sess.trump)
    except Exception:                      # never let an anti-tell break a game
        return card
    if len(block) < 2:
        return card
    return random.Random(sess.p_seed_counter * 7919 + card.id).choice(block)


def _ai_play_pick(sess: Session, play_idx: int):
    sess.p_seed_counter += 1
    is_terit = bool(getattr(sess.bid, "teritett", False))
    with _play_lock:
        if sess.p_weights is not None:
            set_multi_weights(**sess.p_weights)
        ch = None
        # AI SOLOIST → safe exploitation (unless terített = open hand, or EXPLOIT=0 → PIMC below).
        if _EXPLOIT and play_idx == 0 and not is_terit:
            ch = _safe_exploit_pick(sess, sess.p_seed_counter)
        # DEFENDER of a terített game → the soloist's hand is revealed → PIN it (must_hold) so the
        # PIMC samples only the PARTNER's hand, not the (now known) soloist's (near-god defense).
        elif play_idx != 0 and _terit_revealed(sess):
            ch = pimc_pick(pos=sess.p_pos, contract=sess.p_solve_contract, n_samples=_PIMC_N,
                           seed=sess.p_seed_counter, voids_dict=sess.voids.as_dict(),
                           must_hold={0: list(pis_bridge.hands_by_player(sess.p_pos)[0])})
        # exp36: DEFENDER of a PLAIN (hidden-info) betli → the learned defense net (beats PIMC).
        elif (_BETLI_DEF and play_idx != 0 and sess.p_solve_contract == "betli" and not is_terit
              and _exp36 is not None and _exp36.available()):
            ch = _exp36.betli_defense_pick(sess.p_pos, play_idx)
        # Everything else (and any None fallback above) → plain PIMC.
        if ch is None:
            ch = pimc_pick(pos=sess.p_pos, contract=sess.p_solve_contract, n_samples=_PIMC_N,
                           seed=sess.p_seed_counter, voids_dict=sess.voids.as_dict())
    if ch is None:
        ch = random.Random(sess.p_seed_counter).choice(pis_bridge.legal_actions(sess.p_pos))
    return _mix_equivalent(sess, play_idx, ch)


def _advance_play(sess: Session) -> None:
    while not pis_bridge.is_terminal(sess.p_pos):
        # Kontra decisions are interleaved with trick 1: a defender decides right
        # after playing their first card, the soloist rekontras after trick 1.
        offer = _next_kontra_offer(sess)
        if offer is not None:
            role, pidx, avail = offer
            if pidx == sess.human_play_index:
                sess.k_next = {"role": role, "play_index": pidx, "units": avail}
                sess.phase = "kontra"
                return                       # human decides (sidebar box)
            _apply_kontra_ai(sess, role, pidx, avail)
            continue                          # re-check for further offers
        p = pis_bridge.current_player(sess.p_pos)
        if p == sess.human_play_index:
            return
        ch = _ai_play_pick(sess, p)
        sess.voids.observe(sess.p_pos, p, ch)
        pis_bridge.apply_move(sess.p_pos, ch)
        _record_play(sess, p, ch, by_ai=True)
    _finish(sess)


_SILENT_LABEL = {
    "silent_ulti":           "csendes ulti",
    "silent_40_100":         "csendes 40-100",
    "silent_20_100":         "csendes 20-100",
    "silent_durchmars":      "csendes durchmars",
    "def_silent_40_100":     "csendes 40-100 (védő)",
    "def_silent_20_100":     "csendes 20-100 (védő)",
    "def_silent_durchmars":  "csendes durchmars (védő)",
}


def _silent_breakdown(pvec) -> List[dict]:
    """Silent (csendes) contracts that scored this deal — soloist-perspective GP.
    They're already in the total; this just surfaces them so the scoreboard shows
    WHY the GP is what it is (e.g. a silent ulti or a defender's silent 100)."""
    out = []
    for k, v in pvec.components.items():
        if "silent" in k and v != 0:
            out.append({"key": k, "label": _SILENT_LABEL.get(k, k), "gp": int(v)})
    return out


def _finish(sess: Session) -> None:
    pvec = score_oracle(final_pos=sess.p_pos, bid=sess.bid, kontras=_kontra_dict(sess))
    sol_per_def = pvec.total_per_def
    made = _primary_made(sess.bid, pvec)
    soloist_won = sol_per_def > 0
    hpi = sess.human_play_index
    if hpi == 0:
        user_won = soloist_won
        human_gp = pvec.total_sol
    else:
        user_won = not soloist_won
        human_gp = -pvec.gp_vs(hpi - 1)
    # per real-seat GP for the round (zero-sum) — feeds the match scorecard.
    w = sess.a_winner
    seat_gp = [0.0, 0.0, 0.0]
    seat_gp[w] = float(pvec.total_sol)
    seat_gp[(w + 1) % 3] = -float(pvec.gp_vs(0))
    seat_gp[(w + 2) % 3] = -float(pvec.gp_vs(1))
    sess.phase = "done"
    sess.result = {
        "winner": "soloist" if soloist_won else "defenders",
        "made": bool(made),
        "sol_gp_per_def": float(sol_per_def),
        "human_gp": float(human_gp),
        "user_won": bool(user_won),
        "contract": sess.bid_name,
        "kontra_level": sess.k_level,
        "seat_gp": seat_gp,
        "soloist_seat": w,
        "silents": _silent_breakdown(pvec),
    }

    # ── record the finished game for later AI analysis (best-effort; never breaks a game) ──
    try:
        from .recording import record_game
        record_game({
            "id": sess.id, "created_at": time.time(), "seed": sess.seed,
            "contract": sess.bid_name, "trump": sess.trump,
            "soloist_seat": w, "human_seat": sess.seat, "kontra_level": sess.k_level,
            "winner": "soloist" if soloist_won else "defenders", "made": made,
            "seat_gp": seat_gp,
            "players": [                          # seat → identity (user-aware for later auth / human-vs-human)
                {"seat": s, "kind": "human" if s == sess.seat else "ai",
                 "user_id": None, "agent": None if s == sess.seat else "frontier"}
                for s in range(3)
            ],
            "transcript": {                        # play-index space (0 = soloist); auction is real-seat
                "deal": {"hands": [[c.id for c in h] for h in sess.play_hands0],
                         "talon": [c.id for c in sess.play_talon]},
                "auction": sess.a_history,
                "plays": [[h["player_id"], h["card"]["id"], h["trick_index"]] for h in sess.p_history],
                "kontra": _kontra_dict(sess),
                "marriages": [[p, su, pts] for (p, su, pts) in getattr(sess.p_pos, "marriages", [])],
            },
        })
    except Exception:
        pass


