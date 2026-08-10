"""AI card play: exploit soloist (exp31), betli-defense net (exp36), PIMC, the anti-tell mixer, scoring."""


import random
import time
from typing import List

from ulti.bidding.scorers import _primary_made
from ulti.solvers import pis as pis_bridge
from ulti.solvers.blocks import equivalent_moves
from ulti.scoring.oracle import score as score_oracle
from ulti.card import card_from_id

from .serialize import card_to_dict
from .engine import Session, _BETLI_DEF, _EXPLOIT, _MIX_EQUIV, _exp36, _recipe
from . import ai_pool
from .kontra_flow import _apply_kontra_ai, _kontra_dict, _next_kontra_offer


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
    """One AI card. The exp36 betli-defense NET runs here (main process, torch);
    every SOLVER decision (exploit soloist / PIMC) ships to the worker pool."""
    sess.p_seed_counter += 1
    is_terit = bool(getattr(sess.bid, "teritett", False))
    ch = None
    # exp36: DEFENDER of a PLAIN (hidden-info) betli → the learned defense net (beats PIMC).
    if (_BETLI_DEF and play_idx != 0 and sess.p_solve_contract == "betli" and not is_terit
            and _exp36 is not None and _exp36.available()):
        ch = _exp36.betli_defense_pick(sess.p_pos, play_idx)
    if ch is None:
        # AI SOLOIST → safe exploitation (exp31) unless terített (open hand) or EXPLOIT=0.
        if _EXPLOIT and play_idx == 0 and not is_terit:
            mode = "exploit"
        # DEFENDER of a terített game → soloist hand revealed → PIN it in the PIMC.
        elif play_idx != 0 and _terit_revealed(sess):
            mode = "pimc_pinned"
        else:
            mode = "pimc"
        job = _recipe(sess)
        job.update(mode=mode, seed=sess.p_seed_counter, bid=sess.bid)
        cid = ai_pool.run("ai_pick", job)
        ch = card_from_id(cid) if cid is not None else None
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
            if pidx in sess.human_pis:
                sess.k_next = {"role": role, "play_index": pidx, "units": avail}
                sess.phase = "kontra"
                return                       # that human decides (popup at their end)
            _apply_kontra_ai(sess, role, pidx, avail)
            continue                          # re-check for further offers
        p = pis_bridge.current_player(sess.p_pos)
        if p in sess.human_pis:
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
    # ONLY games with a real client behind them: in-process drivers (golden harness,
    # pytest, tournaments) create sessions without an HTTP request → owner "local" →
    # skipped. Before this gate, test runs buried the real games 121 rows to 14.
    origin = getattr(sess, "owner_ip", None)
    try:
        from .recording import record_game, should_record
        if not should_record(origin):
            return
        record_game({
            "id": sess.id, "created_at": time.time(), "seed": sess.seed,
            "contract": sess.bid_name, "trump": sess.trump,
            "soloist_seat": w, "human_seat": sess.seat, "kontra_level": sess.k_level,
            "winner": "soloist" if soloist_won else "defenders", "made": made,
            "seat_gp": seat_gp,
            "players": [                          # seat → identity; ONE shape for solo AND live
                {"seat": s, "kind": "human" if s in sess.humans else "ai",
                 "user_id": (sess.players.get(s) or {}).get("user_id") if s in sess.humans else None,
                 # ip/device describe the session CREATOR's browser — meaningful for the
                 # solo game only; a live table has three browsers, identified by user_id.
                 "ip": origin if (s == sess.seat and not sess.live) else None,
                 "device": getattr(sess, "device_id", None) if (s == sess.seat and not sess.live) else None,
                 "agent": None if s in sess.humans else "frontier"}
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


