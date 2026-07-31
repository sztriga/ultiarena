"""
Play — interactive full-ladder Ulti against the frontier champion.

The user picks a seat (P0 / P1 / P2), then plays a complete game:

  1. AUCTION   — the full 33-rung ladder. ANY seat may open (forehand = P0 acts
                 first; if all three pass the deal is dead → redeal). On your turn
                 you bid a higher contract + trump + discard 2, or pass; the two AI
                 seats bid with the frontier net-bidder (FLOOR=0.7, kontra-aware).
                 We reimplement the exp-23 auction loop so we can pause on the
                 user's turn and resolve all AI turns synchronously in one request.
  2. KONTRA    — for simple contracts (parti/ulti/betli/durchmars) the defenders may
                 kontra and the soloist rekontra. AI decides hand-based (own-hand
                 makeability, cheat-clean); the user gets Kontra/Rekontra buttons.
                 Kontra is a pure scoring multiplier — play is unchanged.
  3. PLAY      — the winning contract is played out. AI = frontier PIMC; user plays
                 their own cards. Mirrors the betli_hu engine (pis bridge).
  4. SCORE     — scoring/oracle.py, with the kontra level applied.

Cheat-clean: every AI decision (bid, kontra, play) sees only its own hand + public
info. The frontier net-bidder ignores the `others` hands; PIMC and the kontra
makeability estimate pool+reshuffle the hidden cards. God is never used here.

Sessions live in-memory; restart the server to clear them.
"""
from __future__ import annotations


import os
import random
import sys
import time
import uuid
from typing import Dict, List, Optional

from ulti.config import apply_deploy_defaults, env_bool, env_float, env_int
from ulti.bidding.ladder import GPTable, overcalls, contract_name
from ulti.bidding.recipe import sol_marriages
from ulti.bidding.auction import net_bid_fn, PASS_PENALTY
from ulti.bidding.scorers import resolve_bidset, _play_weights, _primary_made, _hand_makeability
from ulti.solvers import pis as pis_bridge
from ulti.scoring.units import UNITS_ORDER as _UNITS_ORDER, \
    UNIT_OBJECTIVE as _UNIT_OBJ, kontra_units as _kontra_units
from ultisolver._solver_core import set_multi_weights
from ulti.card import card_from_id, sort_hand
from fastapi import HTTPException

from .serialize import card_to_dict
from fastapi import APIRouter
from pydantic import BaseModel, Field

from .engine import (  # noqa: E402,F401  (re-exports: puzzle uses _provider/_SUIT_HU)
    _GP, _REPO, _SESSION_TTL, _SUIT_HU, _bid_fn, _bid_label, _get, _play_lock,
    _provider, _reap_idle_sessions, _sessions, _sessions_lock, Session,
)
from .auction_flow import (  # noqa: E402
    _advance_auction, _apply_bid, _human_bundle, _legal_bids, _redeal,
    _resolve_auction, _setup_play, _weakest_two,
)
from .kontra_flow import (  # noqa: E402
    _apply_kontra_ai, _available_units, _kontra_dict, _next_kontra_offer,
    _recompute_k_level, _UNIT_HU,
)
from .ai_play import (  # noqa: E402,F401
    _advance_play, _ai_play_pick, _finish, _mix_equivalent, _record_play,
    _terit_revealed,
)
from .snapshots import (  # noqa: E402
    _auction_snapshot, _play_hands_dict, _play_snapshot, _snapshot, _trump_snapshot,
)

router = APIRouter()

# ── Endpoints ───────────────────────────────────────────────────────────────────

class NewRequest(BaseModel):
    seat: int = Field(..., ge=0, le=2)
    seed: Optional[int] = None


@router.post("/play/new")
def play_new(req: NewRequest) -> dict:
    t0 = time.perf_counter()
    rng = random.Random(req.seed)
    seed = req.seed if req.seed is not None else rng.randint(1, 2**31 - 1)
    sess = Session(seat=req.seat, seed=seed)
    with _sessions_lock:
        _reap_idle_sessions()           # cheap O(n) sweep on the rare new-game call
        _sessions[sess.id] = sess
    _advance_auction(sess)
    snap = _snapshot(sess)
    snap["setup_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class BidRequest(BaseModel):
    game_id: str
    rung_index: int
    bid_index: Optional[int] = None       # which interchangeable game on the rung
    trump: Optional[str] = None
    discard_ids: List[int]


@router.post("/play/bid")
def play_bid(req: BidRequest) -> dict:
    t0 = time.perf_counter()
    sess = _get(req.game_id)
    if sess.phase != "bid" or sess.a_done:
        raise HTTPException(status_code=400, detail="not in the bidding phase")
    if sess.a_turn != sess.seat:
        raise HTTPException(status_code=400, detail="not your turn to bid")
    if not sess.a_awaiting_bid:
        raise HTTPException(status_code=400, detail="pick the talon up before bidding")

    current = sess.a_current["rung"] if sess.a_current else None
    legal = {r.index: r for r in overcalls(current)}
    rung = legal.get(req.rung_index)
    if rung is None:
        raise HTTPException(status_code=400, detail="that contract is not a legal overcall")
    # pick the specific game on this rung (None → resolve from hand)
    if req.bid_index is None:
        chosen = None
    elif not (0 <= req.bid_index < len(rung.bids)):
        raise HTTPException(status_code=400, detail="invalid bid_index for this rung")
    else:
        chosen = rung.bids[req.bid_index]
    # Trump color is NOT declared during bidding for a plain colored game — you
    # pick it after the auction (trump_select). Piros = hearts forced; colorless
    # has no trump.
    if rung.colorless:
        trump = None
    elif rung.piros:
        trump = "hearts"
    else:
        trump = None                    # deferred → chosen in trump_select

    bundle = _human_bundle(sess, rung, trump, req.discard_ids)
    # A 100-game needs the marriage in hand. With a known trump (piros) check it
    # exactly; with a deferred trump, require the marriage exists in SOME suit you
    # could pick (else the contract is impossible whatever you choose).
    if chosen is not None and (chosen.forty_hundred or chosen.twenty_hundred):
        hand10 = bundle[4]
        if trump is not None:
            has40, has20 = sol_marriages(hand10, trump)
        else:
            marr = [sol_marriages(hand10, t) for t in ("acorns", "leaves", "bells")]
            has40 = any(m[0] for m in marr)
            has20 = any(m[1] for m in marr)
        if chosen.forty_hundred and not has40:
            raise HTTPException(status_code=400, detail="you don't hold a 40 (trump K+Q) for a 40-100 game")
        if chosen.twenty_hundred and not has20:
            raise HTTPException(status_code=400, detail="you don't hold a 20 (K+Q pair) for a 20-100 game")
    _apply_bid(sess, sess.seat, bundle, bid_override=chosen)
    sess.a_awaiting_bid = False
    sess.a_passes = 0
    sess.a_turn = (sess.seat + 1) % 3
    _advance_auction(sess)
    snap = _snapshot(sess)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class PickupRequest(BaseModel):
    game_id: str


@router.post("/play/pickup")
def play_pickup(req: PickupRequest) -> dict:
    """Auction step: take the talon up (→ 12 cards) so you can discard 2 and bid.
    This is also the holder's RE-RAISE entry point."""
    t0 = time.perf_counter()
    sess = _get(req.game_id)
    if sess.phase != "bid" or sess.a_done:
        raise HTTPException(status_code=400, detail="not in the bidding phase")
    if sess.a_turn != sess.seat:
        raise HTTPException(status_code=400, detail="not your turn")
    if sess.a_awaiting_bid:
        raise HTTPException(status_code=400, detail="you already hold the talon")
    current = sess.a_current["rung"] if sess.a_current else None
    if not overcalls(current):
        raise HTTPException(status_code=400, detail="no higher bid available — accept or pass")
    sess.a_awaiting_bid = True        # → bid step; turn stays with you
    sess.a_picked_up = True           # ring the 2 cards you just took up
    snap = _snapshot(sess)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class PassRequest(BaseModel):
    game_id: str
    discard_ids: List[int] = []      # the 2 cards you bury as the talon (bid-step passz)


@router.post("/play/pass")
def play_pass(req: PassRequest) -> dict:
    t0 = time.perf_counter()
    sess = _get(req.game_id)
    if sess.phase != "bid" or sess.a_done:
        raise HTTPException(status_code=400, detail="not in the bidding phase")
    if sess.a_turn != sess.seat:
        raise HTTPException(status_code=400, detail="not your turn to bid")
    holder = sess.a_current is not None and sess.a_current["pid"] == sess.seat
    if holder:
        # Kezdés / Elfogadom — accept your winning bid, start play. No new discard.
        _resolve_auction(sess)
    elif sess.a_reclaim_offered and sess.a_current is None:
        # Forehand's reclaim declined → the deal is dead → re-deal a fresh hand (same dealer).
        _resolve_auction(sess)
    elif sess.a_awaiting_bid:
        # Bid step (forehand opening / picked-up): passz buries 2 → the talon the
        # next player picks up.
        cards12 = list(sess.a_hands[sess.seat]) + list(sess.a_talon)
        discard = [c for c in cards12 if c.id in req.discard_ids]
        if len(discard) != 2:
            raise HTTPException(status_code=400, detail="select 2 cards to bury as the talon before passing")
        sess.a_hands[sess.seat] = [c for c in cards12 if c.id not in req.discard_ids]
        sess.a_talon = discard
        sess.a_awaiting_bid = False
        sess.a_history.append({"pid": sess.seat, "kind": "pass"})
        sess.a_passes += 1
        sess.a_turn = (sess.seat + 1) % 3
        _advance_auction(sess)
    else:
        # Auction step: decline WITHOUT picking up — the talon is untouched.
        sess.a_history.append({"pid": sess.seat, "kind": "pass"})
        sess.a_passes += 1
        sess.a_turn = (sess.seat + 1) % 3
        _advance_auction(sess)
    snap = _snapshot(sess)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class TrumpRequest(BaseModel):
    game_id: str
    trump: str           # "acorns" | "leaves" | "bells"


@router.post("/play/trump")
def play_trump(req: TrumpRequest) -> dict:
    """Declare the trump color for the plain colored game you just won, then play."""
    t0 = time.perf_counter()
    sess = _get(req.game_id)
    if sess.phase != "trump_select":
        raise HTTPException(status_code=400, detail="not choosing a trump")
    if req.trump not in ("acorns", "leaves", "bells"):
        raise HTTPException(status_code=400, detail="pick makk / zöld / tök")
    sess.a_current["trump"] = req.trump
    _setup_play(sess)
    _advance_play(sess)
    snap = _snapshot(sess)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class KontraRequest(BaseModel):
    game_id: str
    # The units the human chose to kontra/rekontra at this decision point. Empty = pass.
    # (`action` kept for back-compat: "pass" → no units, "double" → all available.)
    units: Optional[List[str]] = None
    action: Optional[str] = None


@router.post("/play/kontra")
def play_kontra(req: KontraRequest) -> dict:
    t0 = time.perf_counter()
    sess = _get(req.game_id)
    if sess.phase != "kontra" or sess.k_next is None:
        raise HTTPException(status_code=400, detail="no kontra decision pending")
    role = sess.k_next["role"]
    pidx = sess.k_next["play_index"]
    avail = sess.k_next.get("units", [])
    # Resolve the chosen units: explicit list (validated ⊆ available), or the
    # legacy action shorthand ("double" = all available, "pass"/None = none).
    if req.units is not None:
        chosen = [U for U in req.units if U in avail]
    elif req.action == "double":
        chosen = list(avail)
    else:
        chosen = []
    if role == "def":
        sess.k_off[pidx] = True
        for U in chosen:
            sess.k_def[U][pidx] = True
        if chosen:
            labels = ", ".join(_UNIT_HU.get(U, U) for U in chosen)
            sess.bubbles.append({"player": pidx, "text": f"Kontra! ({labels})", "ply": pidx})
    else:                                # soloist rekontra
        sess.k_rk_off = True
        for U in chosen:
            sess.k_rekontra[U] = True
        if chosen:
            labels = ", ".join(_UNIT_HU.get(U, U) for U in chosen)
            sess.bubbles.append({"player": 0, "text": f"Rekontra! ({labels})", "ply": 3})
    _recompute_k_level(sess)
    sess.k_next = None
    sess.phase = "play"
    _advance_play(sess)                  # resume play (may offer the next kontra)
    snap = _snapshot(sess)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class MoveRequest(BaseModel):
    game_id: str
    card_id: int = Field(..., ge=0, le=31)


@router.post("/play/move")
def play_move(req: MoveRequest) -> dict:
    t0 = time.perf_counter()
    sess = _get(req.game_id)
    if sess.phase != "play":
        raise HTTPException(status_code=400, detail="not in the play phase")
    if pis_bridge.is_terminal(sess.p_pos):
        raise HTTPException(status_code=400, detail="game already over")
    cur = pis_bridge.current_player(sess.p_pos)
    if cur != sess.human_play_index:
        raise HTTPException(status_code=400, detail=f"not your turn (current player {cur})")

    card = card_from_id(req.card_id)
    legal = pis_bridge.legal_actions(sess.p_pos)
    if card not in legal:
        raise HTTPException(status_code=400, detail=f"card {req.card_id} is not a legal play")

    sess.voids.observe(sess.p_pos, cur, card)
    pis_bridge.apply_move(sess.p_pos, card)
    _record_play(sess, cur, card, by_ai=False)
    _advance_play(sess)
    snap = _snapshot(sess)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class AnalysisRequest(BaseModel):
    game_id: str


@router.post("/play/analysis")
def play_analysis(req: AnalysisRequest) -> dict:
    """God-solver analysis of the hand: rate every played ply (chosen vs god-best
    value + a blunder flag) and hand back everything the client needs to fork
    alternative lines via /pis/explore (the same generalized branch engine the betli
    board uses). Works for any contract — trump, marriages, the weighted multi objective."""
    t0 = time.perf_counter()
    sess = _get(req.game_id)
    if not sess.play_hands0:
        raise HTTPException(status_code=400, detail="no played hand to analyze yet")
    sol, d1, d2 = sess.play_hands0
    solve_c, build_c, weights, trump = (sess.p_solve_contract, sess.p_build_contract,
                                        sess.p_weights, sess.trump)
    per_ply: List[dict] = []
    with _play_lock:
        if weights is not None:
            set_multi_weights(**weights)
        pos = pis_bridge.build_position(
            hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0, contract=build_c,
            trump=trump, talon=list(sess.play_talon),
            declare_marriages=(trump is not None), marriage_restrict=sess.p_restrict)
        for i, step in enumerate(sess.p_history):
            chosen = card_from_id(step["card"]["id"])
            legal_now = pis_bridge.legal_actions(pos)
            if pis_bridge.is_terminal(pos) or chosen not in legal_now:
                break
            pid = pis_bridge.current_player(pos)
            is_solo = (pid == 0)
            if weights is not None:
                set_multi_weights(**weights)
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
            # Blunder = gave up at least half the swing available at this decision.
            is_blunder = loss > 1e-6 and loss >= 0.5 * max(1e-9, vmax - vmin)
            per_ply.append({
                "ply_index": i, "player_id": pid,
                "chosen_card": card_to_dict(chosen),
                "god_best_card": card_to_dict(best_card),
                "god_best_value": float(best_val),
                "god_chosen_value": chosen_val,
                "is_blunder": bool(is_blunder),
                "legal_card_ids": [c.id for c in legal_now],
                "by_ai": bool(step.get("by_ai", False)),
            })
            pis_bridge.apply_move(pos, chosen)
    return {
        "game_id": sess.id,
        "contract": sess.bid_name,
        # everything /pis/explore needs to fork a line off this exact deal:
        "solve_contract": solve_c, "build_contract": build_c,
        "marriage_restrict": sess.p_restrict, "multi_weights": weights,
        "declare_marriages": trump is not None,
        "soloist": 0, "human_play_index": sess.human_play_index, "leader": 0, "trump": trump,
        "initial_hands": [[card_to_dict(c) for c in sort_hand(h, trump is None)]
                          for h in (sol, d1, d2)],
        "talon": [card_to_dict(c) for c in sess.play_talon],
        "per_ply": per_ply,
        "analysis_ms": (time.perf_counter() - t0) * 1000.0,
    }


class StateRequest(BaseModel):
    game_id: str


@router.post("/play/state")
def play_state(req: StateRequest) -> dict:
    return _snapshot(_get(req.game_id))


@router.delete("/play/session/{game_id}")
def play_delete(game_id: str) -> dict:
    with _sessions_lock:
        existed = _sessions.pop(game_id, None) is not None
    return {"deleted": existed}
