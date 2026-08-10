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


import random
import time
from typing import List, Optional

from ulti.bidding.ladder import overcalls
from ulti.bidding.recipe import sol_marriages
from ulti.solvers import pis as pis_bridge
from ulti.scoring.units import kontra_units as _kontra_units  # noqa: F401  (re-export: tests/ulti/test_kontra_units.py)
from ulti.card import card_from_id, sort_hand
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .serialize import card_to_dict
from .limits import guard_new_session
from .users import user_from_request
from .engine import (
    Session, _hold, _play_index, _reap_idle_sessions, _recipe, _sessions,
    _sessions_lock, _viewer_seat,
)
from .auction_flow import (
    _advance_auction, _apply_bid, _human_bundle, _resolve_auction, _setup_play,
)
from .kontra_flow import _recompute_k_level, _UNIT_HU
from .ai_play import _advance_play, _record_play
from . import ai_pool
from .snapshots import _snapshot

router = APIRouter()

# ── Endpoints ───────────────────────────────────────────────────────────────────

class NewRequest(BaseModel):
    seat: int = Field(..., ge=0, le=2)
    seed: Optional[int] = None
    # Anonymous browser identity (localStorage uuid) — lists/resumes "your" games and
    # becomes the user mapping when real accounts land. NOT trusted for limits.
    device_id: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F-]{8,64}$")


@router.post("/play/new")
def play_new(req: NewRequest, request: Request = None) -> dict:
    t0 = time.perf_counter()
    rng = random.Random(req.seed)
    seed = req.seed if req.seed is not None else rng.randint(1, 2**31 - 1)
    sess = Session(seat=req.seat, seed=seed)      # just the deal — no AI work yet
    sess.device_id = req.device_id
    user = user_from_request(request)             # logged in → the recording carries it
    sess.players = {req.seat: {"user_id": user["user_id"] if user else None,
                               "username": user["username"] if user else None}}
    with _sessions_lock:
        _reap_idle_sessions()           # cheap O(n) sweep on the rare new-game call
        # guard + insert under ONE lock hold — the cap check and the insert must be
        # atomic or concurrent creates can both pass and overshoot the cap.
        sess.owner_ip = guard_new_session(
            request, _sessions, lambda s: getattr(s, "owner_ip", None))
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
def play_bid(req: BidRequest, request: Request = None) -> dict:
    t0 = time.perf_counter()
    with _hold(req.game_id) as sess:
        viewer = _viewer_seat(sess, request)
        if sess.phase != "bid" or sess.a_done:
            raise HTTPException(status_code=400, detail="not in the bidding phase")
        if sess.a_turn != viewer:
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

        bundle = _human_bundle(sess, rung, trump, req.discard_ids, seat=viewer)
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
        _apply_bid(sess, viewer, bundle, bid_override=chosen)
        sess.a_awaiting_bid = False
        sess.a_passes = 0
        sess.a_turn = (viewer + 1) % 3
        _advance_auction(sess)
        sess.rev += 1
        snap = _snapshot(sess, viewer)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class PickupRequest(BaseModel):
    game_id: str


@router.post("/play/pickup")
def play_pickup(req: PickupRequest, request: Request = None) -> dict:
    """Auction step: take the talon up (→ 12 cards) so you can discard 2 and bid.
    This is also the holder's RE-RAISE entry point."""
    t0 = time.perf_counter()
    with _hold(req.game_id) as sess:
        viewer = _viewer_seat(sess, request)
        if sess.phase != "bid" or sess.a_done:
            raise HTTPException(status_code=400, detail="not in the bidding phase")
        if sess.a_turn != viewer:
            raise HTTPException(status_code=400, detail="not your turn")
        if sess.a_awaiting_bid:
            raise HTTPException(status_code=400, detail="you already hold the talon")
        current = sess.a_current["rung"] if sess.a_current else None
        if not overcalls(current):
            raise HTTPException(status_code=400, detail="no higher bid available — accept or pass")
        sess.a_awaiting_bid = True        # → bid step; turn stays with you
        sess.a_picked_up = True           # ring the 2 cards you just took up
        sess.rev += 1
        snap = _snapshot(sess, viewer)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class PassRequest(BaseModel):
    game_id: str
    discard_ids: List[int] = []      # the 2 cards you bury as the talon (bid-step passz)


@router.post("/play/pass")
def play_pass(req: PassRequest, request: Request = None) -> dict:
    t0 = time.perf_counter()
    with _hold(req.game_id) as sess:
        viewer = _viewer_seat(sess, request)
        if sess.phase != "bid" or sess.a_done:
            raise HTTPException(status_code=400, detail="not in the bidding phase")
        if sess.a_turn != viewer:
            raise HTTPException(status_code=400, detail="not your turn to bid")
        holder = sess.a_current is not None and sess.a_current["pid"] == viewer
        if holder:
            # Kezdés / Elfogadom — accept your winning bid, start play. No new discard.
            _resolve_auction(sess)
        elif sess.a_reclaim_offered and sess.a_current is None:
            # Forehand's reclaim declined → the deal is dead → re-deal a fresh hand (same dealer).
            _resolve_auction(sess)
        elif sess.a_awaiting_bid:
            # Bid step (forehand opening / picked-up): passz buries 2 → the talon the
            # next player picks up.
            cards12 = list(sess.a_hands[viewer]) + list(sess.a_talon)
            discard = [c for c in cards12 if c.id in req.discard_ids]
            if len(discard) != 2:
                raise HTTPException(status_code=400, detail="select 2 cards to bury as the talon before passing")
            sess.a_hands[viewer] = [c for c in cards12 if c.id not in req.discard_ids]
            sess.a_talon = discard
            sess.a_awaiting_bid = False
            sess.a_history.append({"pid": viewer, "kind": "pass"})
            sess.a_passes += 1
            sess.a_turn = (viewer + 1) % 3
            _advance_auction(sess)
        else:
            # Auction step: decline WITHOUT picking up — the talon is untouched.
            sess.a_history.append({"pid": viewer, "kind": "pass"})
            sess.a_passes += 1
            sess.a_turn = (viewer + 1) % 3
            _advance_auction(sess)
        sess.rev += 1
        snap = _snapshot(sess, viewer)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class TrumpRequest(BaseModel):
    game_id: str
    trump: str           # "acorns" | "leaves" | "bells"


@router.post("/play/trump")
def play_trump(req: TrumpRequest, request: Request = None) -> dict:
    """Declare the trump color for the plain colored game you just won, then play."""
    t0 = time.perf_counter()
    with _hold(req.game_id) as sess:
        viewer = _viewer_seat(sess, request)
        if sess.phase != "trump_select":
            raise HTTPException(status_code=400, detail="not choosing a trump")
        if viewer != sess.a_winner:
            raise HTTPException(status_code=400, detail="a győztes választ adut")
        if req.trump not in ("acorns", "leaves", "bells"):
            raise HTTPException(status_code=400, detail="pick makk / zöld / tök")
        sess.a_current["trump"] = req.trump
        _setup_play(sess)
        _advance_play(sess)
        sess.rev += 1
        snap = _snapshot(sess, viewer)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class KontraRequest(BaseModel):
    game_id: str
    # The units the human chose to kontra/rekontra at this decision point. Empty = pass.
    # (`action` kept for back-compat: "pass" → no units, "double" → all available.)
    units: Optional[List[str]] = None
    action: Optional[str] = None


@router.post("/play/kontra")
def play_kontra(req: KontraRequest, request: Request = None) -> dict:
    t0 = time.perf_counter()
    with _hold(req.game_id) as sess:
        viewer = _viewer_seat(sess, request)
        if sess.phase != "kontra" or sess.k_next is None:
            raise HTTPException(status_code=400, detail="no kontra decision pending")
        if sess.k_next["play_index"] != _play_index(sess, viewer):
            raise HTTPException(status_code=400, detail="nem a te kontra döntésed")
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
        sess.rev += 1
        snap = _snapshot(sess, viewer)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class MoveRequest(BaseModel):
    game_id: str
    card_id: int = Field(..., ge=0, le=31)


@router.post("/play/move")
def play_move(req: MoveRequest, request: Request = None) -> dict:
    t0 = time.perf_counter()
    with _hold(req.game_id) as sess:
        viewer = _viewer_seat(sess, request)
        if sess.phase != "play":
            raise HTTPException(status_code=400, detail="not in the play phase")
        if pis_bridge.is_terminal(sess.p_pos):
            raise HTTPException(status_code=400, detail="game already over")
        cur = pis_bridge.current_player(sess.p_pos)
        if cur != _play_index(sess, viewer):
            raise HTTPException(status_code=400, detail=f"not your turn (current player {cur})")

        card = card_from_id(req.card_id)
        legal = pis_bridge.legal_actions(sess.p_pos)
        if card not in legal:
            raise HTTPException(status_code=400, detail=f"card {req.card_id} is not a legal play")

        sess.voids.observe(sess.p_pos, cur, card)
        pis_bridge.apply_move(sess.p_pos, card)
        _record_play(sess, cur, card, by_ai=False)
        _advance_play(sess)
        sess.rev += 1
        snap = _snapshot(sess, viewer)
    snap["step_ms"] = (time.perf_counter() - t0) * 1000.0
    return snap


class AnalysisRequest(BaseModel):
    game_id: str


@router.post("/play/analysis")
def play_analysis(req: AnalysisRequest, request: Request = None) -> dict:
    """God-solver analysis of the hand: rate every played ply (chosen vs god-best
    value + a blunder flag) and hand back everything the client needs to fork
    alternative lines via /pis/explore (the same generalized branch engine the betli
    board uses). Works for any contract — trump, marriages, the weighted multi objective."""
    t0 = time.perf_counter()
    with _hold(req.game_id) as sess:
        _viewer_seat(sess, request)          # live game → members only (403 otherwise)
        if not sess.play_hands0:
            raise HTTPException(status_code=400, detail="no played hand to analyze yet")
        sol, d1, d2 = sess.play_hands0
        solve_c, build_c, weights, trump = (sess.p_solve_contract, sess.p_build_contract,
                                            sess.p_weights, sess.trump)
        # The whole god-solve loop runs in a WORKER (ids in, ids out); here we only
        # decorate the result with card dicts + the by_ai flags from the history.
        raw = ai_pool.run("analysis", _recipe(sess))
        history = list(sess.p_history)
        game_id, bid_name = sess.id, sess.bid_name
        restrict, human_pi = sess.p_restrict, sess.human_play_index
        talon = list(sess.play_talon)
    per_ply: List[dict] = []
    for row in raw:
        step = history[row["ply_index"]]
        per_ply.append({
            "ply_index": row["ply_index"], "player_id": row["player_id"],
            "chosen_card": card_to_dict(card_from_id(row["chosen_card_id"])),
            "god_best_card": card_to_dict(card_from_id(row["god_best_card_id"])),
            "god_best_value": row["god_best_value"],
            "god_chosen_value": row["god_chosen_value"],
            "is_blunder": row["is_blunder"],
            "legal_card_ids": row["legal_card_ids"],
            "by_ai": bool(step.get("by_ai", False)),
        })
    return {
        "game_id": game_id,
        "contract": bid_name,
        # everything /pis/explore needs to fork a line off this exact deal:
        "solve_contract": solve_c, "build_contract": build_c,
        "marriage_restrict": restrict, "multi_weights": weights,
        "declare_marriages": trump is not None,
        "soloist": 0, "human_play_index": human_pi, "leader": 0, "trump": trump,
        "initial_hands": [[card_to_dict(c) for c in sort_hand(h, trump is None)]
                          for h in (sol, d1, d2)],
        "talon": [card_to_dict(c) for c in talon],
        "per_ply": per_ply,
        "analysis_ms": (time.perf_counter() - t0) * 1000.0,
    }


class StateRequest(BaseModel):
    game_id: str


@router.post("/play/state")
def play_state(req: StateRequest, request: Request = None) -> dict:
    with _hold(req.game_id) as sess:
        return _snapshot(sess, _viewer_seat(sess, request))


class MineRequest(BaseModel):
    device_id: str = Field(..., pattern=r"^[0-9a-fA-F-]{8,64}$")


@router.post("/play/mine")
def play_mine(req: MineRequest) -> dict:
    """This browser's live games (newest first), so the splash can offer resume.
    Listing by device_id leaks nothing across players: the id is a uuid the client
    generated for itself, and game ids only ever go to whoever created them."""
    now = time.time()
    with _sessions_lock:
        mine = [s for s in _sessions.values()
                if getattr(s, "device_id", None) == req.device_id]
    mine.sort(key=lambda s: s.last_touch, reverse=True)
    return {"games": [{
        "game_id": s.id, "phase": s.phase, "seat": s.seat,
        "contract": s.bid_name,               # null until the auction resolves
        "trump": s.trump,
        "idle_s": max(0, round(now - s.last_touch)),
    } for s in mine]}


@router.delete("/play/session/{game_id}")
def play_delete(game_id: str) -> dict:
    with _sessions_lock:
        existed = _sessions.pop(game_id, None) is not None
    return {"deleted": existed}
