"""
Play — interactive full-ladder Ulti against the frontier champion.

The user picks a seat (P0 / P1 / P2), then plays a complete game:

  1. AUCTION   — the full ladder. ANY seat may open (forehand acts first); if all
                 three pass, the deal is SCORED as passz (the forehand pays) and the
                 next hand rotates the dealer. The AI seats bid with the frontier
                 net-bidder (deployed profile in ulti.config, kontra-aware). The
                 auction loop pauses on the user's turn and resolves all AI turns
                 synchronously in one request (auction_flow).
  2. KONTRA    — interleaved with trick 1, PER UNIT: a combined game exposes every
                 committed unit separately (kontra_flow). AI decides hand-based
                 (own-hand signals, cheat-clean); the user gets buttons.
  3. PLAY      — the winning contract is played out. AI = frontier PIMC + the exp36
                 betli-defense net; the user plays their own cards (pis bridge).
  4. SCORE     — scoring/oracle.py, with the per-unit kontra levels applied.

Cheat-clean: every AI decision (bid, kontra, play) sees only its own hand + public
info. The frontier net-bidder ignores the `others` hands; PIMC and the kontra
makeability estimate pool+reshuffle the hidden cards. God is never used here.

Sessions live in-memory with an idle TTL (engine._reap_idle_sessions).
"""
from __future__ import annotations


import random
import time
from typing import List, Optional

from ulti.bidding.ladder import overcalls
from ulti.solvers import pis as pis_bridge
from ulti.card import TRUMP_CHOICES, card_from_id
from ulti.config import env_int
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .users import user_from_request
from .engine import (
    Session, _hold, _install_session, _play_index, _recipe, _sessions,
    _sessions_lock, _viewer_seat,
)
from .auction_flow import (
    _advance_auction, _apply_bid, _human_bundle, _pass_bury, _pass_decline,
    _require_marriage_for_100, _resolve_auction, _setup_play,
)
from .kontra_flow import _apply_kontra_choice
from .ai_play import _advance_play, _record_play, analysis_payload
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
    _install_session(sess, request)
    # Under the session's own lock: /play/mine can already list this game (same
    # device, second tab), so the opening AI turns must not race a /play/state.
    with sess.lock:
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
            raise HTTPException(status_code=400, detail="Nem licitfázis.")
        if sess.a_turn != viewer:
            raise HTTPException(status_code=400, detail="Nem te következel.")
        if not sess.a_awaiting_bid:
            raise HTTPException(status_code=400, detail="Előbb vedd fel a talont.")

        current = sess.a_current["rung"] if sess.a_current else None
        legal = {r.index: r for r in overcalls(current)}
        rung = legal.get(req.rung_index)
        if rung is None:
            raise HTTPException(status_code=400, detail="Ez nem érvényes emelés.")
        # pick the specific game on this rung (None → resolve from hand)
        if req.bid_index is None:
            chosen = None
        elif not (0 <= req.bid_index < len(rung.bids)):
            raise HTTPException(status_code=400, detail="Érvénytelen bid_index ezen a fokon.")
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
        _require_marriage_for_100(chosen, bundle[4], trump)
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
            raise HTTPException(status_code=400, detail="Nem licitfázis.")
        if sess.a_turn != viewer:
            raise HTTPException(status_code=400, detail="Nem te következel.")
        if sess.a_awaiting_bid:
            raise HTTPException(status_code=400, detail="Már nálad a talon.")
        current = sess.a_current["rung"] if sess.a_current else None
        if not overcalls(current):
            raise HTTPException(status_code=400, detail="Nincs magasabb licit — fogadd el vagy passzolj.")
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
            raise HTTPException(status_code=400, detail="Nem licitfázis.")
        if sess.a_turn != viewer:
            raise HTTPException(status_code=400, detail="Nem te következel.")
        holder = sess.a_current is not None and sess.a_current["pid"] == viewer
        if holder or (sess.a_reclaim_offered and sess.a_current is None):
            # Kezdés / Elfogadom (accept your winning bid, start play) — or the
            # forehand declining the reclaim → the deal is dead, scored as passz.
            _resolve_auction(sess)
        elif sess.a_awaiting_bid:
            _pass_bury(sess, viewer, req.discard_ids)    # bid step: bury 2, pass
        else:
            _pass_decline(sess, viewer)                  # auction step: talon untouched
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
            raise HTTPException(status_code=400, detail="Nem adu-választás van.")
        if viewer != sess.a_winner:
            raise HTTPException(status_code=400, detail="A győztes választ adut.")
        if req.trump not in TRUMP_CHOICES:
            raise HTTPException(status_code=400, detail="tök / zöld / makk közül válassz")
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
            raise HTTPException(status_code=400, detail="Nincs függő kontra döntés.")
        if sess.k_next["play_index"] != _play_index(sess, viewer):
            raise HTTPException(status_code=400, detail="Nem a te kontra döntésed.")
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
        _apply_kontra_choice(sess, role, pidx, chosen)
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
            raise HTTPException(status_code=400, detail="Nem játékfázis.")
        if pis_bridge.is_terminal(sess.p_pos):
            raise HTTPException(status_code=400, detail="A játszma már véget ért.")
        cur = pis_bridge.current_player(sess.p_pos)
        if cur != _play_index(sess, viewer):
            raise HTTPException(status_code=400, detail="Nem te következel.")

        card = card_from_id(req.card_id)
        legal = pis_bridge.legal_actions(sess.p_pos)
        if card not in legal:
            raise HTTPException(status_code=400, detail="Ez a lap most nem játszható.")

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
            raise HTTPException(status_code=400, detail="Még nincs lejátszott parti az elemzéshez.")
        # The whole god-solve loop runs in a WORKER (ids in, ids out); here we only
        # decorate the result with card dicts + the by_ai flags from the history.
        # `bid` and the world count are what turn the analysis from solver units into
        # GP (see ai_worker._gp_if_played). ANALYSIS_WORLDS=0 disables the slower
        # "how much of this was knowable" pass.
        job = _recipe(sess)
        job.update(bid=sess.bid, seed=sess.seed,
                   analysis_worlds=env_int("ANALYSIS_WORLDS", 8))
        raw = ai_pool.run("analysis", job)
        history = list(sess.p_history)
        out = analysis_payload(
            raw, game_id=sess.id, contract=sess.bid_name,
            solve_c=sess.p_solve_contract, build_c=sess.p_build_contract,
            restrict=sess.p_restrict, weights=sess.p_weights, trump=sess.trump,
            hands0=sess.play_hands0, talon=sess.play_talon,
            human_pi=sess.human_play_index,
            by_ai=lambda row: history[row["ply_index"]].get("by_ai", False))
    out["analysis_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


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
def play_delete(game_id: str, device: Optional[str] = None) -> dict:
    """Cancel an ongoing solo game (the × on the splash's resume rows). Guarded like
    the other routes: a live table's game can't be deleted here, and a session tied
    to a browser is deletable only by that browser's device id."""
    with _sessions_lock:
        sess = _sessions.get(game_id)
        if sess is None:
            return {"deleted": False}
        if sess.live:
            raise HTTPException(status_code=403, detail="Asztalos játszma — nem törölhető innen.")
        if sess.device_id is not None and sess.device_id != device:
            raise HTTPException(status_code=403, detail="Nem a te játszmád.")
        _sessions.pop(game_id, None)
    return {"deleted": True}
