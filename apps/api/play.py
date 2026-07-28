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
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Repo root — bidder/betli/solver code are proper packages now (no sys.path hacks).
_REPO = Path(__file__).resolve().parents[2]

# The frontier bidder reads these at import time — the champion config.
# Re-tuned 2026-07-22 on the FIXED bidder (post the auction.py generator bug fix that unlocked
# non-piros contracts); true head-to-head +0.40 GP/game vs the old 0.70/0.80/1.0 (exp30/exp32).
os.environ.setdefault("FLOOR", "0.80")            # was 0.7 — suppresses overconfident escalations
os.environ.setdefault("DEBIAS_PCTL", "0.85")      # was 0.80
os.environ.setdefault("DURI_TERIT_MULT", "0.3")   # fixes the terített-duri over-bid leak (exp29/30)
os.environ.setdefault("KONTRA", "1")   # kontra-aware bidder (passes weak hands)

from ulti.bidding.ladder import GPTable, overcalls, contract_name  # noqa: E402
from ulti.bidding.recipe import sol_marriages  # noqa: E402
from ulti.bidding.bidder import rung_ev, _is_simple  # noqa: E402
from ulti.bidding.auction import net_bid_fn, PASS_PENALTY  # noqa: E402
from ulti.bidding.provider import NetProvider  # noqa: E402
from ulti.bidding.scorers import resolve_bidset, _play_weights, _primary_made, _hand_makeability  # noqa: E402
from ulti.bidding.kontra import _sol_ev  # noqa: E402
from ulti.bidding.deal import deal_12_10_10  # noqa: E402
from ulti.solvers import pis as pis_bridge  # noqa: E402
from ulti.solvers import determinize as _det  # noqa: E402
from ulti.eval.pimc_matchup import pimc_pick  # noqa: E402
try:
    from ulti.betli import defense as _exp36  # noqa: E402  (exp36 betli-defense net; models/betli/betli_defense.pt)
except Exception:  # pragma: no cover
    _exp36 = None
from ulti.scoring.oracle import score as score_oracle  # noqa: E402
from ultisolver._solver_core import set_multi_weights  # noqa: E402
from ulti.card import card_from_id  # noqa: E402

from .serialize import card_to_dict

router = APIRouter()

_PIMC_N = int(os.environ.get("PLAY_PIMC_N", "16"))
_KONTRA_NDET = int(os.environ.get("PLAY_KONTRA_NDET", "6"))
_GP = GPTable()

# ── Exploitative soloist play (exp31 — the champion's biggest play-side lever) ──
# When the AI is the SOLOIST it plays SAFE-EXPLOIT instead of plain PIMC: among the
# moves that don't sacrifice the PIMC-optimal (worst-case) value, it picks the one with
# the best EXPECTED GP against a model of the fallible human defender. Safe = never worse
# than PIMC vs perfect defense, better whenever the human slips; cheat-clean (samples worlds
# from its OWN info set only). Validated exp31 (full-game GP, terített-gated): +0.6..+0.75
# GP/deal vs a fallible defender, 0 regression vs a perfect one, ~1.3-1.6 s/move. Gated OFF
# on terített (open hand → no hidden info to exploit + amplified stakes). EXPLOIT=0 disables.
_EXPLOIT      = os.environ.get("EXPLOIT", "1") == "1"          # deployed ON = the frontier
# Modeled human mistake rate. Retuned 0.30 → 0.15 (overnight exp33 robustness matrix, 2026-07-23):
# a LOW modeled ε dominates — pooled strength gain +0.75 vs +0.34 GP/deal (t+3.3), AND it's the only
# value that stays safe against a near-perfect defender (over-modeling — assuming the opponent is
# sloppier than they are — set aggressive traps that BACKFIRE vs strong defense). Conservative
# modeling captures the easy traps without gambling on unlikely mistakes.
_EXPLOIT_EPS  = float(os.environ.get("EXPLOIT_EPS", "0.15"))
_EXPLOIT_NW   = int(os.environ.get("EXPLOIT_NW", "16"))        # sampled worlds per decision
_EXPLOIT_FRAC = float(os.environ.get("EXPLOIT_FRAC", "0.10"))  # safe-set slack (rel. to value spread)

# ── exp37: imperfect/bluff PLAIN betli bidding — PROMOTED 2026-07-24 (BETLI_REAL_BID=0 reverts).
# The bidder scores PLAIN betli by a REALISTIC-defense make-prob (not the god double-dummy), so it
# bids the cheap 5p betli on hands that make vs imperfect defenders (exp38 ladder: 5.9% of bids,
# +4.84 GP/bid; ~+0.16 GP/game). rebetli/terített keep the god prob (they're voluntary doublings).
_BETLI_REAL_BID = os.environ.get("BETLI_REAL_BID", "1") == "1"
# ── exp36: learned betli-DEFENSE net for a PLAIN (hidden-info) betli — PROMOTED 2026-07-24 (BETLI_DEF=0
# reverts to PIMC). −21pp soloist steal vs PIMC (benchmark). Scoped to PLAIN betli — terített betli is
# already defended near-god by the reveal (must_hold). Falls back to PIMC if the net is unavailable.
_BETLI_DEF = os.environ.get("BETLI_DEF", "1") == "1"
# ── exp39: extend the realistic-defense prob to REBETLI (the HIDDEN 10p doubling) — PROMOTED 2026-07-25
# (REBETLI_REAL_BID=0 reverts). Lets the AI escalate betli→rebetli when its realistic make is near-certain
# (gated behind REBETLI_FLOOR=0.90, higher than plain betli's 0.80). exp39 self-play: rebetli 8.5% of bids,
# +8.14 GP/bid, does NOT cannibalise ulti; head-to-head vs the rebetli-off frontier +0.16 GP/game (within
# noise → ~neutral, mildly +). This is the human-like "bid betli, escalate to rebetli when confident" move.
_REBETLI_REAL_BID = os.environ.get("REBETLI_REAL_BID", "1") == "1"


def _bid_label(bid) -> str:
    """Display name for a specific contract. Hungarian 'színtelen' for the
    no-trump games (the internal ladder name says 'colorless', which is
    test-locked — rename only at the UI boundary)."""
    return contract_name(bid).replace("colorless", "színtelen")

# The play solver's multi-game weights (`set_multi_weights`) are PROCESS-GLOBAL,
# so all AI play across every live session must be serialized.
_play_lock = RLock()


# ── AI singletons (lazy) ────────────────────────────────────────────────────────

_provider_lock = RLock()
_provider_obj = None
_bid_fn_obj = None


def _bid_fn():
    global _provider_obj, _bid_fn_obj
    with _provider_lock:
        if _bid_fn_obj is None:
            # Load the exp37 realistic-betli head; net_bid_fn(betli_real=...) gates whether it's used.
            _provider_obj = NetProvider(
                calibrate=True, betli_real_dir=str(_REPO / "models/ulti/betli"))
            _bid_fn_obj = net_bid_fn(_provider_obj, betli_real=_BETLI_REAL_BID,
                                     rebetli_real=_REBETLI_REAL_BID)
        return _bid_fn_obj


def _provider():
    _bid_fn()
    return _provider_obj


# ── Session ─────────────────────────────────────────────────────────────────────

class Session:
    """One in-flight game: auction state machine → kontra → pis play position."""

    def __init__(self, *, seat: int, seed: int) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.seat = seat            # the user's real auction seat (0/1/2)
        self.seed = seed
        self.redeals = 0            # dead-deal (all-pass) re-deals in this session
        self.phase = "bid"          # bid -> (kontra -> play -> done); all-pass re-deals in place

        # ── auction state (real-seat space) ──
        sol12, d1, d2 = deal_12_10_10(seed)
        self.a_hands: List[list] = [list(sol12[:10]), list(d1), list(d2)]
        self.a_talon: list = list(sol12[10:])
        self.a_current: Optional[dict] = None    # {pid, rung, trump, ev, bid}
        self.a_passes = 0
        self.a_turn = 0
        self.a_reclaim_offered = False           # forehand's post-all-pass last look
        self.a_done = False
        self.a_winner: Optional[int] = None
        self.a_history: List[dict] = []
        # Two-step: the forehand (seat 0) is dealt the talon and starts in the BID
        # step (holds 12 → dropdown: passz buries 2, or a contract). Everyone else
        # starts in the AUCTION step (holds 10 → Felveszem to pick the talon up and
        # then bid, or Passz to decline without touching it). `a_awaiting_bid` is the
        # bid step; it flips True on pickup / the forehand's reclaim.
        self.a_awaiting_bid = (seat == 0)
        # True once you REACHED the bid step via a pickup (Felveszem / reclaim) rather
        # than being dealt the 12 — used to ring the 2 talon cards you just took up.
        self.a_picked_up = False

        # ── speech-bubble events: [{player, text}, ...] drained per snapshot ──
        self.bubbles: List[dict] = []

        # ── kontra state (PER-UNIT, interleaved with trick 1) ──
        # Each defender may kontra any live unit right after playing their first card;
        # the soloist may rekontra after trick 1. A simple game has ONE live unit (its
        # primary → unchanged deployed behavior); a combined game (100 / ulti-duri /
        # ulti-40-100 / …) exposes every committed unit separately (milan: "combined →
        # minden külön kontrázható"). Colored units are shared (együtt sírunk); colorless
        # (betli / no-trump duri) keep separate per-defender counters.
        self.k_units: List[str] = []             # live kontra-able units ([] → no kontra round)
        self.k_colorless = False                 # betli / no-trump duri → per-defender separate
        self.k_def: Dict[str, Dict[int, bool]] = {}   # k_def[unit][defender 1/2] = kontra'd
        self.k_off = {1: False, 2: False}        # each defender's kontra decision made (once)
        self.k_rekontra: Dict[str, bool] = {}    # k_rekontra[unit] = soloist rekontra'd that unit
        self.k_rk_off = False                    # soloist rekontra decision made (once)
        self.k_level = 0                          # display level (max over units × defenders)
        self.k_next: Optional[dict] = None       # pending human decision {role, play_index, units}

        # ── play state (play-index space, soloist = 0) ──
        self.p_pos = None
        self.bid = None                          # scoring.oracle.BidSet
        self.bid_name: Optional[str] = None      # display label of the chosen game
        self.rung = None
        self.trump: Optional[str] = None
        self.p_build_contract = "parti"
        self.p_solve_contract = "multi"
        self.p_restrict: Optional[str] = None
        self.p_weights: Optional[dict] = None
        self.human_play_index = 0
        self.play_hands0: List[list] = []        # [sol, d1, d2] in play space
        self.play_talon: list = []
        self.voids = None
        self.p_history: List[dict] = []
        self.p_seed_counter = seed * 31337
        self.result: Optional[dict] = None


_sessions: Dict[str, Session] = {}
_sessions_lock = RLock()


# ── Auction (any seat may open) ─────────────────────────────────────────────────

def _bid_ai(sess: Session, pid: int, current_rung) -> Optional[tuple]:
    cards12 = list(sess.a_hands[pid]) + list(sess.a_talon)
    return _bid_fn()(cards12, current_rung, None)


def _weakest_two(cards12: list, trump: Optional[str]):
    """The 2 cards least worth keeping — and least useful to whoever picks up the
    talon next: non-trump, low card-points, low rank first. Returns (discard2,
    keep10). Used when a player PASSES (buries the 2 junk cards as the new talon).

    We deliberately KEEP 7s (shed 8s/9s ahead of them): a buried 7 — the trump 7
    is the ulti card — could make the next player's ulti. (milan 2026-07)"""
    def junk_key(c):
        is_trump = 1 if (trump is not None and c.suit == trump) else 0
        is_seven = 1 if c.rank == "7" else 0
        return (is_trump, c.points, is_seven, c.rank_index)
    ordered = sorted(cards12, key=junk_key)
    return ordered[:2], ordered[2:]


def _apply_bid(sess: Session, pid: int, bundle: tuple, bid_override=None) -> None:
    ev, rung, trump, discard, hand10 = bundle
    sess.a_hands[pid] = list(hand10)
    sess.a_talon = list(discard)
    # Pin the SPECIFIC game on this rung. The human names it explicitly (bid_index);
    # the AI resolves it from its own hand. This is what breaks the two
    # interchangeable "≡" contracts (40-100-duri vs ulti-duri) apart.
    bid = bid_override if bid_override is not None else resolve_bidset(rung, hand10, trump)
    sess.a_current = {"pid": pid, "rung": rung, "trump": trump, "ev": ev, "bid": bid}
    sess.a_history.append({
        "pid": pid, "kind": "bid",
        "contract": _bid_label(bid), "trump": trump, "rung_index": rung.index,
    })


def _human_bundle(sess: Session, rung, trump: Optional[str], discard_ids: List[int]) -> tuple:
    cards12 = list(sess.a_hands[sess.seat]) + list(sess.a_talon)
    discard = [c for c in cards12 if c.id in discard_ids]
    if len(discard) != 2:
        raise HTTPException(status_code=400, detail="must discard exactly 2 of your 12 cards")
    hand10 = [c for c in cards12 if c.id not in discard_ids]
    # EV for the AI's overcall threshold. A colored contract whose trump is
    # DEFERRED (you pick the suit after the auction) is scored at the BEST over the
    # candidate suits — your true strength, without revealing which suit you'll play.
    # (betli/színtelen-duri use a placeholder trump; their EV is colorless-only.)
    if trump is not None:                       # piros (hearts) or an explicit suit
        ev = rung_ev(rung, _provider().base_probs(hand10, trump), _GP)
    elif rung.colorless:
        ev = rung_ev(rung, _provider().base_probs(hand10, "hearts"), _GP)
    else:                                        # non-piros colored → best over 3 suits
        cand = [rung_ev(rung, _provider().base_probs(hand10, t), _GP)
                for t in ("acorns", "leaves", "bells")]
        cand = [e for e in cand if e is not None]
        ev = max(cand) if cand else None
    ev = float(ev) if ev is not None else float(rung.value)
    return (ev, rung, (None if rung.colorless else trump), discard, hand10)


def _advance_auction(sess: Session) -> None:
    """Run AI bid turns until it's the user's turn or the auction resolves. Any
    seat may open; a full round of passes (3 consecutive) ends it."""
    while True:
        if sess.a_passes >= 3:
            # Everyone passed with no bid → the forehand (the 12-holder, seat 0) gets
            # ONE last look: pick the talon back up and play, or passz for good and
            # pay the penalty. Only offered interactively when the human is forehand;
            # an AI forehand already declined, so it just pays.
            if sess.a_current is None and not sess.a_reclaim_offered and sess.seat == 0:
                sess.a_reclaim_offered = True
                sess.a_turn = 0
                sess.a_awaiting_bid = False    # auction step: Felveszem (play) or Passz (pay)
                return
            _resolve_auction(sess)
            return
        turn = sess.a_turn
        if sess.a_current is not None and turn == sess.a_current["pid"]:
            # Everyone else has passed and it is back to the holder. The holder may
            # take the talon back up and RAISE their own bid — the bluff: win cheap,
            # then climb to your real contract. The human decides (pause here); an
            # AI holder declines (it always bids its true best, nothing to raise to).
            if turn == sess.seat:
                return
            sess.a_passes += 1
            sess.a_turn = (turn + 1) % 3
            continue
        if turn == sess.seat:
            return  # user decides (open or overcall)
        # AI seat
        if sess.a_current is None:
            pick = _bid_ai(sess, turn, None)
            ok = pick is not None and pick[0] > -PASS_PENALTY
        else:
            pick = _bid_ai(sess, turn, sess.a_current["rung"])
            ok = pick is not None and pick[0] > -sess.a_current["ev"]
        if ok:
            _apply_bid(sess, turn, pick)
            sess.a_passes = 0
        else:
            # Only the FOREHAND puts the talon back (buries 2) when passing the
            # opening — it's the one holding the 12. Sheds the 2 least-useful cards
            # so it hands the next picker-up junk. Later passers never held the
            # talon, so they decline without touching it.
            if sess.a_current is None and not sess.a_history:
                disc, hand10 = _weakest_two(list(sess.a_hands[turn]) + list(sess.a_talon), None)
                sess.a_hands[turn] = hand10
                sess.a_talon = disc
            sess.a_history.append({"pid": turn, "kind": "pass"})
            sess.a_passes += 1
        sess.a_turn = (turn + 1) % 3


def _resolve_auction(sess: Session) -> None:
    sess.a_done = True
    if sess.a_current is None:
        sess.a_winner = None
        _redeal(sess)               # dead deal → fresh hand, same dealer (no empty "passed" screen)
        return
    sess.a_winner = sess.a_current["pid"]
    # The human won a plain colored game with the trump deferred → declare it now,
    # before play (Ulti: the color is hidden until the game begins). AI winners and
    # piros/colorless games already carry their trump.
    rung = sess.a_current["rung"]
    if sess.a_winner == sess.seat and sess.a_current["trump"] is None and not rung.colorless:
        sess.phase = "trump_select"
        return
    _setup_play(sess)
    _advance_play(sess)


def _redeal(sess: Session) -> None:
    """Everybody passed and the forehand declined the reclaim → the deal is DEAD, so
    we RE-DEAL a fresh hand with the SAME dealer and reopen the auction (milan's rule,
    docstring: "if all three pass the deal is dead → redeal"). No game is scored and the
    UI never lands on an empty screen — a new hand simply appears. Re-runs the AI bid
    turns so we pause back on the user's turn."""
    sess.redeals += 1
    sess.seed = (sess.seed * 1103515245 + 12345 + sess.redeals) & 0x7FFFFFFF
    sol12, d1, d2 = deal_12_10_10(sess.seed)
    sess.a_hands = [list(sol12[:10]), list(d1), list(d2)]
    sess.a_talon = list(sol12[10:])
    sess.a_current = None
    sess.a_passes = 0
    sess.a_turn = 0
    sess.a_reclaim_offered = False
    sess.a_done = False
    sess.a_winner = None
    sess.a_history = []
    sess.a_awaiting_bid = (sess.seat == 0)
    sess.a_picked_up = False
    sess.bubbles = []
    sess.result = None
    sess.phase = "bid"
    sess.p_seed_counter = sess.seed * 31337
    _advance_auction(sess)


# ── Auction -> Play setup ───────────────────────────────────────────────────────

# Hungarian suit names for the adu (trump) announcement — the declarer names the
# color out loud before the first card ("Színe tök!"). The specific suit stays
# hidden throughout the auction and is only revealed here.
_SUIT_HU = {"hearts": "piros", "acorns": "makk", "leaves": "zöld", "bells": "tök"}


def _setup_play(sess: Session) -> None:
    """Re-index the winner to play-index 0 and build the pis position (no play yet)."""
    w = sess.a_winner
    sol = list(sess.a_hands[w])
    d1 = list(sess.a_hands[(w + 1) % 3])
    d2 = list(sess.a_hands[(w + 2) % 3])
    talon = list(sess.a_talon)
    rung = sess.a_current["rung"]
    trump = sess.a_current["trump"]

    sess.rung = rung
    sess.trump = trump
    sess.human_play_index = (sess.seat - w) % 3

    bid = sess.a_current["bid"]
    sess.bid = bid
    sess.bid_name = _bid_label(bid)

    n_trick = int(bid.ulti) + int(bid.durchmars) + int(bid.betli)
    if bid.betli:
        solve_c, build_c, t, restrict, weights = "betli", "betli", None, None, None
    elif bid.durchmars and rung.colorless and n_trick == 1:
        solve_c, build_c, t, restrict, weights = "durchmars", "durchmars", None, None, None
    else:
        solve_c, build_c, t = "multi", "parti", trump
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
        weights = _play_weights(bid, sol, trump)

    sess.p_solve_contract = solve_c
    sess.p_build_contract = build_c
    sess.p_restrict = restrict
    sess.p_weights = weights
    sess.play_hands0 = [sol, d1, d2]
    sess.play_talon = talon

    with _play_lock:
        if weights is not None:
            set_multi_weights(**weights)
        sess.p_pos = pis_bridge.build_position(
            hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0,
            contract=build_c, trump=t, talon=list(talon),
            declare_marriages=(t is not None), marriage_restrict=restrict,
        )
    sess.voids = _det.Voids()
    # Live kontra-able units: a simple game has one (its primary); a combined game
    # exposes each committed unit separately. Colorless (betli / no-trump duri) keep
    # separate per-defender counters; colored units are shared (együtt sírunk).
    sess.k_units = _kontra_units(bid)
    sess.k_colorless = (trump is None)
    sess.k_def = {U: {1: False, 2: False} for U in sess.k_units}
    sess.k_rekontra = {U: False for U in sess.k_units}
    sess.phase = "play"

    # Marriage announcements (bemondás) → "Van 40-em!" / "Van 20-am!".
    marr: Dict[int, str] = {}
    for player in range(3):
        parts = [f"Van {pts}-{'em' if pts == 40 else 'am'}!"
                 for (p, _suit, pts) in getattr(sess.p_pos, "marriages", []) if p == player]
        if parts:
            marr[player] = " ".join(parts)
    # Defenders call their marriage as they play their first card (play-index P at ply P).
    for player in (1, 2):
        if player in marr:
            sess.bubbles.append({"player": player, "text": marr[player], "ply": player})
    # The soloist declares the adu color AND their own marriage together, up front
    # (ply=-1 → the instant play starts) — one bubble so BOTH are visible instead of
    # the trump announcement clobbering the marriage: "Színe piros! Van 40-em!".
    if sess.trump is not None:
        txt = f"Színe {_SUIT_HU.get(sess.trump, sess.trump)}!"
        if 0 in marr:
            txt += " " + marr[0]
        sess.bubbles.append({"player": 0, "text": txt, "ply": -1})
    elif 0 in marr:                              # trumpless guard (colorless games have no marriage)
        sess.bubbles.append({"player": 0, "text": marr[0], "ply": 0})


# ── Kontra (simple contracts only) ──────────────────────────────────────────────

_UNITS_ORDER = ("parti", "ulti", "40_100", "20_100", "durchmars", "betli")
_UNIT_HU = {"parti": "parti", "ulti": "ulti", "40_100": "40-100", "20_100": "20-100",
            "durchmars": "durchmars", "betli": "betli"}
# Per-unit god objective (mirrors exp23/exp27 _OBJ): (solver, multi-weights, marriage_restrict).
# 100-games are solved on the "multi" objective with a score_geq_100 weight.
_UNIT_OBJ = {
    "parti":     ("parti", None, None),
    "ulti":      ("ulti", None, None),
    "betli":     ("betli", None, None),
    "durchmars": ("durchmars", None, None),
    "40_100":    ("multi", {"score_geq_100": 1.0}, "40"),
    "20_100":    ("multi", {"score_geq_100": 1.0}, "20"),
}


def _kontra_primary(bid) -> Optional[str]:
    if not _is_simple(bid):
        return None
    if bid.betli:
        return "betli"
    if bid.ulti:
        return "ulti"
    if bid.durchmars:
        return "durchmars"
    return "parti"


def _kontra_units(bid) -> List[str]:
    """Every kontra-able unit of this game, in UNITS order — each kontrázható separately.
    An ulti / 100 / plain game ALSO exposes the card-point párti as its own unit (milan:
    a piros ulti → kontra ulti AND kontra párti). betli and the all-tricks durchmars have
    no separate párti."""
    if bid.betli:
        return ["betli"]
    units: List[str] = []
    if bid.ulti:
        units.append("ulti")
    if bid.durchmars:
        units.append("durchmars")
    if bid.forty_hundred:
        units.append("40_100")
    if bid.twenty_hundred:
        units.append("20_100")
    if not bid.durchmars:              # card-point párti exists unless the game is all-tricks (duri)
        units.append("parti")
    return sorted(units, key=lambda u: _UNITS_ORDER.index(u))


def _unit_makeability(sess: Session, viewer: int, unit: str, salt: int) -> float:
    """P(soloist makes `unit` | viewer's own hand) — cheat-clean own-hand sampling,
    god-solved for the unit's objective. Handles the 100-games via the multi solver."""
    from ulti.solvers import determinize as _det
    from ulti.eval.pimc_matchup import god_says_soloist_wins
    solver, weights, restrict = _UNIT_OBJ[unit]
    build_c = "durchmars" if solver == "durchmars" else ("betli" if solver == "betli" else "parti")
    sol, d1, d2 = sess.play_hands0
    trump, talon = sess.trump, sess.play_talon
    with _play_lock:
        if weights is not None:
            set_multi_weights(**weights)
        root = pis_bridge.build_position(
            hands=[list(sol), list(d1), list(d2)], soloist=0, leader=0, contract=build_c,
            trump=trump, talon=list(talon), declare_marriages=(trump is not None),
            marriage_restrict=restrict)
        iset = _det.build_info_set(root, viewer, solver, voids=None)
        rng = random.Random(sess.seed + salt)
        w = valid = 0
        for _ in range(_KONTRA_NDET):
            try:
                hands, tal = _det.sample_world(iset, rng)
                spos = (pis_bridge.clone_with_hands_and_talon(root, hands, tal)
                        if iset.talon_known is None else pis_bridge.clone_with_hands(root, hands))
                if weights is not None:
                    set_multi_weights(**weights)
                valid += 1
                if god_says_soloist_wins(spos, contract=solver):
                    w += 1
            except Exception:
                continue
    return w / float(valid) if valid else 0.0


# exp27 per-unit defender-kontra gates (validated 2026-07-21 — held-out tournament vs
# the old blind-makeability rule: +7.7 GP/deal for the defenders, cheat-clean). The old
# rule `_sol_ev(blind_makeability) < 0` wildly over-kontra'd: it sampled RANDOM soloist
# hands and ignored that the soloist BID the contract, so it "saw" ~6-11% makeability
# when the true make is ~80% → kontra'd makeable ulti/parti and paid double (the soloist's
# rekontra amplified the loss further). Per-unit calibrated signals instead — own hand only.
_KONTRA_ULTI_TRUMPS = 4     # ulti: kontra iff this defender holds >=4 trumps (make ~32%; 3→76%)
_KONTRA_DURI_TRUMPS = 3     # colored durchmars: kontra iff >=3 trumps (make ~2-5%; 0→50%)
_KONTRA_PARTI_MAKE  = 0.10  # parti: kontra iff blind makeability ~0 (far more selective than old rule)


def _ai_defender_kontras_unit(sess: Session, pidx: int, U: str) -> bool:
    """Cheat-clean per-unit kontra from this defender's OWN hand. Trump count is the
    decisive signal for the trick contracts (ulti/duri); parti keeps a makeability
    threshold; the 100-games & betli/colorless-duri abstain (no own-hand signal beats
    not kontra-ing)."""
    own = sess.play_hands0[pidx]
    if U == "ulti":
        return sum(1 for c in own if c.suit == sess.trump) >= _KONTRA_ULTI_TRUMPS
    if U == "durchmars" and sess.trump is not None:
        return sum(1 for c in own if c.suit == sess.trump) >= _KONTRA_DURI_TRUMPS
    if U == "parti":
        return _unit_makeability(sess, pidx, "parti", 100 + pidx) < _KONTRA_PARTI_MAKE
    if U == "40_100" and sess.trump is not None:
        # milan 2026-07-23: a 40-100 declares the TRUMP marriage (the "40"). If a defender
        # holds a card of the trump marriage (K or felső of trump), the soloist can't hold the
        # full trump marriage → the 40-100 is unmakeable → auto-kontra.
        return any(c.suit == sess.trump and c.rank in ("king", "upper") for c in own)
    # 20-100: the 20's colour is NOT declared, so the trump test doesn't apply — the rule is
    # about the NON-trump marriages and is still being pinned down with milan. Abstain for now.
    return False               # 20-100 / betli / colorless durchmars → abstain


def _unit_makeability_post_trick1(sess: Session, unit: str, salt: int) -> float:
    """P(soloist makes `unit`) from the SOLOIST's view AFTER trick 1 — they now know
    how trick 1 went. Sample the defenders' remaining hands from the soloist's current
    info set, rebuild a fresh unit-framed deal, REPLAY trick 1, then god-solve. Falls
    back to the pre-trick-1 root signal for objectives that can't cleanly replay."""
    solver, weights, restrict = _UNIT_OBJ[unit]
    if solver == "multi":            # 100-games: no clean replay path → root signal
        return _unit_makeability(sess, 0, unit, salt)
    from ulti.solvers import determinize as _det
    from ulti.eval.pimc_matchup import god_says_soloist_wins
    trump = sess.trump
    plays = [(h["player_id"], card_from_id(h["card"]["id"])) for h in sess.p_history[:3]]
    iset = _det.build_info_set(sess.p_pos, 0, sess.p_solve_contract, voids=sess.voids.as_dict())
    rng = random.Random(sess.seed + salt)
    w, valid = 0, 0
    for _ in range(_KONTRA_NDET):
        try:
            rem, _tal = _det.sample_world(iset, rng)          # remaining hands at current pos
            init = [list(rem[p]) for p in range(3)]
            for pid, card in plays:
                init[pid].append(card)                         # rebuild the pre-trick-1 hands
            root = pis_bridge.build_position(
                hands=init, soloist=0, leader=0, contract=solver, trump=trump,
                talon=list(sess.play_talon), declare_marriages=(trump is not None))
            for _pid, card in plays:
                pis_bridge.apply_move(root, card)              # replay trick 1
            valid += 1
            if god_says_soloist_wins(root, contract=solver):
                w += 1
        except Exception:
            continue
    if valid == 0:
        return _unit_makeability(sess, 0, unit, salt)          # fall back to the pre-trick-1 signal
    return w / float(valid)


def _ai_soloist_rekontras_unit(sess: Session, U: str) -> bool:
    # The rekontra comes AFTER trick 1 — decide from what the soloist has now seen.
    p = _unit_makeability_post_trick1(sess, U, 200 + _UNITS_ORDER.index(U))
    return _sol_ev(p, sess.bid, 0) > 0


def _recompute_k_level(sess: Session) -> None:
    lvl = 0
    for U in sess.k_units:
        for pidx in (1, 2):
            if sess.k_def.get(U, {}).get(pidx):
                lvl = max(lvl, 2 if sess.k_rekontra.get(U) else 1)
    sess.k_level = lvl


def _available_units(sess: Session, pidx: int) -> List[str]:
    """Units a defender may still kontra at their decision point. Colored units are
    shared (együtt sírunk) → drop ones ANY defender already kontra'd; colorless keep
    separate per-defender counters → drop only ones THIS defender already kontra'd."""
    out = []
    for U in sess.k_units:
        d = sess.k_def.get(U, {})
        taken = d.get(pidx) if sess.k_colorless else (d.get(1) or d.get(2))
        if not taken:
            out.append(U)
    return out


def _kontra_dict(sess: Session) -> dict:
    """Per-UNIT kontra levels for the oracle. Colored units are SHARED (együtt sírunk —
    both defenders together); colorless (betli / no-trump duri) may differ per defender
    (separate counters)."""
    if not sess.k_units:
        return {}
    out: dict = {}
    for U in sess.k_units:
        d = sess.k_def.get(U, {})
        def lvl(pidx: int) -> int:
            if not d.get(pidx):
                return 0
            return 2 if sess.k_rekontra.get(U) else 1
        d1, d2 = lvl(1), lvl(2)
        if d1 == 0 and d2 == 0:
            continue
        if sess.k_colorless:             # separate counters (def0=pidx1, def1=pidx2)
            out[U] = (d1, d2)
        else:                            # colored → shared
            out[U] = max(d1, d2)
    return out


def _next_kontra_offer(sess: Session):
    """The next kontra decision to offer given trick-1 play so far → (role, pidx,
    available_units) or None. Each defender is offered once, right after playing their
    first card (play-index 1 at ply 1, 2 at ply 2), on the units still open to them;
    the soloist's rekontra comes after trick 1 (>=3 plies) once, on the kontra'd units.
    Defenders with nothing left to kontra are auto-skipped (marks k_off)."""
    if not sess.k_units:
        return None
    plies = len(sess.p_history)
    for pidx in (1, 2):
        if plies > pidx and not sess.k_off[pidx]:
            avail = _available_units(sess, pidx)
            if avail:
                return ("def", pidx, avail)
            sess.k_off[pidx] = True       # nothing left to kontra → auto-skip
    if plies >= 3 and not sess.k_rk_off:
        kontrad = [U for U in sess.k_units
                   if sess.k_def.get(U, {}).get(1) or sess.k_def.get(U, {}).get(2)]
        if kontrad:
            return ("sol", 0, kontrad)
        sess.k_rk_off = True              # nothing was kontra'd → no rekontra
    return None


def _apply_kontra_ai(sess: Session, role: str, pidx: int, avail: List[str]) -> None:
    """A non-human seat's per-unit kontra/rekontra decision (own-hand makeability)."""
    if role == "def":
        sess.k_off[pidx] = True
        hit = [U for U in avail if _ai_defender_kontras_unit(sess, pidx, U)]
        for U in hit:
            sess.k_def[U][pidx] = True
        if hit:
            labels = ", ".join(_UNIT_HU.get(U, U) for U in hit)
            sess.bubbles.append({"player": pidx, "text": f"Kontra! ({labels})", "ply": pidx})
    else:                                # soloist rekontra
        sess.k_rk_off = True
        hit = [U for U in avail if _ai_soloist_rekontras_unit(sess, U)]
        for U in hit:
            sess.k_rekontra[U] = True
        if hit:
            labels = ", ".join(_UNIT_HU.get(U, U) for U in hit)
            sess.bubbles.append({"player": 0, "text": f"Rekontra! ({labels})", "ply": 3})
    _recompute_k_level(sess)


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
    (ulti / 40-100 / …), since `bid.teritett` is set on all of them (milan 2026-07-24)."""
    return bool(getattr(sess.bid, "teritett", False)) and len(sess.p_history) >= 3


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
    return ch


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


# ── Serialization ───────────────────────────────────────────────────────────────

def _legal_bids(sess: Session) -> List[dict]:
    current = sess.a_current["rung"] if sess.a_current else None
    out: List[dict] = []
    # passz only on the forehand's VERY FIRST turn (the opening). Once you've picked
    # the talon up (an overcall, or the all-pass reclaim), you're committed to a bid,
    # and start-play / accept is handled by the auction-step Elfogadom button.
    if sess.a_current is None and not sess.a_history:
        out.append({"kind": "pass", "rung_index": -1, "bid_index": -1,
                    "label": "passz", "value": -2,
                    "piros": False, "colorless": False, "trump_options": []})
    for r in overcalls(current):
        if r.colorless:
            trumps = []
        elif r.piros:
            trumps = ["hearts"]
        else:
            trumps = ["acorns", "leaves", "bells"]
        # One option per interchangeable contract on the rung, so the user picks
        # the specific game (40-100-duri vs ulti-duri) rather than us guessing.
        for bi, b in enumerate(r.bids):
            out.append({
                "kind": "bid", "rung_index": r.index, "bid_index": bi,
                "label": _bid_label(b), "value": r.value,
                "piros": r.piros, "colorless": r.colorless, "trump_options": trumps,
            })
    return out


def _auction_snapshot(sess: Session) -> dict:
    is_turn = (not sess.a_done) and (sess.a_turn == sess.seat)
    # You are the holder deciding whether to raise iff it's your turn and you own
    # the standing bid (everyone else has passed back to you).
    is_holder = is_turn and sess.a_current is not None and sess.a_current["pid"] == sess.seat
    # forehand's post-all-pass last look: pick the talon back up (bid) or passz → pay
    reclaim = is_turn and sess.a_reclaim_offered and sess.a_current is None
    awaiting_bid = is_turn and sess.a_awaiting_bid        # bid step: you hold 12
    # auction step (holding 10): you may pick the talon up iff a legal bid exists
    cur_rung = sess.a_current["rung"] if sess.a_current else None
    can_pickup = is_turn and (not sess.a_awaiting_bid) and bool(overcalls(cur_rung))
    own = sorted(sess.a_hands[sess.seat], key=lambda c: c.id)
    bid_hand = None
    talon_ids: List[int] = []
    if awaiting_bid:                # bid step → reveal your 10 + the talon (12)
        talon_ids = [c.id for c in sess.a_talon]
        twelve = sorted(list(sess.a_hands[sess.seat]) + list(sess.a_talon), key=lambda c: c.id)
        bid_hand = [card_to_dict(c) for c in twelve]
    cur = None
    if sess.a_current is not None:
        cur = {
            "pid": sess.a_current["pid"],
            "contract": _bid_label(sess.a_current["bid"]),
            # Ulti keeps the adu (trump) color hidden during the auction — the
            # contract name alone is public (piros/colorless are named in it). The
            # specific suit is announced only when play begins (see _setup_play).
            "trump": None,
            "rung_index": sess.a_current["rung"].index,
        }
    return {
        "own_hand": [card_to_dict(c) for c in own],
        "bid_hand": bid_hand,
        "talon_ids": talon_ids,
        "auction": {
            "turn": None if sess.a_done else sess.a_turn,
            "current": cur,
            "passes": sess.a_passes,
            # Strip the trump from bid history too — hidden until announced.
            "history": [dict(h, trump=None) for h in sess.a_history],
            "done": sess.a_done,
            "winner": sess.a_winner,
            "is_human_turn": is_turn,
            "is_holder": is_holder,
            "reclaim": reclaim,
            "awaiting_bid": awaiting_bid,
            "can_pickup": can_pickup,
            "opening": sess.a_current is None,
            "picked_up": awaiting_bid and sess.a_picked_up,
            "legal_bids": _legal_bids(sess) if awaiting_bid else None,
        },
    }


def _play_hands_dict(sess: Session, reveal_all: bool, reveal_sol: bool = False) -> List[List[Optional[dict]]]:
    hands = pis_bridge.hands_by_player(sess.p_pos)
    out: List[List[Optional[dict]]] = []
    for pid in range(3):
        h = sorted(hands[pid], key=lambda c: c.id)
        # terített reveal: the soloist (play-index 0) is shown to everyone once open
        if reveal_all or pid == sess.human_play_index or (reveal_sol and pid == 0):
            out.append([card_to_dict(c) for c in h])
        else:
            out.append([None for _ in h])
    return out


def _play_snapshot(sess: Session) -> dict:
    terminal = pis_bridge.is_terminal(sess.p_pos)
    in_play = sess.phase == "play" and not terminal
    current = pis_bridge.current_player(sess.p_pos) if in_play else None
    legal_ids: Optional[List[int]] = None
    if in_play and current == sess.human_play_index:
        legal_ids = [c.id for c in pis_bridge.legal_actions(sess.p_pos)]
    trick = [
        {"player_id": pid, "card": card_to_dict(pis_bridge._to_o(card))}
        for pid, card in sess.p_pos.trick_cards
    ]
    kontra = None
    if sess.phase == "kontra" and sess.k_next is not None:
        avail = sess.k_next.get("units", [])
        kontra = {
            "pending": {"role": sess.k_next["role"], "play_index": sess.k_next["play_index"]},
            "is_human_turn": True,
            "role": sess.k_next["role"],
            "units": [{"key": U, "label": _UNIT_HU.get(U, U)} for U in avail],
            # primary kept for back-compat display (first available unit)
            "primary": avail[0] if avail else (sess.k_units[0] if sess.k_units else None),
        }
    caps = sess.p_pos.captured
    scores = sess.p_pos.scores
    # Card-point score incl. declared marriages (already in pos.scores). The talon's
    # points count for the DEFENDERS but only the SOLOIST knows the talon during play,
    # so reveal it to the soloist (and everyone at the end), hide it from defenders.
    talon_pts = sum(int(getattr(c, "points", 0)) for c in sess.play_talon)
    reveal_talon = (sess.human_play_index == 0) or terminal
    # Split the marriage DECLARATION bonus (40/20) out of the raw score so the UI can withhold
    # it until each holder "declares" it (with the bemondás bubble): the soloist up front, a
    # defender at their first card. marr[play-index] = that player's declared marriage points.
    marr_pts = [0, 0, 0]
    for (mp, _suit, mpts) in getattr(sess.p_pos, "marriages", []):
        marr_pts[mp] += int(mpts)
    def_talon = talon_pts if reveal_talon else 0
    score = {
        "sol_points": int(scores[0]),                          # full (back-compat)
        "def_points": int(scores[1]) + int(scores[2]) + def_talon,
        "talon_points": talon_pts if reveal_talon else None,   # null = hidden from you
        "sol_card": int(scores[0]) - marr_pts[0],              # card points only (no marriage bonus)
        "def_card": int(scores[1]) + int(scores[2]) - marr_pts[1] - marr_pts[2] + def_talon,
        "marr": marr_pts,                                      # [soloist, def1, def2] bonus
    }
    # The human's own captured cards (won tricks), for the bottom-of-screen pile.
    captured = [card_to_dict(pis_bridge._to_o(c)) for c in caps[sess.human_play_index]]
    return {
        "soloist": 0,
        "human_play_index": sess.human_play_index,
        "contract": sess.bid_name,
        "contract_value": int(sess.rung.value),
        "trump": sess.trump,
        "kontra_level": sess.k_level,
        "kontra": kontra,
        "score": score,
        "captured": captured,
        # The 2 talon cards: count is always known (rendered face-down beside the
        # table); the actual cards are revealed only at game end.
        "talon_count": len(sess.play_talon),
        "talon": ([card_to_dict(c) for c in sess.play_talon] if terminal else None),
        "reveal_soloist": _terit_revealed(sess),               # terített: soloist hand face-up
        "hands": _play_hands_dict(sess, reveal_all=terminal, reveal_sol=_terit_revealed(sess)),
        "hand_sizes": [len(h) for h in pis_bridge.hands_by_player(sess.p_pos)],
        "current_trick": trick,
        "current_player": current,
        "legal_card_ids": legal_ids,
        "history": list(sess.p_history),
        "terminal": terminal,
        "result": sess.result,
    }


def _trump_snapshot(sess: Session) -> dict:
    """You won a plain colored game — declare the trump before play begins."""
    hand = sorted(sess.a_hands[sess.seat], key=lambda c: c.id)
    return {
        "contract": _bid_label(sess.a_current["bid"]),
        "trump_options": ["acorns", "leaves", "bells"],
        "own_hand": [card_to_dict(c) for c in hand],
    }


def _snapshot(sess: Session) -> dict:
    base = {"game_id": sess.id, "seat": sess.seat, "seed": sess.seed, "phase": sess.phase}
    # Drain speech-bubble events (send once, then clear) — matches trickster.
    base["bubbles"] = list(sess.bubbles)
    sess.bubbles = []
    if sess.phase in ("bid", "passed"):
        base.update(_auction_snapshot(sess))
        if sess.phase == "passed":
            base["result"] = sess.result      # all-pass penalty outcome
    elif sess.phase == "trump_select":
        base.update(_trump_snapshot(sess))
    else:
        base.update(_play_snapshot(sess))
    return base


def _get(game_id: str) -> Session:
    with _sessions_lock:
        sess = _sessions.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"unknown game_id {game_id}")
    return sess


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
        "initial_hands": [[card_to_dict(c) for c in h] for h in (sol, d1, d2)],
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
