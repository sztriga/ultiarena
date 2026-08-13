"""Shared substrate for the play API: config knobs, AI singletons, Session store.

Everything the auction/kontra/play/snapshot modules have in common lives here so the
dependency graph stays a fan (engine <- flows <- routes), never a cycle.
"""


import time
import uuid
from contextlib import contextmanager
from threading import RLock
from typing import Dict, List, Optional

from fastapi import HTTPException

# The frontier bidder reads its knobs at import time — apply the deployment profile
# (FLOOR/DEBIAS_PCTL/DURI_TERIT_MULT/KONTRA, one table in ulti.config) BEFORE the
# bidding imports below, so the library sees the deployed defaults; explicit env wins.
# Champion config re-tuned 2026-07-22 on the FIXED bidder (post the auction.py generator
# bug that unlocked non-piros contracts); true head-to-head +0.40 GP/game (exp30/exp32).
from ulti.config import apply_deploy_defaults, env_bool, env_int
apply_deploy_defaults()

from ulti.bidding.ladder import GPTable, contract_name  # noqa: E402
from ulti.bidding.deal import deal_12_10_10  # noqa: E402
from ulti.card import SUIT_HU as _SUIT_HU  # noqa: E402  (one source for suit names)
try:
    from ulti.betli import defense as _exp36  # noqa: E402  (exp36 betli-defense net; models/ulti/betli/betli_defense.pt)
except Exception:  # pragma: no cover
    _exp36 = None

_GP = GPTable()

# ── Exploitative soloist play (exp31 — the champion's biggest play-side lever) ──
# When the AI is the SOLOIST it plays SAFE-EXPLOIT instead of plain PIMC: among the
# moves that don't sacrifice the PIMC-optimal (worst-case) value, it picks the one with
# the best EXPECTED GP against a model of the fallible human defender. Safe = never worse
# than PIMC vs perfect defense, better whenever the human slips; cheat-clean (samples worlds
# from its OWN info set only). Validated exp31 (full-game GP, terített-gated): +0.6..+0.75
# GP/deal vs a fallible defender, 0 regression vs a perfect one, ~1.3-1.6 s/move. Gated OFF
# on terített (open hand → no hidden info to exploit + amplified stakes). EXPLOIT=0 disables.
_EXPLOIT      = env_bool("EXPLOIT", True)          # deployed ON = the frontier
# (the exploit tuning knobs — EXPLOIT_EPS/NW/FRAC — live where they are read: ai_worker)

# ── exp36: learned betli-DEFENSE net for a PLAIN (hidden-info) betli — PROMOTED 2026-07-24 (BETLI_DEF=0
# reverts to PIMC). −21pp soloist steal vs PIMC (benchmark). Scoped to PLAIN betli — terített betli is
# already defended near-god by the reveal (must_hold). Falls back to PIMC if the net is unavailable.
# (the exp37/39 betli/rebetli BIDDING gates live in ulti.bidding.frontier, which owns
#  the deployed bidder configuration — see _bid_fn below.)
_BETLI_DEF = env_bool("BETLI_DEF", True)
# ── anti-tell: randomise inside an equivalence block (MIX_EQUIV=0 reverts). Whatever picks
# the card — PIMC, the exp36 net, the exploit soloist — deterministically returns the same
# member of a provably-equivalent run, which leaks "I hold nothing above this". Mixing is
# free: block members lead to the same game (tests/ulti/test_block_equivalence.py).
_MIX_EQUIV = env_bool("MIX_EQUIV", True)


def _bid_label(bid) -> str:
    """Display name for a specific contract. Hungarian 'színtelen' for the
    no-trump games (the internal ladder name says 'colorless', which is
    test-locked — rename only at the UI boundary)."""
    return contract_name(bid).replace("colorless", "színtelen")

# The play solver's multi-game weights (`set_multi_weights`) are PROCESS-GLOBAL,
# so all AI play across every live session must be serialized.
# (the old process-wide _play_lock is gone — AI decisions run in the worker pool,
# see apps.api.ai_pool; in-process solver users take ai_pool.solver_lock)


# ── AI singletons (lazy) ────────────────────────────────────────────────────────

_provider_lock = RLock()
_provider_obj = None
_bid_fn_obj = None


def _bid_fn():
    global _provider_obj, _bid_fn_obj
    with _provider_lock:
        if _bid_fn_obj is None:
            # ulti.bidding.frontier owns the deployed configuration (calibration on, exp37
            # betli head loaded, exp37/39 gates) so research harnesses build the SAME
            # bidder instead of an accidental uncalibrated one. milan 2026-08-02.
            from ulti.bidding.frontier import frontier_bid_fn, frontier_provider
            _provider_obj = frontier_provider()
            _bid_fn_obj = frontier_bid_fn(_provider_obj)
        return _bid_fn_obj


def _provider():
    _bid_fn()
    return _provider_obj


# ── Session ─────────────────────────────────────────────────────────────────────

class Session:
    """One in-flight game: auction state machine → kontra → pis play position.

    ONE engine for every mix of players (docs/MULTIPLAYER.md): `humans` is the set
    of REAL seats driven by people; every other seat is the AI — the same frontier
    AI the solo game runs, because it IS the solo game's code path. The solo game
    is simply `humans={seat}`; a live table passes 1..3 human seats and empty
    chairs stay AI without a single new line of AI code."""

    def __init__(self, *, seat: int, seed: int, humans: Optional[set] = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.lock = RLock()             # serializes ALL actions on this one game (see _hold)
        self.last_touch = time.time()   # idle-expiry clock (see _reap)
        self.humans: set = set(humans) if humans is not None else {seat}
        self.players: Dict[int, dict] = {}       # real seat -> {user_id, username} (live tables)
        self.players_by_user: Dict[str, int] = {}
        self.live = False                        # True = a lobby-table game (apps/api/live.py)
        self.human_pis: set = set()              # human PLAY indices, filled at _setup_play
        self.rev = 0                             # bumped on every mutation → poll adoption
        self.bubbles_seen: Dict[int, int] = {}   # per-viewer bubble cursor (real seat)
        # Identity vs abuse-control are SEPARATE axes: device_id is a client-generated
        # uuid (localStorage) = "whose game is this" — it lists/resumes games; user_id
        # is the logged-in account (apps/api/users.py), set by the route when a bearer
        # token is present. owner_ip (set by the route) is what the caps key on,
        # because a client can mint device ids and tokens at will.
        self.device_id: Optional[str] = None
        self.seat = seat            # the user's real auction seat (0/1/2)
        self.seed = seed
        self.phase = "bid"          # bid → (trump_select) → play ⇄ kontra → done | passed

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
        self.a_awaiting_bid = (0 in self.humans)   # forehand holds the 12; == (seat==0) solo
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
# Idle sessions are reaped so an abandoned tab can't grow the process forever (the
# puzzle sessions already expire; play sessions used to live until process death).
# 6h idle default — no live game with a human at the table idles that long.
_SESSION_TTL = env_int("PLAY_SESSION_TTL", 21600)


def _reap_idle_sessions() -> None:
    """Drop sessions idle past the TTL. Called under _sessions_lock."""
    cutoff = time.time() - _SESSION_TTL
    for gid in [g for g, s in _sessions.items() if s.last_touch < cutoff]:
        _sessions.pop(gid, None)


def _install_session(sess: "Session", request, owner: Optional[str] = None) -> None:
    """EVERY session enters the store through here — solo /play/new, a live table,
    an API match — so the reap + cap-guard + insert are one critical section on
    every path (limits.guard_new_session contract). ``owner`` overrides the cap key
    for sessions not owned by the request IP ("live:<table>", "api:<key>")."""
    from .limits import guard_new_session
    with _sessions_lock:
        _reap_idle_sessions()
        sess.owner_ip = guard_new_session(
            request, _sessions, lambda s: getattr(s, "owner_ip", None), owner=owner)
        _sessions[sess.id] = sess


def solve_plan(bid, trump: Optional[str], sol_cards: Optional[list] = None) -> tuple:
    """Contract → solver objective, stated ONCE: (solve_c, build_c, trump, restrict,
    weights). Live play setup, recorded-game analysis and the public rules kernel all
    map through here — three hand-written copies of this drifted apart once already.

    betli solves as betli; a PURE colorless durchmars (no ulti/100 rider — trump is
    None exactly for the colorless rungs) as durchmars; everything else as the
    weighted "multi" objective built on a parti position, with the 100-games'
    marriage restriction. ``weights`` needs the soloist's cards; pass None when the
    caller only builds a position (the public /legal / /score kernel)."""
    if bid.betli:
        return ("betli", "betli", None, None, None)
    if bid.durchmars and trump is None and not (bid.ulti or bid.forty_hundred
                                                or bid.twenty_hundred):
        return ("durchmars", "durchmars", None, None, None)
    restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
    weights = None
    if sol_cards is not None:
        from ulti.bidding.scorers import _play_weights
        weights = _play_weights(bid, sol_cards, trump)
    return ("multi", "parti", trump, restrict, weights)


def _recipe(sess: "Session") -> dict:
    """Everything a worker needs to rebuild this session's live position — primitives
    only (the ai_pool boundary is pickled). Deterministic: build_position + the move
    history reproduce sess.p_pos exactly (same rebuild /play/analysis always used)."""
    return {
        "hands0": [[c.id for c in h] for h in sess.play_hands0],
        "talon": [c.id for c in sess.play_talon],
        "build_c": sess.p_build_contract, "solve_c": sess.p_solve_contract,
        "trump": sess.trump, "restrict": sess.p_restrict,
        "has_ulti": bool(getattr(sess.bid, "ulti", False)),
        "weights": sess.p_weights,
        "history": [(h["player_id"], h["card"]["id"]) for h in sess.p_history],
        "voids": sess.voids.as_dict() if sess.voids is not None else None,
    }


def _play_index(sess: "Session", seat: int) -> int:
    """Real seat → play index (soloist = 0). Valid once the auction resolved."""
    w = sess.a_winner if sess.a_winner is not None else 0
    return (seat - w) % 3


def _viewer_seat(sess: "Session", request) -> int:
    """Whose eyes render this request. Solo game → the one human, exactly as before
    (in-process callers pass no request). Live table game → the authenticated
    member's seat; strangers are refused even with a valid game_id."""
    if not sess.live:
        return sess.seat
    from .users import user_from_request
    user = user_from_request(request)
    seat = sess.players_by_user.get(user["user_id"]) if user else None
    if seat is None:
        raise HTTPException(status_code=403, detail="Nem ülsz ennél az asztalnál.")
    return seat


def _get(game_id: str) -> Session:
    with _sessions_lock:
        sess = _sessions.get(game_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Nincs ilyen játszma.")
    sess.last_touch = time.time()
    return sess


@contextmanager
def _hold(game_id: str):
    """Look up a session and hold ITS lock for the whole request — every action on one
    game is serialized (two tabs on the same game_id, a double-fired click racing the
    check-then-mutate in a route), while different games stay fully parallel.

    Lock order is safe by construction: _sessions_lock is taken only inside _get and
    RELEASED before sess.lock is acquired; nothing under sess.lock ever takes
    _sessions_lock or another session's lock."""
    sess = _get(game_id)
    with sess.lock:
        yield sess


