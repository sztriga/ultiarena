"""
Perfect-information solver bridge.

Translator between oldtawer types (``ulti.card.Card``) and trickster's
contract-agnostic Cython solver in ``ultisolver._solver_core``. The
solver itself — alpha-beta search, transposition table, principal-
variation walker, contract-specific cull/order/terminal hooks — lives
in the Cython module. This file is a thin adapter; nothing here knows
how to search.

Public API
----------

    build_position(hands, soloist, leader, contract, trump=None)
        Construct an opaque solver position.

    solve_best(pos, contract=None)        → (Card | None, float)
    solve_all(pos, contract=None)         → dict[Card, float]
    principal_variation(pos, contract=None) → list[(player_id, Card)]

    legal_actions(pos), current_player(pos), is_terminal(pos)
    apply_move(pos, card)                 (mutates ``pos``)

    get_stats()                           → diagnostics for last solve

Supported contracts: 'betli', 'ulti', 'parti', 'durchmars'
(see ``CONTRACTS``). Auto-detected from the position when ``contract``
is omitted.

Position objects are opaque — callers should treat them as black boxes
and only manipulate them through the helpers above. They can be
mutated by ``apply_move`` to advance the game (e.g. for branch
exploration: build, apply prefix moves, then solve).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ulti.card import Card as _OCard

# ──────────────────────────────────────────────────────────────────────────────
# Card mapping (oldtawer ↔ trickster)
# ──────────────────────────────────────────────────────────────────────────────
# 32-card decks on both sides; rank order matches semantically. Suit names
# are identical in spirit (Hearts/Bells/Leaves/Acorns); only the integer
# index differs because the two libraries enumerate suits in different
# orders. We translate by name, not index.

from ultisolver.games.ulti.cards import (  # noqa: E402
    Card as _TCard,
    Rank as _TRank,
    Suit as _TSuit,
)

_OSUIT_TO_TSUIT = {
    'acorns': _TSuit.ACORNS,
    'leaves': _TSuit.LEAVES,
    'hearts': _TSuit.HEARTS,
    'bells':  _TSuit.BELLS,
}
_TSUIT_TO_OSUIT = {v: k for k, v in _OSUIT_TO_TSUIT.items()}

_ORANK_TO_TRANK = {
    '7':     _TRank.SEVEN,
    '8':     _TRank.EIGHT,
    '9':     _TRank.NINE,
    'lower': _TRank.JACK,
    'upper': _TRank.QUEEN,
    'king':  _TRank.KING,
    '10':    _TRank.TEN,
    'ace':   _TRank.ACE,
}
_TRANK_TO_ORANK = {v: k for k, v in _ORANK_TO_TRANK.items()}


def _to_t(c: _OCard) -> _TCard:
    return _TCard(suit=_OSUIT_TO_TSUIT[c.suit], rank=_ORANK_TO_TRANK[c.rank])


def _to_o(c: _TCard) -> _OCard:
    return _OCard(suit=_TSUIT_TO_OSUIT[c.suit], rank=_TRANK_TO_ORANK[c.rank])


# ──────────────────────────────────────────────────────────────────────────────
# Contract list (mirrors trickster's _CONTRACT_MAP)
# ──────────────────────────────────────────────────────────────────────────────

CONTRACTS: Tuple[str, ...] = ('betli', 'ulti', 'parti', 'durchmars')


# ──────────────────────────────────────────────────────────────────────────────
# Position construction
# ──────────────────────────────────────────────────────────────────────────────

def build_position(
    *,
    hands:    List[List[_OCard]],
    soloist:  int,
    leader:   int,
    contract: str,
    trump:    Optional[str] = None,
    talon:    Optional[List[_OCard]] = None,
    declare_marriages: bool = False,
    marriage_restrict: Optional[str] = None,
) -> Any:
    """Build an opaque solver position from oldtawer-typed inputs.

    The position starts at trick 1 with the given leader on the move and
    no captures yet. To analyse a mid-game position, build the start-of-
    play state, then ``apply_move`` the prefix moves before solving.

    ``talon`` (optional) records the 2 cards set aside at deal time. They
    do not participate in play but are needed by PIMC determinization to
    represent the soloist's information advantage (soloist sees them,
    defenders do not). Pass them whenever you want PIMC to be correct
    from a defender's perspective.
    """
    if contract not in CONTRACTS:
        raise ValueError(f"unknown contract {contract!r}; expected one of {CONTRACTS}")
    if len(hands) != 3 or any(not h for h in hands):
        raise ValueError(f"need 3 non-empty hands, got sizes {[len(h) for h in hands]}")
    _validate_hands(hands)
    if talon is not None and len(talon) > 2:
        raise ValueError(f"talon must have at most 2 cards, got {len(talon)}")

    from ultisolver.games.ulti.game import GameState as _TGameState

    has_ulti = (contract == 'ulti')
    # Colorless contracts: betli is always trumpless; durchmars MAY be
    # played without trump (colorless duri uses the betli-style rank
    # ordering — 10 demoted under J — handled by setting betli=True).
    is_colorless_duri = (contract == 'durchmars' and trump is None)
    is_betli = (contract == 'betli') or is_colorless_duri

    if trump is None:
        if contract not in ('betli', 'durchmars'):
            raise ValueError(f"contract {contract!r} requires a trump suit")
        ttrump = None
    else:
        if trump not in _OSUIT_TO_TSUIT:
            raise ValueError(f"unknown trump {trump!r}; expected one of {sorted(_OSUIT_TO_TSUIT)}")
        ttrump = _OSUIT_TO_TSUIT[trump]

    # training_mode="durchmars" makes ``is_terminal`` use the duri
    # early-termination (sol lost a trick) instead of the betli one
    # (sol won a trick). Crucial for colorless duri where ``betli=True``
    # would otherwise fire the wrong early-loss check.
    training_mode = "durchmars" if contract == 'durchmars' else None

    state = _TGameState(
        hands=[[_to_t(c) for c in h] for h in hands],
        trump=ttrump,
        betli=is_betli,
        soloist=soloist,
        dealer=(soloist + 2) % 3,        # any non-interfering value
        captured=[[], [], []],
        scores=[0, 0, 0],
        leader=leader,
        trick_no=0,
        trick_cards=[],
        last_trick=None,
        contract_type=None,
        kontra_level=0,
        has_ulti=has_ulti,
        talon_discards=[_to_t(c) for c in talon] if talon else [],
        training_mode=training_mode,
    )
    if declare_marriages:
        from ultisolver.games.ulti.game import declare_all_marriages
        declare_all_marriages(state, soloist_marriage_restrict=marriage_restrict)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Position helpers (for endpoints that need to step the game)
# ──────────────────────────────────────────────────────────────────────────────

def legal_actions(pos: Any) -> List[_OCard]:
    from ultisolver.games.ulti.game import legal_actions as _legal
    return [_to_o(c) for c in _legal(pos)]


def current_player(pos: Any) -> int:
    from ultisolver.games.ulti.game import current_player as _cur
    return _cur(pos)


def is_terminal(pos: Any) -> bool:
    from ultisolver.games.ulti.game import is_terminal as _term
    return _term(pos)


def apply_move(pos: Any, card: _OCard) -> None:
    """Play ``card`` on ``pos`` in place. Raises ``ValueError`` if illegal."""
    from ultisolver.games.ulti.game import legal_actions as _legal, play_card as _play
    tcard = _to_t(card)
    if tcard not in _legal(pos):
        raise ValueError(f"illegal move: {card} not in current legal actions")
    _play(pos, tcard)


def hands_by_player(pos: Any) -> List[List[_OCard]]:
    """Snapshot every player's remaining cards (oldtawer ``Card``)."""
    return [[_to_o(c) for c in h] for h in pos.hands]


def clone_with_hands(pos: Any, hands: List[List[_OCard]]) -> Any:
    """Return a deep copy of ``pos`` with the given per-player hands replacing
    the original ones. Used by PIMC to build a sampled determinization while
    preserving captures, current trick, scores, leader, etc."""
    _validate_hands(hands)
    new_pos = pos.clone()
    new_pos.hands = [[_to_t(c) for c in h] for h in hands]
    return new_pos


def clone_with_hands_and_talon(
    pos:   Any,
    hands: List[List[_OCard]],
    talon: List[_OCard],
) -> Any:
    """Like :func:`clone_with_hands` but also overrides the talon. Useful for
    defender-side PIMC where the talon's 2 cards are part of the sampled
    determinization, not known up front."""
    _validate_hands(hands)
    if len(talon) > 2:
        raise ValueError(f"talon must have at most 2 cards, got {len(talon)}")
    new_pos = pos.clone()
    new_pos.hands = [[_to_t(c) for c in h] for h in hands]
    new_pos.talon_discards = [_to_t(c) for c in talon]
    return new_pos


# ──────────────────────────────────────────────────────────────────────────────
# Hand-shape validation (guards the Cython solver from buffer overflows)
# ──────────────────────────────────────────────────────────────────────────────
# The Cython core hard-codes C_MAX_MOVES = 10 — fixed-size move/sort buffers
# in _legal, _order_*, _cull_parti_blocks. Passing >10 cards in any single
# hand silently corrupts the stack and aborts the process (SIGABRT). Bounce
# bad input here so callers get a clean ``ValueError`` instead of a kill.

_MAX_HAND = 10


def _validate_hands(hands: List[List[_OCard]]) -> None:
    if len(hands) != 3:
        raise ValueError(f"need exactly 3 hands, got {len(hands)}")
    sizes = [len(h) for h in hands]
    for i, n in enumerate(sizes):
        if n > _MAX_HAND:
            raise ValueError(
                f"hand[{i}] has {n} cards; solver max is {_MAX_HAND} "
                f"(did you forget to discard the talon?). sizes={sizes}"
            )
    # Per-hand uniqueness + no cross-hand collisions: silent duplicates also
    # confuse _legal (which works on a 32-bit bitmask and would just OR
    # the duplicate away, producing wrong move lists).
    seen = set()
    for i, h in enumerate(hands):
        ids = set()
        for c in h:
            cid = (c.suit, c.rank)
            if cid in ids:
                raise ValueError(f"hand[{i}] contains duplicate {c}")
            if cid in seen:
                raise ValueError(f"card {c} appears in multiple hands")
            ids.add(cid)
        seen.update(ids)


# ──────────────────────────────────────────────────────────────────────────────
# Solver entry points (wrap the Cython solver)
# ──────────────────────────────────────────────────────────────────────────────

def _solver():
    """Prefer the Cython solver; fall back to pure-Python with a warning."""
    try:
        import ultisolver._solver_core as cy   # type: ignore
        return cy
    except ImportError as e:
        raise RuntimeError(
            "ultisolver._solver_core not built. Run "
            "`python setup_cython.py build_ext --inplace`."
        ) from e


def solve_best(
    pos: Any,
    contract: Optional[str] = None,
) -> Tuple[Optional[_OCard], float]:
    """Best move at ``pos`` and its exact minimax value (soloist-perspective).

    Returns ``(None, 0.0)`` if the position has no legal moves (terminal).
    """
    best, value = _solver().solve_best(pos, contract=contract)
    return (_to_o(best) if best is not None else None, float(value))


def solve_all(
    pos: Any,
    contract: Optional[str] = None,
) -> Dict[_OCard, float]:
    """Exact value of every legal move at ``pos`` (soloist-perspective).

    Use this when you need a per-move ranking (e.g. PIMC vote aggregation
    or showing all leads in the UI). Heavier than ``solve_best`` because
    each move is searched with a fresh ``(-∞, +∞)`` window.
    """
    raw = _solver().solve_root(pos, contract=contract)
    return {_to_o(c): float(v) for c, v in raw.items()}


def principal_variation(
    pos: Any,
    contract: Optional[str] = None,
) -> List[Tuple[int, _OCard]]:
    """Optimal play sequence from ``pos`` to game end.

    Returns a list of ``(player_id, Card)`` tuples. The walker keeps
    the transposition table warm across plies, so each step after the
    first is mostly TT hits.
    """
    return [
        (pid, _to_o(c))
        for pid, c in _solver().principal_variation(pos, contract=contract)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    """Diagnostics from the most recent solver call (nodes, cutoffs, TT)."""
    return dict(_solver().get_stats())
