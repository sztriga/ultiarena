"""Equivalence blocks — the "these cards are provably the same move" rule, in Python.

The Cython solver applies this rule internally as a *cull*: inside the search it keeps
one representative per block so alpha-beta doesn't re-explore identical lines. That is a
SPEED optimisation, and it is invisible outside the search.

This module states the same rule so it can also be used at PLAY time, on whatever picked
the card — the PIMC search, the exp36 betli-defense net, anything. Two cards in one block
lead to literally the same game, so swapping one for another is free.

Why we want that: always playing the top of a block is a **tell**. If the engine
deterministically leads the highest card of an equivalent run, an observant opponent
learns "they hold nothing above this". Humans pick arbitrarily inside a block; the engine
should too. :func:`equivalent_moves` gives the block, the caller picks at random.

The rule (mirrors ``_cull_parti_blocks`` / ``_cull_betli_def_groups``)
---------------------------------------------------------------------
Two same-suit cards A < B in the mover's hand are interchangeable iff

  1. no card of that suit with strength strictly between them is still *live*
     (in another hand or on the table) — captured cards are out of circulation and
     do NOT break a run ("plugged hole"), and
  2. they carry the same card points (0 vs 10), and
  3. neither is the trump 7 (it carries the ulti / silent-ulti payoff on its own).

DELIBERATELY CONSERVATIVE vs the solver
---------------------------------------
This rule is *stricter* than any individual cull in the engine, on purpose — every block
it produces is a subset of a block the solver already proved equivalent, so it is safe
under all of them without needing per-contract configuration:

  * points always split, even in binary contracts (ulti/duri ignore points → they merge
    MORE than we do);
  * the trump 7 is always isolated, not just when ``has_ulti``;
  * :func:`live_others` uses PUBLIC information only — every unseen card counts as live,
    including the talon — whereas the solver reads exact hands in a determinized world.
    Unseen-as-live can only ADD holes, i.e. split blocks further, never merge wrongly.

NOT covered: ``_cull_betli_soloist`` (keep the highest card per suit, gaps ignored) is a
*dominance* cull, not an equivalence one — the cards it drops are worse, not equal. Never
randomise over that. Callers skip mixing for the betli soloist.

``tests/test_block_equivalence.py`` proves the rule empirically: every card in a block
gets the identical exact minimax value from ``solve_root``.
"""
from __future__ import annotations

from typing import Any, FrozenSet, Iterable, List, Optional, Sequence

from ulti.card import COLORLESS_RANK, DECK, Card

_TEN_POINT_RANKS = ("10", "ace")

# Sentinel point value that forces a card into a block of its own.
_ISOLATED = -1


def strength(card: Card, colorless: bool) -> int:
    """Rank strength within a suit. Colourless games demote the Ten under the Jack."""
    return COLORLESS_RANK[card.rank] if colorless else card.rank_index


def _points(card: Card) -> int:
    return 10 if card.rank in _TEN_POINT_RANKS else 0


def equivalence_blocks(
    cards: Sequence[Card],
    others_live: Iterable[Card],
    *,
    colorless: bool,
    isolate: FrozenSet[int] = frozenset(),
) -> List[List[Card]]:
    """Partition ``cards`` into blocks of provably interchangeable moves.

    ``cards``        the mover's candidate moves (normally the legal actions).
    ``others_live``  every card that may still be played by someone else or is on the
                     table. Captured cards must NOT be included — they are out of
                     circulation and cannot break a run.
    ``isolate``      card ids that must never share a block (the trump 7).

    Blocks are returned highest-strength-last, in ascending order per suit.
    """
    live_str: dict[int, set[int]] = {}
    for c in others_live:
        live_str.setdefault(c.suit_index, set()).add(strength(c, colorless))

    by_suit: dict[int, List[Card]] = {}
    for c in cards:
        by_suit.setdefault(c.suit_index, []).append(c)

    blocks: List[List[Card]] = []
    for suit, group in by_suit.items():
        group = sorted(group, key=lambda c: strength(c, colorless))
        holes = live_str.get(suit, set())
        cur: List[Card] = [group[0]]
        for prev, card in zip(group, group[1:]):
            prev_s, cur_s = strength(prev, colorless), strength(card, colorless)
            gap = any(s in holes for s in range(prev_s + 1, cur_s))
            prev_p = _ISOLATED if prev.id in isolate else _points(prev)
            cur_p = _ISOLATED if card.id in isolate else _points(card)
            if gap or cur_p != prev_p or prev_p == _ISOLATED:
                blocks.append(cur)
                cur = [card]
            else:
                cur.append(card)
        blocks.append(cur)
    return blocks


def live_others(pos: Any, viewer: int) -> List[Card]:
    """Cards that may still be held by someone else or sit on the table — PUBLIC info.

    Everything the ``viewer`` has not seen leave play: opponents' hands, the current
    trick, and (conservatively — a defender cannot see it) the talon. Cards already
    captured in completed tricks are excluded: they are out of circulation.
    """
    from ulti.solvers import pis as _pis

    mine = {c.id for c in _pis.hands_by_player(pos)[viewer]}
    gone = {_pis._to_o(c).id for cap in pos.captured for c in cap}
    return [c for c in DECK if c.id not in mine and c.id not in gone]


def equivalent_moves(
    pos: Any,
    viewer: int,
    card: Card,
    *,
    colorless: bool,
    trump: Optional[str] = None,
) -> List[Card]:
    """Every legal move at ``pos`` provably interchangeable with ``card``.

    Always contains ``card`` itself. Uses public information only, so it is safe to call
    from real play (no peeking at hidden hands).
    """
    from ulti.solvers import pis as _pis

    legal = _pis.legal_actions(pos)
    isolate: FrozenSet[int] = frozenset()
    if trump is not None:
        isolate = frozenset({Card(suit=trump, rank="7").id})
    for block in equivalence_blocks(legal, live_others(pos, viewer),
                                    colorless=colorless, isolate=isolate):
        if any(c.id == card.id for c in block):
            return block
    return [card]
