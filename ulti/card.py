"""
Hungarian card deck for Ulti.

Suits  (suit_index): hearts=0, bells=1, leaves=2, acorns=3
    — piros, tök, zöld, makk: THE order (milan). Since 2026-08-11 the wire
    encoding, the display order and ultisolver's own Suit enum all agree; the
    old alphabetical-English order (acorns first) is history — see
    migrations/suit_reencode_2026_08_11.py for how ids, models and the game
    database moved.
Ranks  (rank_index in ascending power order):
    7=0, 8=1, 9=2, lower=3, upper=4, king=5, 10=6, ace=7

card_id = suit_index * 8 + rank_index  →  0..31
"""
from __future__ import annotations
from typing import List

SUITS: List[str] = ['hearts', 'bells', 'leaves', 'acorns']
RANKS: List[str] = ['7', '8', '9', 'lower', 'upper', 'king', '10', 'ace']

# Aces and 10s score 10 points when won in a trick.
RANK_POINTS = {r: (10 if r in ('10', 'ace') else 0) for r in RANKS}


class Card:
    """Immutable representation of one card in the 32-card Hungarian deck."""

    __slots__ = ('suit', 'rank', 'id', 'suit_index', 'rank_index', 'points')

    def __init__(self, suit: str, rank: str) -> None:
        if suit not in SUITS:
            raise ValueError(f'Unknown suit: {suit!r}')
        if rank not in RANKS:
            raise ValueError(f'Unknown rank: {rank!r}')
        self.suit: str = suit
        self.rank: str = rank
        self.suit_index: int = SUITS.index(suit)
        self.rank_index: int = RANKS.index(rank)
        self.id: int = self.suit_index * 8 + self.rank_index
        self.points: int = RANK_POINTS[rank]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Card) and self.id == other.id

    def __hash__(self) -> int:
        return self.id

    def __str__(self) -> str:
        return f'{self.rank} of {self.suit}'

    def __repr__(self) -> str:
        return f'Card({self.suit!r}, {self.rank!r})'


# Canonical deck: 32 cards in suit-major, rank-minor order.
# Built once at import time; never mutate this list.
DECK: List[Card] = [Card(suit=s, rank=r) for s in SUITS for r in RANKS]

_CARD_BY_ID: dict = {c.id: c for c in DECK}


def card_from_id(card_id: int) -> Card:
    """Look up a Card by its id (0..31)."""
    try:
        return _CARD_BY_ID[card_id]
    except KeyError:
        raise ValueError(f'No card with id {card_id}')


def fresh_deck() -> List[Card]:
    """Return a new list of all 32 cards (unshuffled)."""
    return list(DECK)


# ──────────────────────────────────────────────────────────────────────────────
# Card ordering — ONE definition, used everywhere
# ──────────────────────────────────────────────────────────────────────────────
# Every hand shown to a player goes through hand_sort_key/sort_hand: the play table,
# the auction, the bid step, Villámtalon. The web UI renders whatever order the API
# sends and does no sorting of its own, so changing the order here changes it
# everywhere — there is deliberately nowhere else to change.
#
# Display order == wire order since the 2026-08-11 re-encode — DERIVED from SUITS
# so they can never diverge again. (Kept as a named mapping because "how a hand is
# laid out" and "how a card is numbered" are different questions, even when the
# answer is the same.)
DISPLAY_SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}

# The Hungarian names — defined HERE, next to the canonical orders, so no other
# module ever spells a suit/rank listing by hand (docs generate from these).
SUIT_HU = {'hearts': 'piros', 'bells': 'tök', 'leaves': 'zöld', 'acorns': 'makk'}

# The non-piros trump choices, DERIVED from the display order (tök, zöld, makk) —
# never hand-listed again (milan 2026-08-11).
TRUMP_CHOICES = [s for s in sorted(DISPLAY_SUIT_ORDER, key=DISPLAY_SUIT_ORDER.get)
                 if s != 'hearts']
RANK_HU = {'7': '7', '8': '8', '9': '9', 'lower': 'alsó', 'upper': 'felső',
           'king': 'király', '10': '10', 'ace': 'ász'}

# Colourless games (betli / színtelen duri) demote the Ten under the alsó, so a suit runs
# 7 8 9 10 J Q K A instead of 7 8 9 J Q K 10 A. Same ordering the solver uses for those
# contracts (see ulti.solvers.blocks.strength) — a hand must be read the way it plays.
COLORLESS_RANK = {'7': 0, '8': 1, '9': 2, '10': 3,
                  'lower': 4, 'upper': 5, 'king': 6, 'ace': 7}


def hand_sort_key(card: Card, colorless: bool = False):
    """Display sort key for one card. ``colorless`` → betli / színtelen duri ordering."""
    rank = COLORLESS_RANK[card.rank] if colorless else card.rank_index
    return (DISPLAY_SUIT_ORDER[card.suit], rank)


def sort_hand(cards, colorless: bool = False) -> List[Card]:
    """Return ``cards`` in display order. The one way a hand gets sorted."""
    return sorted(cards, key=lambda c: hand_sort_key(c, colorless))
