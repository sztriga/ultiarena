"""
Hungarian card deck for Ulti.

Suits  (suit_index): acorns=0, leaves=1, hearts=2, bells=3
Ranks  (rank_index in ascending power order):
    7=0, 8=1, 9=2, lower=3, upper=4, king=5, 10=6, ace=7

card_id = suit_index * 8 + rank_index  →  0..31
"""
from __future__ import annotations
from typing import List

SUITS: List[str] = ['acorns', 'leaves', 'hearts', 'bells']
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
