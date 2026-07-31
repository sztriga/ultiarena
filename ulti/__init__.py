"""Ulti — deck, engine bridges, bidding, scoring, play AI.

The game ENGINE lives in the standalone ``ultisolver`` package (Cython solver +
rules); this package holds everything built around it: the card model, the
solver/PIMC bridges (``solvers``), the base-event bidder (``bidding``), the
scoring oracle (``scoring``), the value nets (``vnet``), betli defense
(``betli``) and the dealers/benchmarks (``eval``).
"""
from .card import Card, SUITS, RANKS, DECK, card_from_id, fresh_deck  # noqa: F401
