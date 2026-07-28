"""Local copy of exp21 recipe.sol_marriages (hand-only, no import coupling)."""
from __future__ import annotations


def sol_marriages(hand, trump):
    """(has_40, has_20) for a soloist hand (ulti.card.Card list).

    has_40 = holds king+upper of the trump suit.
    has_20 = holds king+upper of at least one non-trump suit.
    """
    kings  = {c.suit for c in hand if c.rank == "king"}
    uppers = {c.suit for c in hand if c.rank == "upper"}
    pairs  = kings & uppers
    has_40 = trump in pairs
    has_20 = any(s != trump for s in pairs)
    return has_40, has_20
