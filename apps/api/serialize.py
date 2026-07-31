"""JSON-friendly card views for the web UI.

The frontend sees structured card objects ({suit, rank, id}), never internal
types. Conversion happens here, in one place.

NB: there is deliberately no hand-sorting helper here. Display order lives in
ulti.card.sort_hand — one rule, because the colourless contracts read the Ten low.
"""
from __future__ import annotations

from typing import Dict

from ulti.card import Card


def card_to_dict(card: Card) -> Dict:
    return {'suit': card.suit, 'rank': card.rank, 'id': card.id}
