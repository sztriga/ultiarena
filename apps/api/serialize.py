"""
JSON-friendly views of UltiGame state for the web UI.

The frontend never sees the 264-dim observation vector — it sees structured
card objects ({suit, rank, id}) and named phase/licit strings. Conversion
happens here, in one place.
"""
from __future__ import annotations

from typing import Dict, Optional

from ulti.card import Card, RANKS, SUITS, card_from_id
from ulti.game import Licit, Phase, UltiGame


# ──────────────────────────────────────────────────────────────────────────────
# Card / phase / licit serialization
# ──────────────────────────────────────────────────────────────────────────────

def card_to_dict(card: Card) -> Dict:
    return {'suit': card.suit, 'rank': card.rank, 'id': card.id}


def card_id_to_dict(card_id: int) -> Dict:
    return card_to_dict(card_from_id(card_id))


# NB: there is deliberately no hand-sorting helper here. Display order lives in
# ulti.card.sort_hand — one rule, because the colourless contracts read the Ten low.


def phase_name(phase: Phase) -> str:
    return phase.name


def licit_name(licit: Optional[Licit]) -> Optional[str]:
    return licit.name if licit is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# Über (must-beat) reason — mirrors replay.uber_label, structured for UI
# ──────────────────────────────────────────────────────────────────────────────

def uber_reason(game: UltiGame, player_id: int) -> Dict:
    """
    Why is this player constrained right now?

    Returns
    -------
    {
      'kind': 'lead' | 'must_uber' | 'follow_suit' | 'must_trump' | 'free',
      'led_suit': str | None,
      'beat_rank': str | None,    # only for must_uber
    }
    """
    trick = game.current_trick
    if not trick:
        return {'kind': 'lead', 'led_suit': None, 'beat_rank': None}

    led_suit = trick[0][1].suit
    hand = game.hands[player_id]
    same_suit = [c for c in hand if c.suit == led_suit]

    if same_suit:
        max_rank = max(c.rank_index for _, c in trick if c.suit == led_suit)
        higher = [c for c in same_suit if c.rank_index > max_rank]
        if higher:
            return {'kind': 'must_uber', 'led_suit': led_suit,
                    'beat_rank': RANKS[max_rank]}
        return {'kind': 'follow_suit', 'led_suit': led_suit, 'beat_rank': None}

    if any(c.suit == game.trump_suit for c in hand):
        return {'kind': 'must_trump', 'led_suit': led_suit, 'beat_rank': None}
    return {'kind': 'free', 'led_suit': led_suit, 'beat_rank': None}


# ──────────────────────────────────────────────────────────────────────────────
# Final result snapshot
# ──────────────────────────────────────────────────────────────────────────────

def result_snapshot(game: UltiGame) -> Dict:
    """Snapshot the final scoring breakdown after the game is DONE."""
    pay = game.payoffs()
    return {
        'trump_suit': game.trump_suit,
        'licit': licit_name(game.licit),
        'talon': [card_to_dict(c) for c in game.talon],
        'talon_points': game.talon_points(),
        'trick_scores': list(game.trick_scores),
        'king_upper_points': list(game.king_upper_points),
        'payoffs': list(pay),
        'game_points': list(game.game_points()),
        'declarer_wins': bool(game.declarer_wins()),
        'declarer_ulti': bool(game._declarer_ulti),
        'defender_ulti': bool(game._defender_ulti),
        'declarer_has_trump_pair': bool(game._declarer_has_trump_pair),
        'defender_has_trump_pair': bool(game._defender_has_trump_pair),
    }
