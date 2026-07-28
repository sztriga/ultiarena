"""Betli-specific feature encoder (~50 dimensions).

Encodes a hand of 10 or 12 cards into features that capture
betli-relevant structure: safe floors, holes, voids, danger cards.
"""
from __future__ import annotations

import numpy as np

from trickster.games.ulti.cards import ALL_SUITS, BETLI_STRENGTH, Card, Suit

# Betli strength order: 7=0, 8=1, 9=2, 10=3, J=4, Q=5, K=6, A=7
_NUM_RANKS = 8


def encode_hand(hand: list[Card]) -> np.ndarray:
    """Encode a betli hand into a feature vector.

    Works for both 10-card (play) and 12-card (pre-discard) hands.

    Per-suit features (4 suits × 10 = 40):
        held[8]    — bitmask in betli-strength order
        safe_floor — consecutive cards from 7 upward / 8
        first_hole — position of first gap / 8

    Global features (~10):
        num_voids, total_danger, num_suits, max_hole_size,
        total_cards, top_card_avg, total_holes, low_card_count
    """
    # Build per-suit bitmasks
    suit_bits: dict[Suit, list[int]] = {s: [0] * _NUM_RANKS for s in ALL_SUITS}
    for card in hand:
        strength = BETLI_STRENGTH[card.rank]
        suit_bits[card.suit][strength] = 1

    features: list[float] = []

    num_voids = 0
    total_danger = 0
    total_holes = 0
    max_hole_size = 0
    top_cards: list[int] = []
    low_card_count = 0

    for suit in ALL_SUITS:
        bits = suit_bits[suit]
        suit_count = sum(bits)

        # Per-suit: held bitmask (8 features)
        features.extend(float(b) for b in bits)

        # safe_floor: consecutive cards from 7 (strength 0) upward
        safe_floor = 0
        for b in bits:
            if b:
                safe_floor += 1
            else:
                break

        # first_hole: position of first gap (higher = safer)
        first_hole = _NUM_RANKS  # no hole if all held or void
        if suit_count > 0:
            first_hole = 0
            for i, b in enumerate(bits):
                if not b:
                    first_hole = i
                    break
                first_hole = i + 1

        features.append(safe_floor / _NUM_RANKS)
        features.append(first_hole / _NUM_RANKS)

        # Accumulate globals
        if suit_count == 0:
            num_voids += 1
        else:
            # Danger cards = cards above first hole
            danger = sum(bits[i] for i in range(first_hole, _NUM_RANKS)) if first_hole < _NUM_RANKS else 0
            # Actually danger = cards held that are above the first gap
            # Cards above holes can be forced to win tricks
            suit_danger = 0
            for i in range(first_hole + 1, _NUM_RANKS):
                if bits[i]:
                    suit_danger += 1
            total_danger += suit_danger

            # Hole counting for this suit
            in_hand = False
            suit_holes = 0
            suit_max_hole = 0
            current_hole = 0
            for i in range(_NUM_RANKS):
                if bits[i]:
                    in_hand = True
                    if current_hole > 0:
                        suit_holes += 1
                        suit_max_hole = max(suit_max_hole, current_hole)
                        current_hole = 0
                elif in_hand:
                    current_hole += 1
            total_holes += suit_holes
            max_hole_size = max(max_hole_size, suit_max_hole)

            # Top card in this suit
            top = 0
            for i in range(_NUM_RANKS - 1, -1, -1):
                if bits[i]:
                    top = i
                    break
            top_cards.append(top)

            # Low cards: betli strength <= 2 (7, 8, 9)
            low_card_count += sum(bits[i] for i in range(3))

    # Global features
    num_suits = 4 - num_voids
    top_card_avg = (sum(top_cards) / len(top_cards)) if top_cards else 0.0
    total_cards = len(hand)

    features.append(num_voids / 4.0)
    features.append(total_danger / 32.0)
    features.append(num_suits / 4.0)
    features.append(max_hole_size / _NUM_RANKS)
    features.append(total_cards / 12.0)
    features.append(top_card_avg / 7.0)
    features.append(total_holes / 16.0)
    features.append(low_card_count / 12.0)

    return np.array(features, dtype=np.float32)


BETLI_FEATURE_DIM = 4 * (_NUM_RANKS + 2) + 8  # 4 * 10 + 8 = 48
