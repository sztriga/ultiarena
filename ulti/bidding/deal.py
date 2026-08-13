"""THE dealer — 12 cards to the forehand (incl. the 2-card talon), 10-10 to the
defenders. Every deal in the app, the pipeline and the harnesses comes from here,
so a seed names the same hands everywhere."""
from __future__ import annotations

import random

from ulti.card import fresh_deck


def deal_12_10_10(seed):
    rng = random.Random(seed)
    deck = fresh_deck()
    rng.shuffle(deck)
    return deck[:12], deck[12:22], deck[22:32]
