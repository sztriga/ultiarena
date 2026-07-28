"""Pickup-net featurizer.

Layout (so suit-permutation augmentation works):
  hand[0:32]    multi-hot per card, indexed by card.id (suit * 8 + rank)
  trump[32:36]  one-hot trump suit (trump-aware contracts only)

card.id = SUITS.index(suit) * 8 + rank_index, so reshaping to (4, 8)
gives a suit-major matrix that can be relabeled via suit permutation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ulti.card import SUITS
from .contracts import ContractCfg

_SUIT_IDX = {s: i for i, s in enumerate(SUITS)}

HAND_DIM       = 32
TRUMP_DIM      = 4
HAND_OFFSET    = 0
TRUMP_OFFSET   = 32


def input_dim(cfg: ContractCfg) -> int:
    return HAND_DIM + TRUMP_DIM if cfg.has_trump else HAND_DIM


def featurize(hand, trump: Optional[str], has_trump: bool) -> np.ndarray:
    """32-dim hand multi-hot, optionally + 4-dim trump one-hot."""
    dim = HAND_DIM + TRUMP_DIM if has_trump else HAND_DIM
    v = np.zeros(dim, dtype=np.float32)
    for c in hand:
        v[c.id] = 1.0
    if has_trump:
        if trump is None:
            raise ValueError("trump required for trump-aware contract")
        v[TRUMP_OFFSET + _SUIT_IDX[trump]] = 1.0
    return v
