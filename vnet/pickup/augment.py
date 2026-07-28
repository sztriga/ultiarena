"""Suit-permutation augmentation for pickup-net training.

Same idea as ``vnet.augment``, but for the simpler pickup feature
layout (hand[32] + optional trump[4]). Trumpless contracts (betli,
durchmars) get σ ∈ S₄ → 24× effective data. Trump-aware contracts
(parti, ulti) keep the trump suit pinned and shuffle the other three
→ 6× effective data.
"""
from __future__ import annotations

import numpy as np

from .features import HAND_DIM, HAND_OFFSET, TRUMP_DIM, TRUMP_OFFSET


def random_suit_perm(
    has_trump: bool, x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample a length-4 permutation σ with ``σ[old] = new``.

    Trump contracts: σ fixes the trump suit (read from x's own trump
    one-hot at offset 32).
    """
    if has_trump:
        trump_idx = int(np.argmax(x[TRUMP_OFFSET:TRUMP_OFFSET + TRUMP_DIM]))
        targets = [s for s in range(4) if s != trump_idx]
        rng.shuffle(targets)
        sigma = np.empty(4, dtype=np.int64)
        sigma[trump_idx] = trump_idx
        i = 0
        for old in range(4):
            if old == trump_idx:
                continue
            sigma[old] = targets[i]
            i += 1
        return sigma
    sigma = np.arange(4, dtype=np.int64)
    rng.shuffle(sigma)
    return sigma


def apply_suit_perm(
    x: np.ndarray, sigma: np.ndarray, has_trump: bool
) -> np.ndarray:
    """Return a copy of ``x`` with suit labels relabeled by σ."""
    out = x.copy()
    inv = np.argsort(sigma)   # inv[new] = old

    # hand[32], reshape as (suit, rank) and gather
    block = x[HAND_OFFSET:HAND_OFFSET + HAND_DIM].reshape(4, 8)
    out[HAND_OFFSET:HAND_OFFSET + HAND_DIM] = block[inv].reshape(-1)

    # trump one-hot — moves under σ; lands back on same index since σ
    # fixes the trump, but we apply it for layout consistency.
    if has_trump:
        out[TRUMP_OFFSET:TRUMP_OFFSET + TRUMP_DIM] = \
            x[TRUMP_OFFSET:TRUMP_OFFSET + TRUMP_DIM][inv]

    return out
