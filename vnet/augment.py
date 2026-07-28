"""Suit-permutation augmentation for V-net training.

The four suits in our 32-card deck are *interchangeable* for trumpless
contracts (betli — all 24 permutations preserve the label) and the
three non-trump suits are interchangeable for trump contracts (parti —
the 6 permutations that fix the trump suit). Our features encode suits
positionally, so relabeling them yields a different feature vector with
the same label — i.e. free training data.

This module applies a random suit permutation σ to a single feature row
produced by :func:`vnet.features.extract_features`. Training data
loaders draw a fresh σ per row per epoch; over enough epochs the V-net
sees every member of each row's symmetry orbit.

Effective per-row multiplier:
    betli (σ ∈ S₄):                       24×
    parti (σ ∈ S₃ on the non-trump suits)  6×

Layout assumptions
------------------
Mirrors ``vnet.features`` exactly. If that layout ever moves, update
the ``_OFF_*`` constants here too.
"""
from __future__ import annotations

import numpy as np


# Offsets into the 132-dim base feature vector (mirrors vnet.features).
_OFF_HAND       = 0    # 32 dims, card_id = suit*8 + rank
_OFF_PLAYED     = 32   # 32 dims, same layout
_OFF_TRICK      = 64   # 39 dims: 3 slots × 13 (suit[4] + empty[1] + rank[8])
_OFF_SOL_VOID   = 104  # 4 dims
_OFF_PAR_VOID   = 108  # 4 dims
_OFF_CARDS_OUT  = 117  # 4 dims
_OFF_MY_TOP     = 121  # 4 dims
_OFF_LIVE_TOP   = 125  # 4 dims
# Trump tail (parti only):
_OFF_TRUMP_SUIT = 132  # 4 dims
# my_trumps[8]  @ 136 and live_trumps[8] @ 144 are *rank*-indexed within the
# (fixed) trump suit and so invariant under any σ that fixes the trump.


def random_suit_perm(
    has_trump: bool, x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample a length-4 permutation σ with ``σ[old] = new``.

    For trump contracts σ fixes the trump suit (read from the row's own
    ``trump_suit[4]`` one-hot at offset 132).
    """
    if has_trump:
        trump_idx = int(np.argmax(x[_OFF_TRUMP_SUIT:_OFF_TRUMP_SUIT + 4]))
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
    """Return a copy of ``x`` with suit labels relabeled by σ.

    σ has the semantics ``σ[old] = new`` — i.e. the entry currently
    labeled with suit ``old`` ends up at suit ``new`` after relabeling.
    """
    out = x.copy()
    inv = np.argsort(sigma)   # inv[new] = old

    # 32-card slices: hand, played. Reshape as (suit, rank) and gather.
    for off in (_OFF_HAND, _OFF_PLAYED):
        block = x[off:off + 32].reshape(4, 8)
        out[off:off + 32] = block[inv].reshape(-1)

    # current_trick: 3 slots × 13 dims; only the first 4 (suit one-hots)
    # of each slot move under σ. The empty flag (offset +4) and rank
    # one-hots (offsets +5..+12) are invariant.
    for slot in range(3):
        base = _OFF_TRICK + 13 * slot
        out[base:base + 4] = x[base:base + 4][inv]

    # 4-dim per-suit summaries.
    for off in (_OFF_SOL_VOID, _OFF_PAR_VOID,
                _OFF_CARDS_OUT, _OFF_MY_TOP, _OFF_LIVE_TOP):
        out[off:off + 4] = x[off:off + 4][inv]

    # Trump tail. trump_suit[4] moves under σ (and lands back on the same
    # index because σ fixes the trump). my_trumps[8] / live_trumps[8] are
    # rank-indexed within the trump suit and stay put.
    if has_trump:
        out[_OFF_TRUMP_SUIT:_OFF_TRUMP_SUIT + 4] = \
            x[_OFF_TRUMP_SUIT:_OFF_TRUMP_SUIT + 4][inv]

    return out
