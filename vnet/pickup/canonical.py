"""Canonical (suit-permutation-invariant) pickup features.

The pickup net's target P(make) is invariant under relabeling suits, as
long as the trump label moves with the permutation — nothing in trick
play distinguishes hearts; the piros ×2 is applied to GP outside the
net. Instead of teaching the net this symmetry via augmentation, map
every hand to a canonical representative:

  trump contracts:   trump suit row first, then the other 3 suit rows
                     sorted by descending rank-pattern key
  trumpless:         all 4 suit rows sorted by descending key

Output is always 32-dim (4×8 suit-major, like ``features.featurize``)
— the trump one-hot becomes redundant and is dropped. Tied keys mean
identical rows, so the canonical form is unique either way.
"""
from __future__ import annotations

import numpy as np

from .features import HAND_DIM, TRUMP_DIM, TRUMP_OFFSET

CANON_DIM = HAND_DIM   # 32

_BIT_WEIGHTS = (1 << np.arange(8)).astype(np.int64)
_TRUMP_BONUS = 1 << 8   # > any 8-bit rank pattern → trump row sorts first


def canonicalize(X: np.ndarray, has_trump: bool) -> np.ndarray:
    """Map a feature batch to canonical 32-dim form.

    X: (N, 36) for trump contracts (hand + trump one-hot), or
       (N, 32) / (N, 36 zero-padded) for trumpless ones.
    """
    if X.ndim == 1:
        return canonicalize(X[None, :], has_trump)[0]
    n = X.shape[0]
    hands = X[:, :HAND_DIM].reshape(n, 4, 8)
    keys = (hands.astype(np.int64) * _BIT_WEIGHTS).sum(axis=2)

    if has_trump:
        if X.shape[1] != HAND_DIM + TRUMP_DIM:
            raise ValueError("trump contract needs 36-dim input")
        trump_idx = X[:, TRUMP_OFFSET:TRUMP_OFFSET + TRUMP_DIM].argmax(axis=1)
        keys[np.arange(n), trump_idx] += _TRUMP_BONUS

    order = np.argsort(-keys, axis=1, kind='stable')
    rows = np.arange(n)[:, None]
    return np.ascontiguousarray(
        hands[rows, order].reshape(n, CANON_DIM)).astype(np.float32)
