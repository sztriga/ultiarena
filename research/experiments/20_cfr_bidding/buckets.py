"""Deployable hand bucket: discretize the value-model's raw-10 P(make).

The bucket is the only private information in an infoset key. It must be
computable from the agent's own 10 cards alone (no talon, no opponents), so
the same function indexes the strategy in training and at deployment.
"""
from __future__ import annotations

import bisect

import numpy as np

from common import ACTIONS, action_pvals

# Per-action bin edges (low / med / high). Coarse on purpose — finer buckets
# need more deals per (bucket, history) to estimate regrets.
EDGES = {
    'parti':      [0.30, 0.55],
    'ulti':       [0.40, 0.70],
    'betli':      [0.20, 0.40],
    'duri':       [0.30],
    'ulti_piros': [0.40, 0.70],
}
_BASES = [len(EDGES[a]) + 1 for a in ACTIONS]   # bins per action


def levels(pvals: np.ndarray):
    return tuple(bisect.bisect_right(EDGES[a], float(pvals[i]))
                 for i, a in enumerate(ACTIONS))


def encode(lv) -> int:
    b = 0
    for level, base in zip(lv, _BASES):
        b = b * base + level
    return b


def to_bucket(pvals: np.ndarray) -> int:
    return encode(levels(pvals))


def bucket_of_hand(picker, hand10) -> int:
    return to_bucket(action_pvals(picker, hand10))


N_BUCKETS = 1
for _b in _BASES:
    N_BUCKETS *= _b
