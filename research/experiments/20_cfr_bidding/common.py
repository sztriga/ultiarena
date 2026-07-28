"""Shared primitives for the CFR bidding experiment.

Defines the standard-auction action set, the symmetric deal, deployable
availability (from own 10 cards), and the two value-model queries we need:

  * ``action_pvals(picker, hand10)``      — raw-10 P(make) per action, for the
                                            *bucket* (deployable belief).
  * ``action_realization(picker, h, t, a)`` — the value-model's best
                                            (suit, discard) from the real 12,
                                            for the *leaf* god-evaluation.

card.id = suit_index*8 + rank_index;  suits = acorns,leaves,hearts,bells.
"""
from __future__ import annotations

import itertools
import random
from typing import Optional

import numpy as np

from ulti.card import SUITS, fresh_deck
from vnet.pickup import featurize, CONTRACT_CONFIGS

# ── action set (escalating rank, mirrors auction_h2h.contract_rank) ──────────
ACTIONS = ['parti', 'ulti', 'betli', 'duri', 'ulti_piros']
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
RANK = {'parti': 2, 'ulti': 3, 'betli': 4, 'duri': 5, 'ulti_piros': 6}
RANK_TO_ACTION = {v: k for k, v in RANK.items()}
PASS = 'pass'

# action → (solver contract name, trump constraint)
NONHEARTS = [s for s in SUITS if s != 'hearts']
_CONTRACT = {'parti': 'parti', 'ulti': 'ulti', 'betli': 'betli',
             'duri': 'durchmars', 'ulti_piros': 'ulti'}
N_ACTIONS = len(ACTIONS)


def deal_10_10_10_2(seed):
    """Standard deal: three 10-card hands + a fixed hidden 2-card talon."""
    rng = random.Random(seed)
    deck = fresh_deck()
    rng.shuffle(deck)
    return deck[:10], deck[10:20], deck[20:30], deck[30:32]


def _has7(hand, suit):
    return any(c.suit == suit and c.rank == '7' for c in hand)


def raw_avail(hand10) -> dict:
    """Which actions a player may declare, decided from its OWN 10 cards
    only (deployable). The agent keeps its trump-7, so an available ulti is
    always playable after talon pickup."""
    return {
        'parti': True,
        'betli': True,
        'duri': True,
        'ulti': any(_has7(hand10, s) for s in NONHEARTS),
        'ulti_piros': _has7(hand10, 'hearts'),
    }


# ── value-model queries ──────────────────────────────────────────────────────

def action_pvals(picker, hand10) -> np.ndarray:
    """Raw-10 P(make) per action (treating the 10 dealt cards as the final
    hand). Used for the deployable bucket. Unavailable → 0.0."""
    av = raw_avail(hand10)
    p = np.zeros(N_ACTIONS, dtype=np.float32)
    # trump contracts on the raw 10
    p[0] = float(picker.predict(featurize(hand10, 'hearts', True)[None], 'parti')[0])
    if av['ulti']:
        best = 0.0
        for s in NONHEARTS:
            if _has7(hand10, s):
                pv = float(picker.predict(featurize(hand10, s, True)[None], 'ulti')[0])
                best = max(best, pv)
        p[1] = best
    p[2] = float(picker.predict(featurize(hand10, None, False)[None], 'betli')[0])
    p[3] = float(picker.predict(featurize(hand10, None, False)[None], 'durchmars')[0])
    if av['ulti_piros']:
        p[4] = float(picker.predict(featurize(hand10, 'hearts', True)[None], 'ulti')[0])
    return p


def _best_discard(picker, hand12, contract, trump, keep_suit7=None):
    """Value-model best (sol10, discard) over the 66 discards of hand12.
    If ``keep_suit7`` is set, only discards that retain that suit's 7 are
    considered (ulti). Returns (sol10, discard, p) or None."""
    cfg = CONTRACT_CONFIGS[contract]
    finals, keeps = [], []
    for dp in itertools.combinations(hand12, 2):
        rem = [c for c in hand12 if c not in dp]
        if keep_suit7 is not None and not _has7(rem, keep_suit7):
            continue
        finals.append(rem)
        keeps.append(dp)
    if not finals:
        return None
    X = np.stack([featurize(h, trump, cfg.has_trump) for h in finals])
    ps = picker.predict(X, contract)
    bi = int(np.argmax(ps))
    return finals[bi], list(keeps[bi]), float(ps[bi])


def action_realization(picker, hand10, talon, action):
    """The value-model's best realization of ``action`` from the real 12
    (hand10 + talon). Returns dict(sol10, discard, trump, contract, p) or
    None if the action is unavailable from the raw 10-card hand."""
    av = raw_avail(hand10)
    if not av[action]:
        return None
    hand12 = list(hand10) + list(talon)
    contract = _CONTRACT[action]

    if action == 'parti':
        r = _best_discard(picker, hand12, 'parti', 'hearts')
        trump = 'hearts'
    elif action == 'betli':
        r = _best_discard(picker, hand12, 'betli', None)
        trump = None
    elif action == 'duri':
        r = _best_discard(picker, hand12, 'durchmars', None)
        trump = None
    elif action == 'ulti_piros':
        r = _best_discard(picker, hand12, 'ulti', 'hearts', keep_suit7='hearts')
        trump = 'hearts'
    elif action == 'ulti':
        best = None
        trump = None
        for s in NONHEARTS:
            if not _has7(hand10, s):
                continue
            rr = _best_discard(picker, hand12, 'ulti', s, keep_suit7=s)
            if rr is not None and (best is None or rr[2] > best[2]):
                best = rr
                trump = s
        r = best
    else:
        raise ValueError(action)

    if r is None:
        return None
    sol10, discard, p = r
    return {'sol10': sol10, 'discard': discard, 'trump': trump,
            'contract': contract, 'p': p}
