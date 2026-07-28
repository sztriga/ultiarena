"""Bidding agents for the tournament. Both see only (own 10 cards, public bid
history) at each decision — no talon, no opponent hands.

CompositeAgent — the current bidder, made deployable: value-model raw-10
    action EVs, greedy argmax with a fixed per-contract make-prob prior `Q`
    for the pass/overtake decision. History-BLIND (no belief updating).

CFRAgent — looks up the learned average strategy at (bucket, avail, history)
    and samples. Falls back to the composite rule on unseen infosets.
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))
sys.path.insert(0, str(Path(__file__).parent))   # local modules win over repo pkgs

from _lib import _ev_per_def                              # noqa: E402

import game as G                                          # noqa: E402
from common import (ACTIONS, ACTION_INDEX, PASS, action_pvals,  # noqa: E402
                    raw_avail)
from buckets import to_bucket                             # noqa: E402

# action → (solver contract, piros) for EV conversion
_EVSPEC = {'parti': ('parti', True), 'ulti': ('ulti', False),
           'betli': ('betli', False), 'duri': ('durchmars', False),
           'ulti_piros': ('ulti', True)}
# heuristic prior: P(holder makes its contract | it was bid). Used only for
# the composite's deployable pass/defend value (it can't see the holder hand).
Q_DEFAULT = {'parti': 0.50, 'ulti': 0.60, 'betli': 0.50,
             'duri': 0.50, 'ulti_piros': 0.60}
OPEN_FLOOR = -2.0   # P0 bids at open iff best EV/def > -2 (else pass-out)


def action_ev(action, p):
    contract, piros = _EVSPEC[action]
    return _ev_per_def(contract, piros, p)


class CompositeAgent:
    def __init__(self, picker, Q=None):
        self.picker = picker
        self.Q = Q or Q_DEFAULT
        self._cache = {}

    def _evs(self, hand10):
        key = tuple(sorted(c.id for c in hand10))
        ev = self._cache.get(key)
        if ev is None:
            p = action_pvals(self.picker, hand10)
            ev = {a: action_ev(a, float(p[ACTION_INDEX[a]])) for a in ACTIONS}
            self._cache[key] = ev
        return ev

    def act(self, *, hand10, history, legal_actions, rng=None):
        evs = self._evs(hand10)
        cands = [a for a in legal_actions if a != PASS]
        if not cands:
            return PASS
        best = max(cands, key=lambda a: evs[a])
        if len(history) == 0:                       # open
            return best if evs[best] > OPEN_FLOOR else PASS
        _, level = G._holder_level(history)
        holder_action = G.RANK_TO_ACTION[level]
        pass_ev = -action_ev(holder_action, self.Q[holder_action])
        return best if evs[best] > pass_ev else PASS


# ── faithful "oracle" composite: the actual deployed bidding logic ───────────
# Mirrors auction_h2h._oracle_evaluate: for each contract, mean over sampled
# talons of (MAX over the 66 discards of the value-model p). The max-over-66 is
# the optimizer's-curse / betli-inflation source we want CFR to be tested
# against. Vectorized featurizer keeps it tractable.
from ulti.card import SUITS                                  # noqa: E402

_PAIRS = np.array([(a, b) for a in range(12) for b in range(a + 1, 12)])  # (66,2)
_ORACLE_CACHE = {}


def _disc_rows(hand12_ids, keep7_id):
    """All 66 (or 55, if keep7) discard-10 multi-hots of a 12-card hand."""
    ids = np.asarray(hand12_ids)
    p = _PAIRS
    if keep7_id is not None:
        p = p[(ids[_PAIRS[:, 0]] != keep7_id) & (ids[_PAIRS[:, 1]] != keep7_id)]
    nd = len(p)
    X = np.zeros((nd, 32), np.float32)
    X[:, ids] = 1.0
    ar = np.arange(nd)
    X[ar, ids[p[:, 0]]] = 0.0
    X[ar, ids[p[:, 1]]] = 0.0
    return X


def _talon_avg_ev(picker, hand_ids, talons, contract, trump, has_trump, keep7):
    blocks = [_disc_rows(hand_ids + list(t), keep7) for t in talons]
    nd = blocks[0].shape[0]
    X = np.concatenate(blocks, 0)
    if has_trump:
        oh = np.zeros((X.shape[0], 4), np.float32)
        oh[:, SUITS.index(trump)] = 1.0
        X = np.concatenate([X, oh], 1)
    p = picker.predict(X, contract).reshape(len(talons), nd)
    mean_p = float(p.max(1).mean())          # max over discards, mean over talons
    return _ev_per_def(contract, trump == 'hearts', mean_p)


def oracle_action_evs(picker, hand10, K, rng):
    hand_ids = [c.id for c in hand10]
    av = raw_avail(hand10)
    unseen = [i for i in range(32) if i not in set(hand_ids)]
    talons = [rng.sample(unseen, 2) for _ in range(K)]
    ev = {}
    ev['parti'] = _talon_avg_ev(picker, hand_ids, talons, 'parti', 'hearts', True, None)
    ev['betli'] = _talon_avg_ev(picker, hand_ids, talons, 'betli', None, False, None)
    ev['duri'] = _talon_avg_ev(picker, hand_ids, talons, 'durchmars', None, False, None)
    ev['ulti_piros'] = (_talon_avg_ev(picker, hand_ids, talons, 'ulti', 'hearts',
                                      True, SUITS.index('hearts') * 8)
                        if av['ulti_piros'] else action_ev('ulti_piros', 0.0))
    if av['ulti']:
        best = -1e9
        for s in SUITS:
            if s == 'hearts':
                continue
            if any(c.suit == s and c.rank == '7' for c in hand10):
                best = max(best, _talon_avg_ev(picker, hand_ids, talons, 'ulti',
                                               s, True, SUITS.index(s) * 8))
        ev['ulti'] = best
    else:
        ev['ulti'] = action_ev('ulti', 0.0)
    return ev


class OracleComposite(CompositeAgent):
    """CompositeAgent with the deployed talon-averaged argmax-over-66 eval."""
    def __init__(self, picker, Q=None, K=24):
        super().__init__(picker, Q)
        self.K = K

    def _evs(self, hand10):
        key = tuple(sorted(c.id for c in hand10))
        ev = _ORACLE_CACHE.get(key)
        if ev is None:
            seedrng = random.Random(hash(key) & 0xFFFFFFFF)
            ev = oracle_action_evs(self.picker, hand10, self.K, seedrng)
            _ORACLE_CACHE[key] = ev
        return ev


class CFRAgent:
    def __init__(self, picker, avg, fallback):
        self.picker = picker
        self.avg = avg
        self.fallback = fallback   # CompositeAgent for unseen infosets
        self._bkcache = {}

    def _bucket_avail(self, hand10):
        key = tuple(sorted(c.id for c in hand10))
        v = self._bkcache.get(key)
        if v is None:
            pv = action_pvals(self.picker, hand10)
            av = raw_avail(hand10)
            bucket = to_bucket(pv)
            v = (bucket, (bool(av['ulti']), bool(av['ulti_piros'])))
            self._bkcache[key] = v
        return v

    def act(self, *, hand10, history, legal_actions, rng=None):
        bucket, avbits = self._bucket_avail(hand10)
        key = (bucket, avbits, history)
        dist = self.avg.get(key)
        if not dist:
            return self.fallback.act(hand10=hand10, history=history,
                                     legal_actions=legal_actions, rng=rng)
        # restrict to legal (should already match) and renormalize
        items = [(a, p) for a, p in dist.items() if a in legal_actions]
        if not items:
            return self.fallback.act(hand10=hand10, history=history,
                                     legal_actions=legal_actions, rng=rng)
        tot = sum(p for _, p in items)
        rng = rng or random
        r = rng.random() * tot
        c = 0.0
        for a, p in items:
            c += p
            if r <= c:
                return a
        return items[-1][0]

    @classmethod
    def load(cls, picker, strategy_path, fallback):
        with open(strategy_path, 'rb') as f:
            d = pickle.load(f)
        return cls(picker, d['avg'], fallback)
