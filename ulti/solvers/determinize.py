"""Information-set construction and sampling for PIMC.

Pure module — no solver / no MCTS. Splits the work so that ``solvers.pimc``
becomes a thin loop around two contract-agnostic primitives:

    iset = build_info_set(pos, viewer, contract, voids=..., must_hold=...)
    hands, talon = sample_world(iset, rng)

The same machinery handles betli / ulti / 40-100 / parti / durchmars; the
only per-contract knowledge lives in :data:`_AUCTION_RULES` (cards a
specific player must hold given the auction history).

Information-set rules currently encoded
---------------------------------------
* **Talon visibility.** Soloist sees the talon (the 2 cards set aside after
  the bid+discard step); defenders do not. Implemented by either pinning
  the talon to its known cards (soloist viewer) or treating it as 2 extra
  unknown slots to fill from the pool (defender viewer).
* **Known voids.** If a player failed to follow suit on a past trick they
  are void in that suit forever. Caller passes a ``voids`` map; sampler
  refuses to deal cards of that suit to that player whenever possible.
* **Must-hold.** Cards a specific player is provably holding given the
  auction (e.g. ulti contract → soloist holds trump 7). Empty for betli.

What's intentionally still loose
--------------------------------
* No upper-bound inference on must-hold (e.g. "player X cannot hold > k
  trumps"). That would warrant a constraint solver.
* No correlation between voids and exact counts (we honour each void
  independently, not joint).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from ulti.card import Card

from . import pis as _pis


# Rank-strength used by the betli dealer (eval.dojo._BETLI_STRENGTH). Mirrored
# here so the determinizer's bid-conditional prior matches the dealer's
# distribution without a runtime import from ulti.eval/.
_BETLI_STRENGTH = {'7': 0, '8': 1, '9': 2, '10': 3,
                   'lower': 4, 'upper': 5, 'king': 6, 'ace': 7}


def _weighted_sample_no_replace(items, weights, k, rng):
    """Efraimidis-Spirakis weighted sampling without replacement. Returns
    (picked, leftover). Same algorithm the dojo uses for dealing."""
    keys = [(rng.random() ** (1.0 / max(w, 1e-9)), i) for i, w in enumerate(weights)]
    keys.sort(reverse=True)
    chosen_idx    = [i for _, i in keys[:k]]
    remaining_idx = [i for _, i in keys[k:]]
    return [items[i] for i in chosen_idx], [items[i] for i in remaining_idx]


# ---------------------------------------------------------------------------
# Suit string conversion (trickster enum → oldtawer name)
# ---------------------------------------------------------------------------

def _suit_name(s) -> str:
    """Return the oldtawer-style suit string for either an oldtawer ``str``
    or a trickster ``Suit`` enum value (which is itself a ``StrEnum`` —
    careful, ``isinstance(s, str)`` is True even for the enum)."""
    if s in _pis._TSUIT_TO_OSUIT:
        return _pis._TSUIT_TO_OSUIT[s]
    return s  # oldtawer plain string suit


# ---------------------------------------------------------------------------
# InfoSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InfoSet:
    """Everything the determinizer needs to produce a sampled world."""

    viewer:      int
    soloist:     int
    own_hand:    List[Card]
    hidden_pool: List[Card]
    hand_sizes:  Dict[int, int]              # player_id → size, only for unknown players
    talon_size:  int                         # cards to sample into the talon slot
    talon_known: Optional[List[Card]]        # talon cards if known to viewer (else None)
    voids:       Dict[int, FrozenSet[str]]   # known voids per *unknown* player (suit strings)
    must_hold:   Dict[int, List[Card]]       # cards forced into a specific player's hand


# ---------------------------------------------------------------------------
# Contract-specific must-hold hooks (registered, not branched)
# ---------------------------------------------------------------------------

def _no_must_hold(pos) -> Dict[int, List[Card]]:
    return {}


# Stubs for future contracts. Port the trickster ``build_auction_constraints``
# logic here when you wire up ulti / 40-100 benches.
def _ulti_must_hold(pos) -> Dict[int, List[Card]]:
    """Ulti auction rule: the soloist must hold the trump 7 (and is
    barred from discarding it to the talon). Plant it on the soloist's
    slot in every sampled world so PIMC defenders don't waste samples
    planning for impossible worlds where def2 secretly held it.

    Edge case: if the viewer is a defender who somehow sees the trump-7
    in their own hand (e.g., a hand-built position that violates the
    auction), the determinizer's must-hold loop will silently skip the
    constraint (``pool.remove(c)`` raises ValueError → ``continue``).
    PIMC then averages over worlds where sol can't have trump-7, the
    solver returns LOSE for every move, and MIN over flat zeros leaves
    the defender's pick arbitrary — exactly the right inference
    (silent ulti is impossible, play any move)."""
    from .pis import _TSUIT_TO_OSUIT
    if pos.trump is None:
        return {}
    trump_str = _TSUIT_TO_OSUIT[pos.trump]
    trump_7 = Card(suit=trump_str, rank='7')
    return {pos.soloist: [trump_7]}


def _forty_must_hold(pos) -> Dict[int, List[Card]]:
    return {}      # TODO: trump K+Q for soloist when bid includes 40-100


_AUCTION_RULES: Dict[str, Callable[[object], Dict[int, List[Card]]]] = {
    "betli":      _no_must_hold,
    "ulti":       _ulti_must_hold,
    "40-100":     _forty_must_hold,
    "parti":      _no_must_hold,
    "durchmars":  _no_must_hold,
    # EV_MULTI: caller picks the auction rule via the active weight set.
    # For v1 (parti + silent ulti) the auction is parti's → no must-hold.
    "multi":      _no_must_hold,
}


# ---------------------------------------------------------------------------
# Building the info set
# ---------------------------------------------------------------------------

def build_info_set(
    pos,
    viewer:    int,
    contract:  str,
    *,
    voids:     Optional[Dict[int, FrozenSet[str]]] = None,
    must_hold: Optional[Dict[int, List[Card]]] = None,
) -> InfoSet:
    """Construct the info set for player ``viewer`` at ``pos``.

    ``voids`` should be the play-history voids tracked by the caller (see
    :class:`Voids`). If omitted, no voids are enforced (current behaviour).

    ``must_hold`` overrides the contract's default auction-derived must-hold
    map. Pass an empty dict to disable, ``None`` to use the registered rule.
    """
    if contract not in _AUCTION_RULES:
        raise ValueError(f"unknown contract {contract!r}")

    soloist  = pos.soloist
    hands_o  = _pis.hands_by_player(pos)
    own_hand = list(hands_o[viewer])

    other_ids = [p for p in range(3) if p != viewer]

    # Talon
    talon_cards = [_pis._to_o(c) for c in (pos.talon_discards or [])]
    if viewer == soloist:
        talon_known: Optional[List[Card]] = talon_cards
        talon_size = 0
    else:
        talon_known = None
        talon_size = len(talon_cards)

    # Hidden pool
    hidden_pool: List[Card] = []
    hand_sizes: Dict[int, int] = {}
    for p in other_ids:
        hand_sizes[p] = len(hands_o[p])
        hidden_pool.extend(hands_o[p])
    if talon_known is None:
        hidden_pool.extend(talon_cards)

    voids_use: Dict[int, FrozenSet[str]] = {}
    if voids is not None:
        for p in other_ids:
            voids_use[p] = voids.get(p, frozenset())
    else:
        for p in other_ids:
            voids_use[p] = frozenset()

    if must_hold is None:
        must_hold_full = _AUCTION_RULES[contract](pos)
    else:
        must_hold_full = must_hold
    must_hold_others = {p: list(must_hold_full.get(p, [])) for p in other_ids}

    return InfoSet(
        viewer=viewer,
        soloist=soloist,
        own_hand=own_hand,
        hidden_pool=hidden_pool,
        hand_sizes=hand_sizes,
        talon_size=talon_size,
        talon_known=talon_known,
        voids=voids_use,
        must_hold=must_hold_others,
    )


# ---------------------------------------------------------------------------
# Sampling a determinization
# ---------------------------------------------------------------------------

def sample_world(
    info_set: InfoSet,
    rng:      random.Random,
    *,
    sol_prior_alpha: Optional[float] = None,
) -> Tuple[List[List[Card]], List[Card]]:
    """Sample one concrete world consistent with ``info_set``.

    Returns ``(hands, talon)`` where ``hands[viewer]`` is the viewer's own
    hand verbatim; ``hands[p]`` for the other two players is a sampled
    hand of the right size respecting voids and must-holds; ``talon``
    is either the known talon or a sampled list of length ``talon_size``.

    ``sol_prior_alpha`` enables a bid-conditional prior on the soloist's
    hand (defender viewer only): the soloist's slots are filled by
    weighted sampling-without-replacement with per-card weight
    ``exp(-alpha * BETLI_STRENGTH(card))`` — same Boltzmann form the
    dealer uses, so passing the same alpha as the dealer recovers (a
    close approximation of) the dealer's distribution. No effect when
    the viewer is the soloist.
    """
    pool = list(info_set.hidden_pool)
    rng.shuffle(pool)

    hands: Dict[int, List[Card]] = {info_set.viewer: list(info_set.own_hand)}
    for p in info_set.hand_sizes:
        hands[p] = []

    # 1. Plant must-hold cards first (no choice involved).
    for p, cards in info_set.must_hold.items():
        for c in cards:
            try:
                pool.remove(c)
            except ValueError:
                # must-hold card already in someone's visible hand or talon — skip.
                continue
            hands[p].append(c)

    # 1b. Bid-conditional prior on soloist's hand (defender viewer).
    sol = info_set.soloist
    if (sol_prior_alpha is not None
        and info_set.viewer != sol
        and sol in info_set.hand_sizes):
        sol_voids = info_set.voids[sol]
        need = info_set.hand_sizes[sol] - len(hands[sol])
        eligible:   List[Card] = []
        ineligible: List[Card] = []
        for c in pool:
            if c.suit in sol_voids:
                ineligible.append(c)
            else:
                eligible.append(c)
        if len(eligible) >= need:
            weights = [math.exp(-sol_prior_alpha * _BETLI_STRENGTH[c.rank])
                       for c in eligible]
            picked, leftover = _weighted_sample_no_replace(
                eligible, weights, need, rng,
            )
            hands[sol].extend(picked)
            pool = leftover + ineligible
            rng.shuffle(pool)
        # else: not enough eligible cards — fall through to the uniform
        # void-aware fallback below (very rare; cardinality forcing).

    # 2. Fill void-constrained players first — they have a smaller eligible
    #    pool, so giving them priority prevents the greedy fallback from
    #    forcing voids violations when the unconstrained slots aren't yet
    #    consuming options.
    for p, sz in info_set.hand_sizes.items():
        voids = info_set.voids[p]
        if not voids:
            continue
        need = sz - len(hands[p])
        picked: List[Card] = []
        leftover: List[Card] = []
        for c in pool:
            if len(picked) < need and c.suit not in voids:
                picked.append(c)
            else:
                leftover.append(c)
        if len(picked) < need:
            # Forced: pool didn't have enough eligible cards. Fall back to
            # whatever remains (rare; cardinality-forcing).
            extra = need - len(picked)
            picked.extend(leftover[:extra])
            leftover = leftover[extra:]
        hands[p].extend(picked)
        pool = leftover

    # 3. Fill the remaining slots (unconstrained players + talon) from
    #    what's left. Pool is still uniformly random over its remainder.
    talon: List[Card] = []
    for p, sz in info_set.hand_sizes.items():
        if info_set.voids[p]:
            continue
        need = sz - len(hands[p])
        hands[p].extend(pool[:need])
        pool = pool[need:]
    for _ in range(info_set.talon_size):
        talon.append(pool.pop())

    if info_set.talon_known is not None:
        assert info_set.talon_size == 0
        talon = list(info_set.talon_known)

    hands_list = [hands[p] for p in range(3)]
    return hands_list, talon


# ---------------------------------------------------------------------------
# Voids tracker — caller-managed
# ---------------------------------------------------------------------------

class Voids:
    """Tracks known voids per player from observed play.

    Usage::

        v = Voids()
        for each move in play order:
            v.observe(pos_before_move, player, card)
            pis.apply_move(pos, card)
        ...
        pimc_decision(..., voids=v.as_dict())

    Looks at ``pos.trick_cards`` before the move to decide lead vs follow,
    and the trump (``pos.trump``) to also infer trump-side voids per
    Hungarian rules (must-overtrump when out of led suit).
    """

    def __init__(self) -> None:
        self._v: Dict[int, set] = {0: set(), 1: set(), 2: set()}

    def observe(self, pos_before_move, player: int, card: Card) -> None:
        # Leading a trick: no information is leaked.
        if not pos_before_move.trick_cards:
            return

        led_suit = _suit_name(pos_before_move.trick_cards[0][1].suit)
        played_suit = card.suit
        if played_suit == led_suit:
            return  # followed suit — no void revealed

        # Did not follow suit → void in led suit.
        self._v[player].add(led_suit)

        # Hungarian rules: when out of led suit, you must overtrump if you
        # can. If trump exists and the player still didn't play trump, they
        # are also out of trumps (provided no one already trumped above
        # what they could beat — which the rule covers by the cull). We
        # encode the simple, sound case: no trump played at all by them
        # while they had to → void in trump too.
        trump = getattr(pos_before_move, "trump", None)
        if trump is not None:
            trump_name = _suit_name(trump)
            if played_suit != trump_name:
                # Player skipped trump while off-suit → void in trump.
                self._v[player].add(trump_name)

    def as_dict(self) -> Dict[int, FrozenSet[str]]:
        return {p: frozenset(s) for p, s in self._v.items()}
