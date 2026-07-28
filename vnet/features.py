"""Contract-parameterised feature extraction for the V-net pipeline.

Pure function over an explicit game-state snapshot. We don't peek into the
opaque solver position — the caller maintains its own Python-side state
during playout (hands, played cards, current trick, voids) and passes it in.

Base layout (132 dims, defender POV — applies to all contracts):

  hand[32]                 my current hand, one-hot
  played[32]               all cards played in completed tricks so far
  current_trick[39]        3 slots × (suit5 + rank8); empty slot = suit[4]=1
  trick_no[1]              trick_no / 10
  sol_voids[4]             one-hot per suit the soloist is known void in
  partner_voids[4]         same for my partner
  role[2]                  is_def1, is_def2 (one-hot for viewer id)
  partner_played[1]        1 if my partner already played in current trick
  sol_played[1]            1 if soloist already played in current trick
  on_lead[1]               1 if I lead the next card
  cards_out_per_suit[4]    (8 - played_in_suit) / 8
  my_top_per_suit[4]       max contract-strength I hold / 7, suit-wise
  live_top_per_suit[4]     max contract-strength still alive / 7
  remaining_tricks[1]      (10 - trick_no) / 10
  partner_hand_size[1]     |partner's hand| / 10
  sol_hand_size[1]         |soloist's hand| / 10

Trump tail (only when ``cfg.has_trump``, +20 dims):

  trump_suit[4]            one-hot for the trump suit
  my_trumps[8]             one-hot over trump-suit ranks I hold
  live_trumps[8]           one-hot over trump-suit ranks still in any hand

The ``cfg.strength_fn`` parameter controls per-suit "top card" summaries —
betli uses ``betli_strength`` (10 weak), other contracts use ``parti_strength``
(standard ulti order, 10 strong).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np

from ulti.card import Card, RANKS, SUITS

from .contracts import ContractConfig


_BASE_DIM = 132


def extract_features(
    *,
    cfg:           ContractConfig,
    hands:         Sequence[Sequence[Card]],     # per-player current hand
    played_cards:  Sequence[Card],               # all cards in completed tricks
    current_trick: Sequence[Tuple[int, Card]],   # (player_id, card) in play order
    leader:        int,                          # leader of the current trick
    trick_no:      int,                          # 0..9
    voids:         Dict[int, Set[str]],          # voids.as_dict() output
    soloist:       int,
    viewer:        int,                          # 0=soloist, 1/2=defenders
    trump:         Optional[str] = None,         # required iff cfg.has_trump
) -> np.ndarray:
    assert viewer in (0, 1, 2), "viewer must be 0 (soloist) or 1/2 (defender)"
    if cfg.has_trump and trump is None:
        raise ValueError(f"contract {cfg.name!r} needs a trump suit")
    if (not cfg.has_trump) and trump is not None:
        raise ValueError(f"contract {cfg.name!r} is trumpless; trump must be None")

    f = np.zeros(cfg.feature_dim, dtype=np.float32)
    o = 0  # cursor

    # ── base 132 dims ───────────────────────────────────────────────────────

    # hand[32]
    for c in hands[viewer]:
        f[o + c.id] = 1.0
    o += 32

    # played[32]
    for c in played_cards:
        f[o + c.id] = 1.0
    o += 32

    # current_trick[39]: 3 slots × (suit5 incl. "empty" + rank8)
    for slot in range(3):
        if slot < len(current_trick):
            _, c = current_trick[slot]
            f[o + 13 * slot + c.suit_index] = 1.0       # suit 0..3
            f[o + 13 * slot + 5 + c.rank_index] = 1.0   # rank 0..7
        else:
            f[o + 13 * slot + 4] = 1.0                  # suit index 4 = empty
    o += 39

    # trick_no[1]
    f[o] = trick_no / 10.0
    o += 1

    # sol_voids[4]
    for s in voids.get(soloist, ()):
        f[o + SUITS.index(s)] = 1.0
    o += 4

    # partner_voids[4]
    # For defenders: voids of the *other defender*. For soloist (no
    # partner): union of both defenders' voids — the natural extension
    # ("any opponent known to be void in suit s").
    if viewer == soloist:
        partner = None
        partner_void_set = set()
        for d in (1, 2):
            partner_void_set.update(voids.get(d, ()))
        partner_iter = partner_void_set
    else:
        partner = 3 - viewer
        partner_iter = voids.get(partner, ())
    for s in partner_iter:
        f[o + SUITS.index(s)] = 1.0
    o += 4

    # role[2] — (1, 0)=def1, (0, 1)=def2, (0, 0)=soloist (no flag set).
    if viewer == 1:
        f[o] = 1.0
    elif viewer == 2:
        f[o + 1] = 1.0
    o += 2

    # partner_played[1], sol_played[1]
    # partner_played for soloist: 1 if *either* defender already played this trick.
    in_trick_pids = {p for p, _ in current_trick}
    if viewer == soloist:
        f[o] = 1.0 if (1 in in_trick_pids or 2 in in_trick_pids) else 0.0
    else:
        f[o] = 1.0 if partner in in_trick_pids else 0.0
    o += 1
    f[o] = 1.0 if soloist in in_trick_pids else 0.0; o += 1

    # on_lead[1]
    if len(current_trick) == 0:
        next_player = leader
    else:
        next_player = (leader + len(current_trick)) % 3
    f[o] = 1.0 if next_player == viewer else 0.0
    o += 1

    # cards_out_per_suit[4]
    played_per_suit = [0, 0, 0, 0]
    for c in played_cards:
        played_per_suit[c.suit_index] += 1
    for s_idx in range(4):
        f[o + s_idx] = (8 - played_per_suit[s_idx]) / 8.0
    o += 4

    # my_top_per_suit[4] — uses contract strength order
    my_top = [0, 0, 0, 0]
    for c in hands[viewer]:
        st = cfg.strength_fn(c)
        if st > my_top[c.suit_index]:
            my_top[c.suit_index] = st
    for s_idx in range(4):
        f[o + s_idx] = my_top[s_idx] / 7.0
    o += 4

    # live_top_per_suit[4] — uses contract strength order
    live_top = [-1, -1, -1, -1]
    for p in range(3):
        for c in hands[p]:
            st = cfg.strength_fn(c)
            if st > live_top[c.suit_index]:
                live_top[c.suit_index] = st
    for s_idx in range(4):
        f[o + s_idx] = max(live_top[s_idx], 0) / 7.0
    o += 4

    # remaining_tricks[1]
    f[o] = (10 - trick_no) / 10.0
    o += 1

    # partner_hand_size[1], sol_hand_size[1]
    # For soloist: "partner_hand_size" = sum of both defender hand sizes
    # (natural opponent-mass; scale unchanged: /20 to match betli's /10 magnitude).
    if viewer == soloist:
        f[o] = (len(hands[1]) + len(hands[2])) / 20.0
    else:
        f[o] = len(hands[partner]) / 10.0
    o += 1
    f[o] = len(hands[soloist])  / 10.0; o += 1

    assert o == _BASE_DIM, (o, _BASE_DIM)

    # ── trump tail (+20 dims, contracts with trump only) ────────────────────

    if cfg.has_trump:
        trump_idx = SUITS.index(trump)

        # trump_suit[4]
        f[o + trump_idx] = 1.0
        o += 4

        # my_trumps[8] — one-hot over trump-suit ranks in viewer's hand
        for c in hands[viewer]:
            if c.suit_index == trump_idx:
                f[o + c.rank_index] = 1.0
        o += 8

        # live_trumps[8] — one-hot over trump-suit ranks still in any hand
        for p in range(3):
            for c in hands[p]:
                if c.suit_index == trump_idx:
                    f[o + c.rank_index] = 1.0
        o += 8

    assert o == cfg.feature_dim, (o, cfg.feature_dim)
    return f
