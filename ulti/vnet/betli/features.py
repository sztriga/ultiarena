"""Feature extraction for the betli defender value-net.

Pure function over an explicit game-state snapshot. We don't peek into the
opaque solver position — the caller maintains its own Python-side state
during playout (hands, played cards, current trick, voids) and passes it in.

Output vector layout (132 dims total, defender POV):

  hand[32]                 my current hand, one-hot
  played[32]               all cards played in completed tricks so far
  current_trick[39]        3 slots × (suit5 + rank8); empty slot = suit[4]=1
  trick_no[1]              trick_no / 10
  sol_voids[4]             one-hot per suit the soloist is known void in
  partner_voids[4]         same for my partner
  role[2]                  is_def1, is_def2 (one-hot for viewer id)
  partner_played[1]        1 if my partner already played in current trick
  sol_played[1]            1 if soloist already played in current trick
  on_lead[1]               1 if I lead the *next* card (current trick empty AND
                           current leader == me, or current trick non-empty AND
                           I haven't played yet AND I'm the next to act)
  cards_out_per_suit[4]    (8 - played_in_suit) / 8 — how many cards of each
                           suit are still in someone's hand
  my_top_per_suit[4]       max rank_index I hold / 7, suit-wise; 0 if void
  live_top_per_suit[4]     max rank_index still alive / 7; 0 if suit exhausted
  remaining_tricks[1]      (10 - trick_no) / 10
  partner_hand_size[1]     |partner's hand| / 10
  sol_hand_size[1]         |soloist's hand| / 10
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ulti.card import Card, SUITS, RANKS  # noqa: F401  (SUITS used implicitly)

# Betli trick-taking strength order, NOT the ulti rank_index. In betli the
# 10 is weak (sits between 9 and lower/J), not between K and A like in
# regular ulti. Using rank_index for "top in suit" summaries gives the
# 10 a higher score than the K, which is wrong for betli.
# Mirrors ultisolver._solver_core._str_b and eval.dojo._BETLI_STRENGTH.
from ulti.card import COLORLESS_RANK as _BETLI_STRENGTH


def _betli_strength(c: Card) -> int:
    return _BETLI_STRENGTH[c.rank]


FEATURE_DIM = 32 + 32 + 39 + 1 + 4 + 4 + 2 + 1 + 1 + 1 + 4 + 4 + 4 + 1 + 1 + 1


def _suit_index(c: Card) -> int:
    return c.suit_index


def _rank_index(c: Card) -> int:
    return c.rank_index


def extract_features(
    *,
    hands:         Sequence[Sequence[Card]],     # per-player current hand
    played_cards:  Sequence[Card],               # all cards in completed tricks
    current_trick: Sequence[Tuple[int, Card]],   # (player_id, card) in play order
    leader:        int,                          # leader of the current trick
    trick_no:      int,                          # 0..9
    voids:         Dict[int, Set[str]],          # voids.as_dict() output
    soloist:       int,
    viewer:        int,                          # must be a defender (1 or 2)
) -> np.ndarray:
    assert viewer in (1, 2), "value net is defender-only"
    f = np.zeros(FEATURE_DIM, dtype=np.float32)
    o = 0  # cursor

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
            f[o + 13 * slot + _suit_index(c)] = 1.0       # suit 0..3
            f[o + 13 * slot + 5 + _rank_index(c)] = 1.0   # rank 0..7
        else:
            f[o + 13 * slot + 4] = 1.0                    # suit index 4 = empty
    o += 39

    # trick_no[1]
    f[o] = trick_no / 10.0
    o += 1

    # sol_voids[4]
    for s in voids.get(soloist, ()):
        f[o + SUITS.index(s)] = 1.0
    o += 4

    # partner_voids[4]
    partner = 3 - viewer  # since viewer ∈ {1,2}, partner ∈ {2,1}
    for s in voids.get(partner, ()):
        f[o + SUITS.index(s)] = 1.0
    o += 4

    # role[2]
    f[o + (0 if viewer == 1 else 1)] = 1.0
    o += 2

    # partner_played[1], sol_played[1]
    in_trick_pids = {p for p, _ in current_trick}
    f[o] = 1.0 if partner in in_trick_pids else 0.0; o += 1
    f[o] = 1.0 if soloist in in_trick_pids else 0.0; o += 1

    # on_lead[1] — viewer leads the next action of the trick.
    # if current trick empty: leader plays next ⇒ on_lead = (leader == viewer)
    # else: next player = (leader + len(current_trick)) % 3
    if len(current_trick) == 0:
        next_player = leader
    else:
        next_player = (leader + len(current_trick)) % 3
    f[o] = 1.0 if next_player == viewer else 0.0
    o += 1

    # cards_out_per_suit[4]
    played_per_suit = [0, 0, 0, 0]
    for c in played_cards:
        played_per_suit[_suit_index(c)] += 1
    for s_idx in range(4):
        f[o + s_idx] = (8 - played_per_suit[s_idx]) / 8.0
    o += 4

    # my_top_per_suit[4]  (highest *betli-strength* I hold per suit, 0 if void)
    my_top = [0, 0, 0, 0]
    for c in hands[viewer]:
        st = _betli_strength(c)
        if st > my_top[_suit_index(c)]:
            my_top[_suit_index(c)] = st
    for s_idx in range(4):
        f[o + s_idx] = my_top[s_idx] / 7.0
    o += 4

    # live_top_per_suit[4]  (highest betli-strength still in any hand per suit)
    live_top = [-1, -1, -1, -1]
    for p in range(3):
        for c in hands[p]:
            st = _betli_strength(c)
            if st > live_top[_suit_index(c)]:
                live_top[_suit_index(c)] = st
    for s_idx in range(4):
        f[o + s_idx] = max(live_top[s_idx], 0) / 7.0
    o += 4

    # remaining_tricks[1]
    f[o] = (10 - trick_no) / 10.0
    o += 1

    # partner_hand_size[1], sol_hand_size[1]
    f[o] = len(hands[partner]) / 10.0; o += 1
    f[o] = len(hands[soloist]) / 10.0; o += 1

    assert o == FEATURE_DIM, (o, FEATURE_DIM)
    return f
