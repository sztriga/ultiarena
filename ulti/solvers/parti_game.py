"""Parti GameInterface adapter for trickster's MCTS.

Mirrors :mod:`solvers.betli_game` but for parti's regression value head.

Convention
----------
Our parti V predicts *soloist points captured* (target 0–90, scaled to
[0, 1] during training, unscaled at inference). MCTS expects a per-leaf
value from **the on-move player's POV**, in roughly [-1, +1].

Mapping: pred_norm ∈ [0, 1] is the scaled soloist-score prediction.
  centered = 2 * pred_norm - 1                 ∈ [-1, +1]
  leaf_val = +centered  if leaf is soloist     (high V = good for sol)
  leaf_val = -centered  if leaf is defender    (low V = good for def)

The encode_state extracts features from `viewer = on-move player` —
the training data covered both soloist and defender viewers, so V is
expected to work from either side.
"""
from __future__ import annotations

import random
from typing import Any, List

import numpy as np

from ulti.solvers import determinize as _det
from ulti.solvers import pis as _pis
from ulti.card import Card, card_from_id
from ulti.vnet.features import extract_features as _extract_features
from ulti.vnet.contracts import get as _get_contract


_CONTRACT = "parti"
_SOLOIST = 0
_DEFENDERS = (1, 2)
_ACTION_SPACE = 32

_cfg = _get_contract(_CONTRACT)
_FEATURE_DIM = _cfg.feature_dim          # parti: 152
_ENCODED_DIM = 1 + _FEATURE_DIM          # 1 sign-flag + 152 features


def _played_from_pos_simple(pos) -> List[Card]:
    all_ids = set(range(32))
    in_hand = {c.id for h in _pis.hands_by_player(pos) for c in h}
    in_trick = {_pis._to_o(c).id for _, c in pos.trick_cards}
    in_talon = {_pis._to_o(c).id for c in (pos.talon_discards or [])}
    return [card_from_id(i) for i in (all_ids - in_hand - in_trick - in_talon)]


class PartiGame:
    """GameInterface impl for parti, regression-V MCTS."""

    num_players: int = 3
    state_dim: int = _ENCODED_DIM
    action_space_size: int = _ACTION_SPACE

    def __init__(self):
        # Caller (matchup script) updates this before each MCTS decision
        # so encode_state can feed the trained V the same voids feature
        # distribution it saw during datagen.
        self.live_voids = {0: frozenset(), 1: frozenset(), 2: frozenset()}

    def current_player(self, state):    return _pis.current_player(state)
    def legal_actions(self, state):     return _pis.legal_actions(state)

    def apply(self, state, action):
        new_state = state.clone()
        _pis.apply_move(new_state, action)
        return new_state

    def is_terminal(self, state):       return _pis.is_terminal(state)

    def outcome(self, state, player):
        """+1 if player on winning team, -1 else. Parti: defenders win
        iff soloist scored <51 points."""
        sol_score = float(state.scores[_SOLOIST])
        sol_wins = sol_score >= 51
        is_sol = (player == _SOLOIST)
        if sol_wins:
            return 1.0 if is_sol else -1.0
        return -1.0 if is_sol else 1.0

    def same_team(self, state, player_a, player_b):
        if player_a == player_b: return True
        return (player_a in _DEFENDERS) == (player_b in _DEFENDERS)

    def determinize(self, state, player, rng):
        # All three perspectives have hidden info in parti: defenders
        # don't know soloist's hand / talon contents, and the soloist
        # doesn't know how the remaining 20 cards split between the two
        # defenders. Sample a consistent world from the on-move player's
        # info-set in all cases.
        iset = _det.build_info_set(state, viewer=player, contract=_CONTRACT)
        hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            return _pis.clone_with_hands_and_talon(state, hands, talon)
        return _pis.clone_with_hands(state, hands)

    def encode_state(self, state, player):
        """Return [sign_flag | 152-dim features].

        sign_flag = +1 if leaf is soloist, -1 if defender. The companion
        `PartiValueOnlyNet` wrapper consumes the flag to flip the
        centered (2V-1) score so MCTS sees leaf-player POV.

        Features are extracted from `viewer = player` — the retrained V
        knows about all 3 viewers.
        """
        viewer = player
        hands = _pis.hands_by_player(state)
        played = _played_from_pos_simple(state)
        current_trick = [(p, _pis._to_o(c)) for (p, c) in state.trick_cards]
        trump_str = state.trump.value.lower() if hasattr(state.trump, "value") \
                    else str(state.trump).lower()
        feats = _extract_features(
            cfg=_cfg,
            hands=hands,
            played_cards=played,
            current_trick=current_trick,
            leader=state.leader,
            trick_no=state.trick_no,
            voids=self.live_voids,
            soloist=_SOLOIST,
            viewer=viewer,
            trump=trump_str,
        )
        sign = 1.0 if player == _SOLOIST else -1.0
        out = np.empty(_ENCODED_DIM, dtype=np.float32)
        out[0] = sign
        out[1:] = feats
        return out

    def action_to_index(self, action): return action.id

    def legal_action_mask(self, state):
        mask = np.zeros(_ACTION_SPACE, dtype=bool)
        for c in _pis.legal_actions(state):
            mask[c.id] = True
        return mask

    def new_game(self, seed: int, **kwargs):
        alpha = float(kwargs.get("alpha", 0.5))
        deal = _cfg.deal_fn(seed=seed, alpha=alpha)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        return _pis.build_position(
            hands=hands, soloist=_SOLOIST, leader=_SOLOIST,
            contract=_CONTRACT, trump=deal.trump, talon=list(deal.talon),
        )
