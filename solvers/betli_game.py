"""Betli GameInterface adapter for trickster's MCTS.

Implements the protocol from ``trickster.games.interface.GameInterface``
so that ``trickster.mcts.alpha_mcts_choose`` can run on betli positions
using ``pis_bridge`` for rules and ``determinize`` for imperfect-info
world sampling.

Perspective convention
----------------------
MCTS expects ``predict_value(feats)`` to return *leaf player's* POV. Our
V-net is trained to output *defender's* POV (+1 = defenders win). To
bridge:

  * ``encode_state(state, player)`` always extracts features from a
    defender's POV (viewer=1 if player==0, else viewer=player) and
    prepends a 1-dim sign flag:
        flag = +1 if player ∈ {1, 2}    (leaf is on defender side)
        flag = -1 if player == 0        (leaf is on soloist side)

  * The companion ``ValueOnlyNet`` wrapper reads the flag and returns
    ``flag * v_def``, which is leaf-player POV as MCTS expects.

This keeps the V-net unchanged and lets us reuse the existing MCTS
without modification.
"""
from __future__ import annotations

import random
from typing import Any, List

import numpy as np

from solvers import determinize as _det
from solvers import pis as _pis
from ulti.card import Card

from vnet.betli import features as _feat_mod
_FEATURE_DIM = _feat_mod.FEATURE_DIM
_extract_features = _feat_mod.extract_features


_CONTRACT = "betli"
_SOLOIST = 0
_DEFENDERS = (1, 2)
_ACTION_SPACE = 32  # 32 cards
_ENCODED_DIM = 1 + _FEATURE_DIM  # 1 sign-flag + 132 features


def _played_from_pos_simple(pos) -> List[Card]:
    """Compute the played-cards list (oldtawer Cards) from a position state."""
    from ulti.card import card_from_id
    all_ids = set(range(32))
    in_hand = {c.id for h in _pis.hands_by_player(pos) for c in h}
    in_trick = {_pis._to_o(c).id for _, c in pos.trick_cards}
    in_talon = {_pis._to_o(c).id for c in (pos.talon_discards or [])}
    return [card_from_id(i) for i in (all_ids - in_hand - in_trick - in_talon)]


class BetliGame:
    """GameInterface impl for betli, defender-perspective MCTS."""

    num_players: int = 3
    state_dim: int = _ENCODED_DIM
    action_space_size: int = _ACTION_SPACE

    # ── rules ────────────────────────────────────────────────────────────
    def current_player(self, state: Any) -> int:
        return _pis.current_player(state)

    def legal_actions(self, state: Any) -> List[Card]:
        return _pis.legal_actions(state)

    def apply(self, state: Any, action: Card) -> Any:
        new_state = state.clone()
        _pis.apply_move(new_state, action)
        return new_state

    def is_terminal(self, state: Any) -> bool:
        return _pis.is_terminal(state)

    def outcome(self, state: Any, player: int) -> float:
        """+1 if player on winning team, -1 else. Betli: defenders win iff
        soloist captured ≥1 trick (≥1 card)."""
        def_wins = len(state.captured[_SOLOIST]) > 0
        is_def = player in _DEFENDERS
        if def_wins:
            return 1.0 if is_def else -1.0
        return -1.0 if is_def else 1.0

    # ── coalition ────────────────────────────────────────────────────────
    def same_team(self, state: Any, player_a: int, player_b: int) -> bool:
        if player_a == player_b:
            return True
        return (player_a in _DEFENDERS) == (player_b in _DEFENDERS)

    # ── determinization ──────────────────────────────────────────────────
    def determinize(self, state: Any, player: int, rng: random.Random) -> Any:
        if player == _SOLOIST:
            return state.clone()  # soloist has full info; no resampling
        iset = _det.build_info_set(state, viewer=player, contract=_CONTRACT)
        hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            return _pis.clone_with_hands_and_talon(state, hands, talon)
        return _pis.clone_with_hands(state, hands)

    # ── encoding ─────────────────────────────────────────────────────────
    def encode_state(self, state: Any, player: int) -> np.ndarray:
        """Return [sign_flag | 132-dim defender-POV features].

        sign_flag = +1 if leaf is a defender, -1 if leaf is soloist. The
        ValueOnlyNet wrapper multiplies V's defender-POV output by this
        flag to give leaf-player POV (the convention MCTS expects).
        """
        viewer = 1 if player == _SOLOIST else player
        hands = _pis.hands_by_player(state)
        played = _played_from_pos_simple(state)
        current_trick = [(p, _pis._to_o(c)) for (p, c) in state.trick_cards]
        # Build a fresh voids dict from the state-derived observation. We
        # don't have the live Voids object here; for MCTS-internal leaves
        # we approximate by using empty voids. (Tree depths are shallow
        # relative to the full game, and voids only refine the partner-
        # hand prior which is already pinned by determinize. So this is
        # cheap and correct enough.)
        feats = _extract_features(
            hands=hands,
            played_cards=played,
            current_trick=current_trick,
            leader=state.leader,
            trick_no=state.trick_no,
            voids={0: frozenset(), 1: frozenset(), 2: frozenset()},
            soloist=_SOLOIST,
            viewer=viewer,
        )
        sign = 1.0 if player in _DEFENDERS else -1.0
        out = np.empty(_ENCODED_DIM, dtype=np.float32)
        out[0] = sign
        out[1:] = feats
        return out

    def action_to_index(self, action: Card) -> int:
        return action.id

    def legal_action_mask(self, state: Any) -> np.ndarray:
        mask = np.zeros(_ACTION_SPACE, dtype=bool)
        for c in _pis.legal_actions(state):
            mask[c.id] = True
        return mask

    def new_game(self, seed: int, **kwargs: Any) -> Any:
        from eval.dojo import deal_betli
        alpha = float(kwargs.get("alpha", 0.5))
        deal = deal_betli(seed=seed, alpha=alpha)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        return _pis.build_position(
            hands=hands, soloist=_SOLOIST, leader=_SOLOIST,
            contract=_CONTRACT, talon=list(deal.talon),
        )
