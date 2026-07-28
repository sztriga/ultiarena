"""
Shared constants and factories for defender pools and pregame selectors.

Exports
-------
PASSZ_ACTION
    Action index (16) reserved for PASSZ in the Discrete(32) pre-game space.

DEFENDERS, DEFENDER_POOL
    Standard four-agent registry used across all training scripts.

make_adversarial_cls(model)
    Wrap a trained defender MaskablePPO as a drop-in defender class.

make_pregame_selector(pregame_agent, trump_lo, trump_hi)
    Build a trump selector backed by a trained pre-game agent.  Returns the
    chosen game action (trump_lo … trump_hi) when the agent bids, or None
    when it would PASSZ.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Type

import numpy as np

from agents.heuristic import RandomAgent, GreedyAgent, ConservativeAgent, SmartAgent
from envs.obs import play_flatten as _def_flatten

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Action slot reserved for PASSZ in the Discrete(32) pre-game action space.
PASSZ_ACTION = 16

# ──────────────────────────────────────────────────────────────────────────────
# Defender registry
# ──────────────────────────────────────────────────────────────────────────────

DEFENDERS: Dict[str, Type] = {
    'random':       RandomAgent,
    'greedy':       GreedyAgent,
    'conservative': ConservativeAgent,
    'smart':        SmartAgent,
}
DEFENDER_POOL: List[Type] = list(DEFENDERS.values())


def make_adversarial_cls(model):
    """Wrap a trained defender MaskablePPO as a drop-in defender class."""
    class AdversarialDefender:
        def act(self, obs_dict: dict) -> int:
            flat = _def_flatten(obs_dict)
            mask = obs_dict['action_mask'].astype(bool)
            action, _ = model.predict(flat, action_masks=mask, deterministic=True)
            return int(action)
    return AdversarialDefender


# ──────────────────────────────────────────────────────────────────────────────
# Trump selector factory
# ──────────────────────────────────────────────────────────────────────────────

def make_pregame_selector(
    pregame_agent,
    trump_lo: int,
    trump_hi: int,
) -> Callable:
    """
    Build a trump selector backed by a trained pre-game agent.

    At the TRUMP decision step the pre-game agent sees a 264-dim obs and
    chooses from a mask containing {trump_lo … trump_hi} ∪ {PASSZ_ACTION=16}.
    Returns the chosen game action (trump_lo … trump_hi) when the agent bids,
    or None when it would PASSZ (so PlayEnv resamples the hand).
    """
    def selector(obs_dict: dict, legal: list) -> Optional[int]:
        flat = _def_flatten(obs_dict)
        mask = np.zeros(32, dtype=bool)
        mask[PASSZ_ACTION] = True
        for a in legal:
            if trump_lo <= a <= trump_hi:
                mask[a] = True
        action, _ = pregame_agent.predict(flat, action_masks=mask, deterministic=True)
        if int(action) == PASSZ_ACTION:
            return None
        return int(action)
    return selector
