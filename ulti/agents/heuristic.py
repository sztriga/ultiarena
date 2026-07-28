"""
Rule-based agents for Ulti.

All agents implement the same interface:

    agent.act(obs: dict) -> int

where obs is the dict returned by UltiGame.get_observation(player_id).

The action_mask in obs guarantees that only legal actions are returned;
agents only need to choose WHICH legal action to take.

Agents are instantiated fresh for each game — no state carries over.
"""
from __future__ import annotations

import numpy as np

from ulti.card import card_from_id, SUITS


# ──────────────────────────────────────────────────────────────────────────────
# Observation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _legal(obs: dict) -> np.ndarray:
    """Array of legal action IDs for the current state."""
    return np.flatnonzero(obs['action_mask'])


def _phase(obs: dict) -> int:
    """Current phase index: 1=DISCARD, 2=TRUMP, 3=PLAY."""
    return int(np.argmax(obs['phase']))


def _suit_with_most_cards(obs: dict) -> int:
    """Suit index (0-3) of the suit with the most cards currently in hand."""
    hand = obs['hand']
    counts = [int(hand[s * 8:(s + 1) * 8].sum()) for s in range(4)]
    return int(np.argmax(counts))


def _is_leading(obs: dict) -> bool:
    """True if the current trick is empty (this player leads)."""
    return obs['current_trick'].sum() == 0


# ──────────────────────────────────────────────────────────────────────────────
# Agent classes
# ──────────────────────────────────────────────────────────────────────────────

class RandomAgent:
    """
    Baseline: picks uniformly at random from all legal actions.

    Works correctly in every phase without any game knowledge.
    """

    def __init__(self, seed: int = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> int:
        return int(self._rng.choice(_legal(obs)))


class GreedyAgent:
    """
    Aggressive agent: always plays the highest-ranking legal card.

    Phase behaviour
    ---------------
    TRUMP    Declare the suit with the most cards in hand (hearts → Piros Parti).
    DISCARD  Discard the lowest-rank card first (keep high-value cards).
    PLAY     Always play the highest-rank legal card.

    This agent maximises immediate trick-winning power at the cost of
    card efficiency — a strong declarer, a predictable defender.
    """

    def __init__(self, seed: int = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> int:
        legal = _legal(obs)
        p = _phase(obs)

        if p == 1:  # DISCARD: shed the lowest-rank card
            return int(min(legal, key=lambda a: card_from_id(a).rank_index))

        if p == 2:  # TRUMP: longest suit among Parti/Piros options (actions 0-3)
            hand = obs['hand']
            counts = [int(hand[s * 8:(s + 1) * 8].sum()) for s in range(4)]
            parti_actions = [a for a in legal if a < 4]
            return int(max(parti_actions, key=lambda s: counts[s]))

        # PLAY: highest rank
        return int(max(legal, key=lambda a: card_from_id(a).rank_index))


class ConservativeAgent:
    """
    Defensive agent: always plays the lowest-ranking legal card.

    Phase behaviour
    ---------------
    TRUMP    Same as GreedyAgent (longest suit).
    DISCARD  Same as GreedyAgent (discard lowest-rank first).
    PLAY     Always play the lowest-rank legal card.

    Avoids wasting high-value cards, but also avoids winning tricks.
    A useful defender baseline: doesn't gift aces but won't über aggressively.
    """

    def __init__(self, seed: int = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> int:
        legal = _legal(obs)
        p = _phase(obs)

        if p == 1:  # DISCARD: shed the lowest-rank card
            return int(min(legal, key=lambda a: card_from_id(a).rank_index))

        if p == 2:  # TRUMP: longest suit among Parti/Piros options (actions 0-3)
            hand = obs['hand']
            counts = [int(hand[s * 8:(s + 1) * 8].sum()) for s in range(4)]
            parti_actions = [a for a in legal if a < 4]
            return int(max(parti_actions, key=lambda s: counts[s]))

        # PLAY: lowest rank
        return int(min(legal, key=lambda a: card_from_id(a).rank_index))


class SmartAgent:
    """
    Heuristic agent with role-aware behaviour.

    TRUMP    Longest suit (hearts → Piros Parti if hearts-dominant hand).
    DISCARD  Prefer to discard non-point cards (not aces or 10s), then by
             lowest rank. This preserves scoring cards for tricks.
    PLAY when leading   Highest card (press the advantage).
    PLAY when following Minimum legal card (efficient über or safe discard).

    Compared to GreedyAgent, SmartAgent discards more carefully and plays
    more efficiently when not leading — a stronger overall heuristic.
    """

    def __init__(self, seed: int = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> int:
        legal = _legal(obs)
        p = _phase(obs)

        if p == 1:  # DISCARD: non-point first, then by lowest rank
            def _discard_key(a: int):
                c = card_from_id(a)
                return (1 if c.points > 0 else 0, c.rank_index)
            return int(min(legal, key=_discard_key))

        if p == 2:  # TRUMP: longest suit among Parti/Piros options (actions 0-3)
            hand = obs['hand']
            counts = [int(hand[s * 8:(s + 1) * 8].sum()) for s in range(4)]
            parti_actions = [a for a in legal if a < 4]
            return int(max(parti_actions, key=lambda s: counts[s]))

        # PLAY
        if _is_leading(obs):
            return int(max(legal, key=lambda a: card_from_id(a).rank_index))
        else:
            return int(min(legal, key=lambda a: card_from_id(a).rank_index))
