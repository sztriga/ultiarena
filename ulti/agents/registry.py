"""
Global agent registry.

Populated at import time with the four heuristics from ``game.agents``. Other
agent families (sb3, trickster, …) self-register at module load via their own
``register(...)`` calls — see ``models/trickster/adapter.py`` for an example.

The registry is intentionally a process-global dict. It's fine because agents
have no shared mutable state; each ``make_agent(...)`` call produces a fresh
``ActFn`` instance.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import ActFn, AgentFactory, AgentSpec

_REGISTRY: Dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> None:
    """Add or replace an entry. Replaces silently — last loader wins."""
    _REGISTRY[spec.id] = spec


def list_agents() -> List[AgentSpec]:
    return list(_REGISTRY.values())


def make_agent(agent_id: str, seed: Optional[int] = None) -> ActFn:
    spec = _REGISTRY.get(agent_id)
    if spec is None:
        raise KeyError(
            f"Unknown agent: {agent_id!r}. Known: {sorted(_REGISTRY)}"
        )
    return spec.factory(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Seed: friend's heuristics from game.agents
# ──────────────────────────────────────────────────────────────────────────────

def _heuristic_factory(cls) -> AgentFactory:
    def make(seed: Optional[int] = None) -> ActFn:
        return cls(seed=seed).act
    return make


def _seed_heuristics() -> None:
    from ulti.agents.heuristic import (
        ConservativeAgent,
        GreedyAgent,
        RandomAgent,
        SmartAgent,
    )
    for spec in (
        AgentSpec(
            id="random", label="Random", kind="heuristic",
            description="Uniform random over legal actions.",
            factory=_heuristic_factory(RandomAgent),
        ),
        AgentSpec(
            id="greedy", label="Greedy", kind="heuristic",
            description="Always plays the highest-rank legal card.",
            factory=_heuristic_factory(GreedyAgent),
        ),
        AgentSpec(
            id="conservative", label="Conservative", kind="heuristic",
            description="Always plays the lowest-rank legal card.",
            factory=_heuristic_factory(ConservativeAgent),
        ),
        AgentSpec(
            id="smart", label="Smart", kind="heuristic",
            description="Role-aware: leads high, follows low; preserves point cards on discard.",
            factory=_heuristic_factory(SmartAgent),
        ),
    ):
        register(spec)


_seed_heuristics()
