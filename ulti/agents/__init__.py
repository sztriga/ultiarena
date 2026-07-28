"""
Agent abstraction for the ML-ops stack.

An *agent* is anything callable on a single observation dict:

    act(obs: dict) -> int

Heuristics, SB3 checkpoints, trickster ONNX models, and future families all
expose this interface. This package owns the type, the registry, and the
factory pattern; consumers (arena, eval, FastAPI) look agents up by string id
and never import the concrete agent classes directly.

The current FastAPI agent registry lives in ``apps/api/agents.py`` and
predates this package. The two coexist for now — once the new registry has
checkpoints wired in, we'll have ``apps/api/agents.py`` consume from here.
"""
from .base import ActFn, AgentFactory, AgentSpec
from .registry import list_agents, make_agent, register

__all__ = [
    "ActFn",
    "AgentFactory",
    "AgentSpec",
    "list_agents",
    "make_agent",
    "register",
]
