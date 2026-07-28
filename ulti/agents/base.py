"""Agent type aliases and the registry entry shape."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

# Anything callable on a single observation dict, returning a legal action id.
ActFn = Callable[[Dict], int]

# Factory: optional seed for stochastic agents → fresh ActFn instance.
AgentFactory = Callable[[Optional[int]], ActFn]


@dataclass(frozen=True)
class AgentSpec:
    """Public metadata for an agent — what the UI / eval reports show."""
    id:          str
    label:       str
    kind:        str    # "heuristic" | "sb3" | "trickster" | …
    description: str
    factory:     AgentFactory
