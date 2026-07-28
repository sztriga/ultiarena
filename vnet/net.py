"""Defender value-net: simple MLP, tanh output.

V(info_set) ∈ (−1, +1) predicts defender-team game value:
  +1 = defenders win, −1 = soloist wins (deterministic god-solver label).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple = (256, 128),
                 dropout: float = 0.0, target_kind: str = "binary") -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        if target_kind == "binary":
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
