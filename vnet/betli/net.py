"""Defender value-net: simple MLP, tanh output.

V(info_set) ∈ (−1, +1) predicts defender-team game value:
  +1 = defenders win, −1 = soloist wins (deterministic god-solver label).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple = (256, 128),
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers += [nn.Linear(prev, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_net_from_ckpt(ckpt: dict) -> nn.Module:
    """Build a defender V-net from a saved checkpoint (value or nnue).

    Lives here (rather than in the eval scripts) so model-loading is
    decoupled from the matchup harness — used at runtime by
    ``apps/api/betli_hu.py`` to serve the trained checkpoint.
    """
    net_type = ckpt.get("net_type", "value")
    if net_type == "nnue":
        from .net_nnue import NnueValueNet
        net = NnueValueNet(
            input_dim=ckpt["input_dim"],
            embed_dim=ckpt["nnue_embed"],
            trunk=tuple(ckpt["nnue_trunk"]),
            dropout=ckpt.get("dropout", 0.0),
        )
    else:
        net = ValueNet(
            input_dim=ckpt["input_dim"], hidden=tuple(ckpt["hidden"]),
        )
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    return net
