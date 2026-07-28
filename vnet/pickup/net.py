"""Pickup-net architectures.

PickupNet: small MLP (default 64→64) matching v1 weights for back-compat.
PickupNetV2: larger MLP (256→128) for god-label training.
"""
from __future__ import annotations

import torch.nn as nn


class PickupNet(nn.Module):
    """v1 architecture: in → 64 → 64 → 1 (sigmoid). ~5k params."""
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PickupNetV2(nn.Module):
    """v2 architecture: in → 256 → 128 → 1 (sigmoid), dropout 0.1.

    Sized for god-label training on 250k records per contract.
    """
    def __init__(self, in_dim: int, hidden: tuple[int, int] = (256, 128),
                 dropout: float = 0.1):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, 1),      nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
