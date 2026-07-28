"""Multi-head pickup-net: shared body + 4 contract heads.

Trumpless contracts (betli, durchmars) feed a 36-dim input with zeros
in the trump slots, so the body input is uniform across all contracts.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .contracts import CONTRACT_CONFIGS
from .features import HAND_DIM, TRUMP_DIM

UNIFIED_DIM = HAND_DIM + TRUMP_DIM   # always 36


class MultiHeadPickupNet(nn.Module):
    """36 → body(128, 64) → 4 heads (32 → 1).

    ``input_dim=32`` for canonical (suit-invariant) features, which
    drop the trump one-hot — see ``vnet.pickup.canonical``.
    """
    def __init__(self,
                 body_hidden: tuple[int, int] = (128, 64),
                 head_hidden: int = 32,
                 dropout: float = 0.1,
                 contracts: Iterable[str] = ('betli', 'durchmars',
                                             'parti', 'ulti'),
                 input_dim: int = UNIFIED_DIM):
        super().__init__()
        h1, h2 = body_hidden
        self.body = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),         nn.ReLU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            c: nn.Sequential(
                nn.Linear(h2, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, 1), nn.Sigmoid(),
            )
            for c in contracts
        })

    def forward(self, x: torch.Tensor, contract: str) -> torch.Tensor:
        return self.heads[contract](self.body(x)).squeeze(-1)


def pad_to_unified(X: np.ndarray, has_trump: bool) -> np.ndarray:
    """Pad a trumpless 32-dim batch to 36-dim with zero trump slots."""
    if has_trump:
        assert X.shape[1] == UNIFIED_DIM
        return X
    assert X.shape[1] == HAND_DIM
    out = np.zeros((X.shape[0], UNIFIED_DIM), dtype=X.dtype)
    out[:, :HAND_DIM] = X
    return out
