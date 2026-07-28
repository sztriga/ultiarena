"""Small neural network for betli hand evaluation."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from trickster.betli.encoder import BETLI_FEATURE_DIM

_DEFAULT_PATH = Path("models/betli/betli_eval.pt")


class BetliNet(nn.Module):
    """Predicts betli expected value from hand features.

    Architecture: 48 → 128 → ReLU → 128 → ReLU → 64 → ReLU → 1
    Raw output, same as UltiNet value heads (no activation).
    Trained with MSE on shaped rewards.
    """

    def __init__(self, input_dim: int = BETLI_FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_model(net: BetliNet, path: Path | str = _DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)


def load_model(path: Path | str = _DEFAULT_PATH) -> BetliNet:
    net = BetliNet()
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net
