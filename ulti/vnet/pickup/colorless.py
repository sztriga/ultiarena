"""Structural pickup model for the colorless contracts (betli, durchmars).

Why separate from the trump multi-head net:
  - colorless contracts use a DIFFERENT trick order (Ten demoted under the
    Jack — see solvers/pis.py is_colorless_duri), so the shared trump body
    represents hands with the wrong rank-strength for these contracts;
  - their win-logic is structural (holes / voids / runnable suits), not
    trumps + points;
  - at α=0 they are positive-starved, but a STRUCTURAL model is
    distribution-invariant (a hole pattern has the same win-prob however
    the hand was dealt), so it can train on the plentiful biased god data
    and still calibrate on the α=0 deployment distribution.

``colorless_features`` is vectorized (numpy) so it is fast enough for the
auction's 66-discard × 231-talon enumeration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# rank_index strongest→weakest in the colorless (betli/duri) order:
#   A > K > Q(upper) > J(lower) > 10 > 9 > 8 > 7
STR_ORDER = [7, 5, 4, 3, 6, 2, 1, 0]
N_FEATURES = 20


def colorless_features(X: np.ndarray) -> np.ndarray:
    """(N,32) or (32,) multi-hot → (N,20) suit-invariant structural feats.

    Per suit (sorted canonically for suit-invariance): length, top-run,
    losers. Plus aggregates: n_void, n_single, longest, n_ace, n_ten,
    n_king, total_losers, total_top_run.
    """
    single = X.ndim == 1
    if single:
        X = X[None]
    N = X.shape[0]
    M = X[:, :32].reshape(N, 4, 8).astype(np.float32)
    Ms = M[:, :, STR_ORDER]                  # col 0 = strongest

    length = Ms.sum(2)                        # (N,4)
    top_run = np.cumprod(Ms, axis=2).sum(2)   # leading-1s run

    # cover-walk losers, vectorized over (N,4), only within the held range
    idx = np.arange(8)[None, None, :]
    held_idx = np.where(Ms > 0, idx, -1)
    last = held_idx.max(2, keepdims=True)
    inrange = idx <= last
    cover = np.zeros((N, 4), dtype=np.float32)
    losers = np.zeros((N, 4), dtype=np.float32)
    for k in range(8):
        col = Ms[:, :, k]
        active = inrange[:, :, k]
        is_held = (col > 0) & active
        is_gap = (col == 0) & active
        cover = cover + is_held
        can_cover = is_gap & (cover > 0)
        loser_here = is_gap & (cover <= 0)
        cover = cover - can_cover
        losers = losers + loser_here

    n_void = (length == 0).sum(1)
    n_single = (length == 1).sum(1)
    longest = length.max(1)
    n_ace = M[:, :, 7].sum(1)
    n_ten = M[:, :, 6].sum(1)
    n_king = M[:, :, 5].sum(1)
    total_losers = losers.sum(1)
    total_top = top_run.sum(1)

    key = length * 100 + top_run * 10 - losers
    order = np.argsort(-key, axis=1)
    gthr = lambda a: np.take_along_axis(a, order, axis=1)
    feats = np.concatenate([
        gthr(length), gthr(top_run), gthr(losers),
        np.stack([n_void, n_single, longest, n_ace, n_ten, n_king,
                  total_losers, total_top], axis=1),
    ], axis=1).astype(np.float32)
    return feats[0] if single else feats


class ColorlessStructNet(nn.Module):
    """Tiny MLP on structural features → P(make)."""
    def __init__(self, in_dim: int = N_FEATURES, hidden=(32, 16),
                 dropout: float = 0.05):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ColorlessPickup:
    """Drop-in picker (predict(X, contract)) for one colorless contract.

    Standardizes features with the mean/std baked into the checkpoint.
    ``contract`` arg is accepted but ignored — the model is per-contract.
    """
    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std

    def predict(self, X: np.ndarray, contract: str | None = None) -> np.ndarray:
        f = colorless_features(X)
        if f.ndim == 1:
            f = f[None]
        f = (f - self.mean) / self.std
        with torch.no_grad():
            return self.model(torch.from_numpy(f.astype(np.float32))).numpy()

    @classmethod
    def load(cls, path: Path) -> "ColorlessPickup":
        ckpt = torch.load(str(path), weights_only=True)
        m = ColorlessStructNet(in_dim=ckpt['in_dim'])
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        mean = np.array(ckpt['mean'], dtype=np.float32)
        std = np.array(ckpt['std'], dtype=np.float32)
        return cls(m, mean, std)
