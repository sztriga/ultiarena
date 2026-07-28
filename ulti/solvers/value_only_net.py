"""Adapter wrapping a defender V-net into the duck-typed net interface
that ``trickster.mcts._run_mcts`` calls via ``predict_value``.

Expects features in the format produced by ``BetliGame.encode_state``:
    feats[0]   = sign flag (+1 leaf is defender, -1 leaf is soloist)
    feats[1:]  = 132-dim defender-POV features

Returns leaf-player-POV value as MCTS expects.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class ValueOnlyNet:
    """Duck-typed wrapper: predict_value only. policy is uniform (handled
    by MCTSConfig.use_policy_priors=False)."""

    def __init__(self, torch_net: torch.nn.Module) -> None:
        self._net = torch_net
        self._net.eval()

    def predict_value(self, state_feats: np.ndarray) -> float:
        sign = float(state_feats[0])
        body = state_feats[1:]
        with torch.no_grad():
            x = torch.from_numpy(body.astype(np.float32, copy=False)).unsqueeze(0)
            v_def = float(self._net(x).item())
        return sign * v_def  # leaf-player-POV

    # The MCTS only calls predict_value when use_policy_priors=False and
    # use_value_head=True. predict_policy / predict_both are not needed
    # but we provide minimal stubs for safety.
    def predict_policy(self, state_feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n = int(mask.sum())
        out = np.zeros_like(mask, dtype=np.float64)
        if n > 0:
            out[mask] = 1.0 / n
        return out

    def predict_both(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        return self.predict_value(state_feats), self.predict_policy(state_feats, mask)


class PartiValueOnlyNet:
    """Regression-V adapter for parti.

    Reads the (1 + 152)-dim feature vector from :class:`PartiGame`:
        feats[0]   = sign flag (+1 leaf is soloist, -1 leaf is defender)
        feats[1:]  = 152-dim defender-POV features

    The net outputs v_norm ∈ [0, 1] = predicted soloist score / 90.
    We center to v_centered = 2*v_norm − 1 ∈ [−1, +1], then multiply by
    the sign flag to give leaf-player POV (high = team likes this leaf).
    """

    def __init__(self, torch_net: torch.nn.Module) -> None:
        self._net = torch_net
        self._net.eval()

    def predict_value(self, state_feats: np.ndarray) -> float:
        sign = float(state_feats[0])
        body = state_feats[1:]
        with torch.no_grad():
            x = torch.from_numpy(body.astype(np.float32, copy=False)).unsqueeze(0)
            v_norm = float(self._net(x).item())
        v_centered = 2.0 * v_norm - 1.0
        return sign * v_centered

    def predict_policy(self, state_feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n = int(mask.sum())
        out = np.zeros_like(mask, dtype=np.float64)
        if n > 0:
            out[mask] = 1.0 / n
        return out

    def predict_both(
        self, state_feats: np.ndarray, mask: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        return self.predict_value(state_feats), self.predict_policy(state_feats, mask)
