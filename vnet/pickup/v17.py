"""Exp 17 multi-head pickup net wrapper.

Drop-in replacement for CalibratedPickupNet: same `.predict(X, contract)`
interface. Pads trumpless inputs (32-dim) to the unified 36-dim shape
the multi-head body expects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .multihead import MultiHeadPickupNet, pad_to_unified, UNIFIED_DIM
from .features import HAND_DIM
from .contracts import CONTRACT_CONFIGS


class Exp17Pickup:
    def __init__(self, model: MultiHeadPickupNet):
        self.model = model

    def predict(self, X: np.ndarray, contract: str) -> np.ndarray:
        has_trump = CONTRACT_CONFIGS[contract].has_trump
        if X.shape[1] == HAND_DIM and not has_trump:
            X = pad_to_unified(X, False)
        with torch.no_grad():
            return self.model(torch.from_numpy(X), contract).numpy()

    @classmethod
    def load(cls, weights_path: Path) -> "Exp17Pickup":
        m = MultiHeadPickupNet()
        m.load_state_dict(torch.load(str(weights_path), weights_only=True))
        m.eval()
        return cls(m)
