"""Exp 18 canonical multi-head pickup net wrapper.

Drop-in replacement for Exp17Pickup / CalibratedPickupNet: same
`.predict(X, contract)` interface, where X uses the standard
``featurize`` layout (36-dim for trump contracts, 32-dim trumpless).
Canonicalizes to the suit-invariant 32-dim form internally.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .multihead import MultiHeadPickupNet
from .canonical import CANON_DIM, canonicalize
from .contracts import CONTRACT_CONFIGS


class Exp18Pickup:
    def __init__(self, model: MultiHeadPickupNet):
        self.model = model

    def predict(self, X: np.ndarray, contract: str) -> np.ndarray:
        has_trump = CONTRACT_CONFIGS[contract].has_trump
        Xc = canonicalize(X, has_trump)
        with torch.no_grad():
            return self.model(torch.from_numpy(Xc), contract).numpy()

    @classmethod
    def load(cls, weights_path: Path) -> "Exp18Pickup":
        m = MultiHeadPickupNet(input_dim=CANON_DIM)
        m.load_state_dict(torch.load(str(weights_path), weights_only=True))
        m.eval()
        return cls(m)
