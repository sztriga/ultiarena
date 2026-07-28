"""Isotonic calibration wrapper for v2 pickup nets.

The v2 nets are trained on biased deals (per-contract α tuned for ~50%
positive labels). At inference on RANDOM deals, predictions are
miscalibrated (e.g. v2 says betli wins 95%, actual is 41%). This module
fits a 1-D monotonic correction p̂ → p_true per contract.

Usage:
    from vnet.pickup.calibration import CalibratedPickupNet
    cnet = CalibratedPickupNet.load(weights_dir)
    p_corrected = cnet.predict(hand_batch, contract)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .contracts import CONTRACT_CONFIGS
from .features import featurize, input_dim
from .net import PickupNetV2


class CalibratedPickupNet:
    """Wraps PickupNetV2 with per-contract isotonic correction."""

    def __init__(self,
                 vnets: Dict[str, PickupNetV2],
                 isotonics: Dict[str, "IsotonicRegression"]):
        self.vnets = vnets
        self.isotonics = isotonics

    def predict(self, X: np.ndarray, contract: str) -> np.ndarray:
        """Return calibrated P_make for a batch of featurized hands."""
        with torch.no_grad():
            raw = self.vnets[contract](torch.from_numpy(X)).numpy()
        if contract in self.isotonics:
            return self.isotonics[contract].predict(raw)
        return raw

    @classmethod
    def load(cls, weights_dir: Path, calib_dir: Optional[Path] = None):
        """Load v2 weights from weights_dir, isotonic from calib_dir
        (defaults to weights_dir)."""
        weights_dir = Path(weights_dir)
        calib_dir = Path(calib_dir) if calib_dir else weights_dir
        vnets = {}
        isos = {}
        for name, cfg in CONTRACT_CONFIGS.items():
            m = PickupNetV2(in_dim=input_dim(cfg))
            m.load_state_dict(
                torch.load(weights_dir / f"{name}_vnet_v2.pt",
                           weights_only=True))
            m.eval()
            vnets[name] = m
            cpath = calib_dir / f"{name}_calib.pkl"
            if cpath.exists():
                with open(cpath, 'rb') as f:
                    isos[name] = pickle.load(f)
        return cls(vnets, isos)
