"""Experiment-local path helpers; library code lives in vnet/pickup/.

Re-exports the canonical pickup-net pieces so existing exp 15 scripts
don't need to change their import sites.
"""
from __future__ import annotations

from pathlib import Path

from ulti.vnet.pickup import (
    CONTRACT_CONFIGS, ContractCfg,
    featurize, input_dim,
    PickupNet,
)

EXP_DIR = Path(__file__).parent


def data_path(name: str) -> Path:
    return EXP_DIR / f"{name}_data_10k.npz"


def weights_path(name: str) -> Path:
    return EXP_DIR / f"{name}_vnet.pt"


__all__ = [
    "CONTRACT_CONFIGS", "ContractCfg",
    "featurize", "input_dim",
    "PickupNet",
    "EXP_DIR", "data_path", "weights_path",
]
