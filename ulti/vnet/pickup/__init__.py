"""Pickup featurizer — P(soloist makes contract X with trump T) inputs from a
10-card post-discard hand. The DEPLOYED surface is ``featurize``/``input_dim``
(+ the suit-permutation augmentation the pipeline trains with); the provider's
heads themselves are ``ulti.bidding.base_head.Head`` + per-head isotonic tables.

The pre-retrain model stack (contracts/net/multihead/canonical, exp15-19) is
research-replay material: still importable by name, but loaded LAZILY so the
serving chain never drags it (contracts.py pulls in the dojo)."""
from .features import featurize, input_dim
from .augment import apply_suit_perm, random_suit_perm

_LEGACY = {
    "CONTRACT_CONFIGS": "contracts", "ContractCfg": "contracts",
    "PickupNet": "net", "PickupNetV2": "net",
    "MultiHeadPickupNet": "multihead", "pad_to_unified": "multihead",
    "UNIFIED_DIM": "multihead",
    "canonicalize": "canonical", "CANON_DIM": "canonical",
}

__all__ = ["featurize", "input_dim", "random_suit_perm", "apply_suit_perm",
           *_LEGACY]


def __getattr__(name: str):
    if name in _LEGACY:
        from importlib import import_module
        return getattr(import_module(f".{_LEGACY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
