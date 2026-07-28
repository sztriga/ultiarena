"""Canonical pickup-net library.

Pickup-net: P(soloist makes contract X with trump T) given a 10-card
post-discard hand. Used during the bidding phase.
"""
from .contracts import CONTRACT_CONFIGS, ContractCfg
from .features import featurize, input_dim
from .net import PickupNet, PickupNetV2
from .augment import random_suit_perm, apply_suit_perm
from .multihead import MultiHeadPickupNet, pad_to_unified, UNIFIED_DIM
from .canonical import canonicalize, CANON_DIM

__all__ = [
    "CONTRACT_CONFIGS", "ContractCfg",
    "featurize", "input_dim",
    "PickupNet", "PickupNetV2",
    "random_suit_perm", "apply_suit_perm",
    "MultiHeadPickupNet", "pad_to_unified", "UNIFIED_DIM",
    "canonicalize", "CANON_DIM",
]
