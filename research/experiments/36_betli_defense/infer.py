"""Deployable betli-DEFENSE policy (exp36). Self-contained cheat-clean inference: given a betli
position and the defender to move, pick the card most likely to keep the soloist from making the
betli. Beats PIMC defense by ~18pp steal-rate on defender-holdable betlis (recovers ~30% of the
headroom), and is FASTER than PIMC (a single forward pass, no search).

play.py can import this and call `betli_defense_pick(pos, viewer)` when the AI is a defender in a
(plain, trumpless) betli game. Model path via env BETLI_DEF_MODEL (default models/betli/betli_defense.pt).
Torch-lazy: nothing is imported until the first call, so importing this module is cheap.
"""
from __future__ import annotations

import os

import numpy as np

from ulti.solvers import pis as _pis

_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
_MODEL_PATH = os.environ.get("BETLI_DEF_MODEL", os.path.join(_REPO, "models/betli/betli_defense.pt"))
FEAT_DIM = 139
_NET = None
_SUIT_IX = {"acorns": 0, "leaves": 1, "hearts": 2, "bells": 3}


def encode(pos, viewer: int) -> np.ndarray:
    """139-dim cheat-clean features for `viewer`'s betli-defense decision (must match training)."""
    hands = _pis.hands_by_player(pos)
    own = hands[viewer]
    trick = [_pis._to_o(c) for _p, c in pos.trick_cards]
    played = []
    for cap in pos.captured:
        played += [_pis._to_o(c) for c in cap]
    played += trick
    x = np.zeros(FEAT_DIM, dtype=np.float32)
    for c in own:
        x[c.id] = 1.0
    seen = set()
    for c in played:
        x[32 + c.id] = 1.0
        seen.add(c.id)
    own_ids = {c.id for c in own}
    for i in range(32):
        if i not in own_ids and i not in seen:
            x[64 + i] = 1.0
    for c in trick:
        x[96 + c.id] = 1.0
    if trick:
        x[128 + _SUIT_IX[trick[0].suit]] = 1.0
    else:
        x[132] = 1.0
    x[133 + min(2, len(trick))] = 1.0
    x[136 + (viewer - 1)] = 1.0
    x[138] = len(own) / 10.0
    return x


def _make_net(hidden: int, arch: str, dropout: float):
    import torch.nn as nn
    if arch == "wide":
        dims = [FEAT_DIM, hidden, hidden, hidden, 32]
    elif arch == "deep":
        dims = [FEAT_DIM, hidden, hidden, hidden, hidden, 32]
    else:
        dims = [FEAT_DIM, hidden, hidden, 32]
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)


def available() -> bool:
    return os.path.exists(_MODEL_PATH)


def _load():
    global _NET
    import torch
    ckpt = torch.load(_MODEL_PATH, weights_only=False)
    net = _make_net(ckpt["hidden"], ckpt["arch"], ckpt.get("dropout", 0.0))
    net.load_state_dict(ckpt["state"]); net.eval()
    _NET = net


def betli_defense_pick(pos, viewer: int):
    """The learned defensive card for `viewer` at `pos`, or None if the model is unavailable."""
    if not available():
        return None
    if _NET is None:
        _load()
    import torch
    with torch.no_grad():
        lg = _NET(torch.from_numpy(encode(pos, viewer)).unsqueeze(0))[0].numpy()
    return max(_pis.legal_actions(pos), key=lambda c: lg[c.id])
