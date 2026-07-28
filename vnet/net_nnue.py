"""NNUE-style defender value-net.

Motivation
----------
The hand-engineered 132-dim feature vector in ``features.py`` mixes:

  * sparse one-hot card indicators (own hand 32, played 32, current trick 39)
  * hand-derived summaries that *could* be re-derived by the net itself
    (top_per_suit, live_top_per_suit, cards_out_per_suit)
  * a small dense tail of scalars (role, voids, trick_no, hand sizes, flags)

The NNUE design philosophy: feed the *raw sparse* card-position indicators
into a *wide* first layer with no hand-crafted summaries. The first-layer
weights, summed only over the active (set-to-1) indicator bits, act as a
learned per-card embedding. Then a small dense MLP processes the resulting
embedding plus the irreducibly-dense scalar features.

Why this might beat the engineered MLP
---------------------------------------
* Our engineered summaries (`top_per_suit`) are lossy: an ace+seven of
  hearts and an ace+ten of hearts have the same `my_top_per_suit[hearts]`
  but very different game-theoretic value. A learned embedding can
  preserve the joint card-set structure.
* The first layer has independent weights per card, so it can learn
  non-monotone per-card valuations (e.g. the 7 of a suit is actually the
  *worst* card to hold in betli because it can never take a trick — but a
  monotone "top rank" summary doesn't surface that).

Layout for this implementation
------------------------------
* **Sparse branch (NNUE bit):** concatenation of the two card one-hots
    [own_hand (32)] [played (32)]  → 64-dim
  fed into ``Linear(64, embed_dim)`` → ReLU. Acts as a learned embedding
  of "set of cards in defender hand" ⊕ "set of cards already seen".
* **Dense branch:** the remaining scalar/categorical features
    [trick state (39)] [trick_no (1)] [sol_voids (4)] [par_voids (4)]
    [role (2)] [played-flags (3)] [remaining_tricks (1)]
    [partner_sz (1)] [sol_sz (1)] = 56-dim
  (intentionally drops `cards_out_per_suit` (4), `my_top_per_suit` (4),
   `live_top_per_suit` (4): all are derivable from the sparse branch.)
* **Trunk:** concat(sparse_embed, dense) → MLP → Tanh.

Defaults are intentionally on the larger side (embed=256, trunk=(512,256))
since we now have 400k training examples — capacity isn't the constraint.

Note on slicing
---------------
This module hard-codes the feature layout from ``features.py``. If that
layout changes the slice constants below must change too.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# Slice indices into the 132-dim feature vector from features.py.
_SLICE_OWN_HAND = (0, 32)
_SLICE_PLAYED   = (32, 64)
_SLICE_TRICK    = (64, 103)
_SLICE_TRICK_NO = (103, 104)
_SLICE_SOL_VOID = (104, 108)
_SLICE_PAR_VOID = (108, 112)
_SLICE_ROLE     = (112, 114)
_SLICE_FLAGS    = (114, 117)
# 117:121 cards_out_per_suit  — DROPPED (derivable)
# 121:125 my_top_per_suit    — DROPPED (derivable)
# 125:129 live_top_per_suit  — DROPPED (derivable)
_SLICE_REM_TR   = (129, 130)
_SLICE_PAR_SZ   = (130, 131)
_SLICE_SOL_SZ   = (131, 132)

# Sparse (card-indicator) feature bits used by the NNUE-style first layer.
_SPARSE_SLICES = [_SLICE_OWN_HAND, _SLICE_PLAYED]      # 64 bits total
# Dense (scalar/categorical) features kept for the dense branch.
_DENSE_SLICES = [
    _SLICE_TRICK, _SLICE_TRICK_NO,
    _SLICE_SOL_VOID, _SLICE_PAR_VOID,
    _SLICE_ROLE, _SLICE_FLAGS,
    _SLICE_REM_TR, _SLICE_PAR_SZ, _SLICE_SOL_SZ,
]  # 39+1+4+4+2+3+1+1+1 = 56 dims


def _gather(x: torch.Tensor, slices) -> torch.Tensor:
    return torch.cat([x[:, a:b] for (a, b) in slices], dim=-1)


class NnueValueNet(nn.Module):
    """NNUE-style: wide first layer over sparse card indicators, then trunk."""

    def __init__(self, input_dim: int = 132,
                 embed_dim: int = 256,
                 trunk: tuple = (512, 256),
                 dropout: float = 0.0) -> None:
        super().__init__()
        assert input_dim == 132, "feature layout hard-coded for the 132-dim spec"

        sparse_dim = sum(b - a for (a, b) in _SPARSE_SLICES)   # 64
        dense_dim  = sum(b - a for (a, b) in _DENSE_SLICES)    # 56

        self.embed = nn.Sequential(
            nn.Linear(sparse_dim, embed_dim),
            nn.ReLU(),
        )

        layers: list[nn.Module] = []
        prev = embed_dim + dense_dim
        for h in trunk:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers += [nn.Linear(prev, 1), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sparse = _gather(x, _SPARSE_SLICES)
        dense  = _gather(x, _DENSE_SLICES)
        emb = self.embed(sparse)
        return self.trunk(torch.cat([emb, dense], dim=-1)).squeeze(-1)
