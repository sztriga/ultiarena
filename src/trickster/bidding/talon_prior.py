"""Learned talon prior: per-card weights conditioned on the winning bid.

During training, after each auction resolves, we observe the actual talon
and update a running average of how likely each card is to appear in the
talon when a given contract is bid.

At pickup time, ``evaluate_pickup`` uses these weights to sample more
realistic talons instead of uniform random.

The prior captures correlations like:
  - When betli is bid, the bidder kept low cards → aces are more likely in talon
  - When ulti in hearts is bid, the bidder kept heart 7 → it's unlikely in talon
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from trickster.games.ulti.cards import ALL_RANKS, ALL_SUITS, Card

# Canonical card ordering: 4 suits × 8 ranks = 32 cards
_DECK = [Card(s, r) for s in ALL_SUITS for r in ALL_RANKS]
_CARD_TO_IDX = {c: i for i, c in enumerate(_DECK)}
NUM_CARDS = len(_DECK)


class TalonPrior:
    """Per-card talon probability conditioned on bid contract.

    For each contract key, maintains a 32-element vector of weights
    representing how likely each card is to appear in the talon when
    that contract wins the auction.

    Updated with exponential moving average during training.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        # contract_key → (32,) float array of running-average talon frequencies
        self._counts: dict[str, np.ndarray] = {}
        self._n: dict[str, int] = {}

    def _ensure_key(self, key: str) -> None:
        if key not in self._counts:
            # Start uniform: each card has 2/32 = 0.0625 base probability
            self._counts[key] = np.full(NUM_CARDS, 1.0 / NUM_CARDS, dtype=np.float64)
            self._n[key] = 0

    def update(self, contract_key: str, talon: list[Card]) -> None:
        """Record that *talon* was the actual talon when *contract_key* was bid."""
        self._ensure_key(contract_key)
        # Build observation: 1 for cards in talon, 0 otherwise
        obs = np.zeros(NUM_CARDS, dtype=np.float64)
        for card in talon:
            idx = _CARD_TO_IDX.get(card)
            if idx is not None:
                obs[idx] = 1.0

        n = self._n[contract_key]
        if n == 0:
            self._counts[contract_key] = obs
        else:
            a = self.alpha
            self._counts[contract_key] = (1 - a) * self._counts[contract_key] + a * obs
        self._n[contract_key] = n + 1

    def get_weights(self, contract_key: str, unknown_cards: list[Card]) -> np.ndarray:
        """Get sampling weights for *unknown_cards* given *contract_key*.

        Returns a weight array aligned with *unknown_cards* (same length).
        Higher weight = more likely to be in the talon for this contract.
        Cards not tracked default to uniform weight.
        """
        if contract_key not in self._counts or self._n.get(contract_key, 0) < 50:
            # Not enough data — fall back to uniform
            return np.ones(len(unknown_cards), dtype=np.float64)

        prior = self._counts[contract_key]
        weights = np.empty(len(unknown_cards), dtype=np.float64)
        for i, card in enumerate(unknown_cards):
            idx = _CARD_TO_IDX.get(card)
            if idx is not None:
                weights[i] = max(prior[idx], 1e-6)
            else:
                weights[i] = 1.0 / NUM_CARDS
        return weights

    @property
    def contract_keys(self) -> list[str]:
        return list(self._counts.keys())

    def sample_count(self, contract_key: str) -> int:
        return self._n.get(contract_key, 0)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "counts": self._counts,
            "n": self._n,
            "alpha": self.alpha,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> TalonPrior:
        with open(path, "rb") as f:
            data = pickle.load(f)
        prior = cls(alpha=data.get("alpha", 0.01))
        prior._counts = data["counts"]
        prior._n = data["n"]
        return prior
