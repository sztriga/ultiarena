"""CompositePickup: route pickup queries to the right model per contract.

  parti, ulti  → trump multi-head net (exp 18 v18a, Exp18Pickup)
  betli        → dedicated colorless structural net
  durchmars    → dedicated colorless structural net

Implements the standard ``predict(X, contract)`` interface, so it is a
drop-in for the auction / god_check / tournament harness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .v18 import Exp18Pickup
from .colorless import ColorlessPickup


class CompositePickup:
    def __init__(self, trump, betli, durchmars):
        self.trump = trump
        self.betli = betli
        self.durchmars = durchmars

    def predict(self, X: np.ndarray, contract: str) -> np.ndarray:
        if contract in ('parti', 'ulti'):
            return self.trump.predict(X, contract)
        if contract == 'betli':
            return self.betli.predict(X, contract)
        if contract == 'durchmars':
            return self.durchmars.predict(X, contract)
        raise ValueError(f"unknown contract {contract!r}")

    @classmethod
    def load(cls, *, trump_weights: Path, betli_weights: Path,
             durchmars_weights: Path) -> "CompositePickup":
        return cls(
            trump=Exp18Pickup.load(trump_weights),
            betli=ColorlessPickup.load(betli_weights),
            durchmars=ColorlessPickup.load(durchmars_weights),
        )
