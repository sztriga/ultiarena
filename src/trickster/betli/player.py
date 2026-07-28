"""Betli engine: solver for play, specialized NN for hand evaluation."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from trickster.betli.datagen import discard_candidates
from trickster.betli.encoder import encode_hand
from trickster.betli.model import BetliNet, load_model
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import Card

try:
    from trickster._solver_core import solve_root as _solve_root
except ImportError:
    from trickster.games.ulti.solver import solve_root as _solve_root


class BetliEngine:
    """Standalone betli engine.

    - Play decisions: pure PIMC solver (no NN)
    - Hand evaluation: betli-specific NN
    - Discard: evaluate top-per-suit pairs via NN, pick best
    """

    def __init__(
        self,
        model_path: Path | str = "models/betli/betli_eval.pt",
        pimc_dets: int = 30,
    ) -> None:
        import torch
        self.net = load_model(model_path)
        self.pimc_dets = pimc_dets
        self.game = UltiGame(restrictions=[])
        self._torch = torch

    def evaluate_hand(self, hand_10: list[Card]) -> float:
        """Predict expected value for a 10-card betli hand ([-1, 1])."""
        feats = encode_hand(hand_10)
        return self.evaluate_hand_from_features(feats)

    def evaluate_hand_from_features(self, feats: np.ndarray) -> float:
        """Predict expected value from pre-computed feature vector ([-1, 1])."""
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(feats).unsqueeze(0)
            return self.net(tensor).item()

    def best_discard(self, hand_12: list[Card]) -> tuple[list[Card], float]:
        """Find best 2-card discard from 12-card hand.

        Only considers top-per-suit pairs: cross-suit (top × top) or
        same-suit (top 2). At most ~10 pairs instead of C(12,2)=66.

        Returns (discard_pair, predicted_value_of_remaining_10).
        """
        pairs = discard_candidates(hand_12)
        if not pairs:
            return [hand_12[0], hand_12[1]], 0.0

        feats_list = []
        for c1, c2 in pairs:
            remaining = [c for c in hand_12 if c is not c1 and c is not c2]
            feats_list.append(encode_hand(remaining))

        feats_batch = np.stack(feats_list)
        with self._torch.no_grad():
            preds = self.net(self._torch.from_numpy(feats_batch)).numpy()

        best_idx = int(np.argmax(preds))
        best_pair = list(pairs[best_idx])
        best_val = float(preds[best_idx])

        return best_pair, best_val

    def choose_action(
        self,
        state: UltiNode,
        player: int,
        rng: random.Random,
    ) -> Card:
        """Choose best card using pure PIMC solver."""
        legal = self.game.legal_actions(state)
        if len(legal) <= 1:
            return legal[0]

        totals: dict[Card, float] = {c: 0.0 for c in legal}
        counts: dict[Card, int] = {c: 0 for c in legal}

        for _ in range(self.pimc_dets):
            det = self.game.determinize(state, player, rng)
            vals = _solve_root(det.gs, contract="betli")
            for card in legal:
                if card in vals:
                    totals[card] += vals[card]
                    counts[card] += 1

        avg = {c: totals[c] / max(1, counts[c]) for c in legal}

        if player == state.gs.soloist:
            return max(avg, key=avg.__getitem__)
        return min(avg, key=avg.__getitem__)
