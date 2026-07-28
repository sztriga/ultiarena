"""Betli-specific HandEvaluator using BetliNet.

Replaces the default UltiNet-based evaluation for betli contracts
with a specialized small NN that operates on betli-specific features.
Output is in [-1, 1], same scale as UltiNet value heads.
Also supports online training from game outcomes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from trickster.betli.datagen import discard_candidates
from trickster.betli.encoder import BETLI_FEATURE_DIM, encode_hand
from trickster.betli.model import BetliNet
from trickster.bidding.evaluator import ContractEval, DiscardChoice
from trickster.bidding.registry import ContractDef
from trickster.games.ulti.cards import Card, Suit
from trickster.games.ulti.game import GameState
from trickster.train_utils import _GAME_PTS_MAX


class BetliHandEvaluator:
    """HandEvaluator for betli using a specialized BetliNet.

    Evaluation:
        Uses ``discard_candidates`` (<=10 pairs) instead of all C(12,2)=66,
        encodes remaining 10-card hands with the betli-specific encoder,
        and predicts expected value via BetliNet ([-1, 1] scale).

    Training:
        Call ``record_outcome`` after each betli game with the soloist's
        10-card hand and the shaped reward from ``simple_outcome``.
        Call ``train_step`` periodically to update the network.
    """

    def __init__(
        self,
        net: BetliNet,
        device: str = "cpu",
        buffer_capacity: int = 10_000,
        lr: float = 1e-3,
    ) -> None:
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)
        self._lr = lr

        # Ring buffer for training
        self._buf_feats = np.zeros((buffer_capacity, BETLI_FEATURE_DIM), dtype=np.float32)
        self._buf_outcomes = np.zeros(buffer_capacity, dtype=np.float32)
        self._buf_size = 0
        self._buf_pos = 0
        self._buf_cap = buffer_capacity

        self._optimizer: torch.optim.Adam | None = None

    # ------------------------------------------------------------------
    #  HandEvaluator protocol
    # ------------------------------------------------------------------

    def evaluate_contract(
        self,
        gs: GameState,
        soloist: int,
        dealer: int,
        contract_def: ContractDef,
        trump: Suit | None,
        is_piros: bool,
    ) -> ContractEval | None:
        hand = gs.hands[soloist]
        if len(hand) != 12:
            return None

        pairs = discard_candidates(hand)
        if not pairs:
            return None

        feats_list = []
        for c1, c2 in pairs:
            remaining = [c for c in hand if c is not c1 and c is not c2]
            feats_list.append(encode_hand(remaining))

        feats_batch = np.stack(feats_list)
        with torch.no_grad():
            values = self.net(
                torch.from_numpy(feats_batch).to(self.device)
            ).cpu().numpy()

        # Net outputs [-1, 1] directly, same scale as UltiNet value heads.
        scale = _GAME_PTS_MAX / 2
        game_pts_arr = values * scale

        best_idx = int(np.argmax(values))
        mean_pts = float(np.mean(game_pts_arr))

        return ContractEval(
            contract_key=contract_def.key,
            trump=None,
            is_piros=is_piros,
            best_discard=DiscardChoice(
                discard=pairs[best_idx],
                value=float(values[best_idx]),
                game_pts=float(game_pts_arr[best_idx]),
            ),
            game_pts=mean_pts,
            stakes_pts=mean_pts,
        )

    # ------------------------------------------------------------------
    #  Online training
    # ------------------------------------------------------------------

    def record_outcome(self, hand_10: list[Card], outcome: float) -> None:
        """Record (hand_features, outcome) for training.

        Parameters
        ----------
        hand_10 : 10-card hand at game start
        outcome : shaped reward from simple_outcome, in [-1, 1]
        """
        feats = encode_hand(hand_10)
        self._buf_feats[self._buf_pos] = feats
        self._buf_outcomes[self._buf_pos] = outcome
        self._buf_pos = (self._buf_pos + 1) % self._buf_cap
        self._buf_size = min(self._buf_size + 1, self._buf_cap)

    @property
    def buffer_size(self) -> int:
        return self._buf_size

    def train_step(self, batch_size: int = 256) -> float | None:
        """Run one SGD step on accumulated data. Returns loss or None."""
        if self._buf_size < 16:
            return None

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.net.parameters(), lr=self._lr)

        n = min(batch_size, self._buf_size)
        indices = np.random.choice(self._buf_size, size=n, replace=False)

        x = torch.from_numpy(self._buf_feats[indices]).to(self.device)
        y = torch.from_numpy(self._buf_outcomes[indices]).to(self.device)

        self.net.train()
        pred = self.net(x)
        loss = F.mse_loss(pred, y)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self._optimizer.step()
        self.net.eval()

        return loss.item()
