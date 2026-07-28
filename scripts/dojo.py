#!/usr/bin/env python3
"""Dojo — focused contract training with biased dealing.

Forces specific contract games with biased dealing that gives the soloist
hands resembling real contract hands.  The model learns both what good
hands look like and how to play them.

Each contract type provides its own dealing, discard, and game setup
logic via a :class:`ContractDojo` interface.

Usage:
    python scripts/dojo.py trinity --contract betli --steps 200 --workers 4
    python scripts/dojo.py trinity --contract ulti --steps 200 --workers 4
    python scripts/dojo.py trinity --contract betli --freeze-value --steps 500
"""
from __future__ import annotations

import abc
import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.bidding.evaluator import _make_eval_state
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import (
    ALL_SUITS,
    BETLI_STRENGTH,
    Card,
    Rank,
    Suit,
    make_deck,
)
from trickster.games.ulti.game import (
    GameState,
    deal,
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
)
from trickster.games.ulti.hybrid import HybridPlayer
from trickster.games.ulti.rewards import simple_outcome
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, make_wrapper
from trickster.train_utils import ReplayBuffer
from trickster.training.model_io import auto_device, load_net
from trickster.training.tiers import TIERS


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _weighted_sample_without_replacement(
    deck: list[Card],
    weights: list[float],
    n: int,
    rng: random.Random,
) -> tuple[list[Card], list[Card]]:
    """Pick *n* cards from *deck* using *weights*, return (chosen, rest)."""
    remaining = list(range(len(deck)))
    remaining_weights = list(weights)
    chosen_indices: list[int] = []

    for _ in range(n):
        total = sum(remaining_weights)
        r = rng.random() * total
        cumulative = 0.0
        pick = 0
        for i, w in enumerate(remaining_weights):
            cumulative += w
            if cumulative >= r:
                pick = i
                break
        chosen_indices.append(remaining[pick])
        remaining.pop(pick)
        remaining_weights.pop(pick)

    chosen = [deck[i] for i in chosen_indices]
    rest = [deck[i] for i in remaining]
    return chosen, rest


def _deal_rest(
    rest: list[Card],
    rng: random.Random,
) -> tuple[list[list[Card]], list[Card]]:
    """Shuffle *rest* (22 cards) into def1(10), def2(10), talon(2)."""
    rng.shuffle(rest)
    return [rest[:10], rest[10:20]], rest[20:22]


def _make_gs(
    hands: list[list[Card]],
    dealer: int,
    soloist: int,
) -> GameState:
    """Build a fresh 10-card GameState."""
    return GameState(
        hands=hands,
        trump=None,
        betli=False,
        soloist=soloist,
        dealer=dealer,
        captured=[[], [], []],
        scores=[0, 0, 0],
        leader=next_player(dealer),
        trick_no=0,
        trick_cards=[],
        last_trick=None,
    )


# ---------------------------------------------------------------------------
#  Contract dojo interface
# ---------------------------------------------------------------------------

_Sample = tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]


class ContractDojo(abc.ABC):
    """Interface for contract-specific dojo logic."""

    def _randomize_alpha(self, rng: random.Random, alpha: float) -> float:
        """Randomize alpha per game: Uniform(0, alpha).

        Gives a wide quality spectrum so the value head sees both good
        and bad hands, preventing calibration collapse.
        """
        return rng.uniform(0, alpha)

    @abc.abstractmethod
    def deal(
        self, rng: random.Random, alpha: float, suit_sigma: float,
    ) -> tuple[list[list[Card]], list[Card]]:
        """Return (hands[3], talon[2]) with biased soloist hand at index 0."""

    @abc.abstractmethod
    def hand_quality(self, hand: list[Card]) -> float:
        """Return a 0-1 quality score for the soloist hand."""

    @abc.abstractmethod
    def best_discard(
        self, gs: GameState, soloist: int, wrapper, game: UltiGame,
    ) -> list[Card]:
        """Return the 2 cards to discard from the 12-card hand."""

    @abc.abstractmethod
    def setup_node(
        self, gs: GameState, soloist: int, dealer: int,
    ) -> UltiNode:
        """Set contract on gs and return a ready-to-play UltiNode."""

    @property
    @abc.abstractmethod
    def kontra_key(self) -> str:
        """Component key used for kontra bookkeeping."""

    @property
    @abc.abstractmethod
    def training_mode(self) -> str:
        """Value for gs.training_mode."""


# ---------------------------------------------------------------------------
#  Betli dojo
# ---------------------------------------------------------------------------


class BetliDojo(ContractDojo):

    def deal(self, rng, alpha, suit_sigma):
        deck = make_deck()
        alpha = self._randomize_alpha(rng, alpha)
        suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in ALL_SUITS}
        weights = [
            math.exp(-alpha * BETLI_STRENGTH[c.rank]) * suit_mult[c.suit]
            for c in deck
        ]
        sol_hand, rest = _weighted_sample_without_replacement(deck, weights, 10, rng)
        def_hands, talon = _deal_rest(rest, rng)
        return [sol_hand] + def_hands, talon

    def hand_quality(self, hand):
        return sum(7 - BETLI_STRENGTH[c.rank] for c in hand) / 70.0

    def best_discard(self, gs, soloist, wrapper, game):
        hand = gs.hands[soloist]
        assert len(hand) == 12
        cdef = CONTRACT_DEFS["betli"]
        best_val = -float("inf")
        best_pair = (hand[0], hand[1])
        for c1, c2 in combinations(hand, 2):
            node = _make_eval_state(
                gs, soloist, trump=None, discards=(c1, c2),
                contract_def=cdef, is_piros=False, dealer=gs.dealer,
            )
            feats = game.encode_state(node, soloist)
            val = wrapper.predict_value(feats)
            if val > best_val:
                best_val = val
                best_pair = (c1, c2)
        return list(best_pair)

    def setup_node(self, gs, soloist, dealer):
        set_contract(gs, soloist, trump=None, betli=True)
        gs.training_mode = "betli"
        declare_all_marriages(gs, soloist_marriage_restrict=None)
        return UltiNode(
            gs=gs,
            known_voids=(frozenset(), frozenset(), frozenset()),
            bid_rank=0, is_red=False,
            contract_components=frozenset({"betli"}),
            dealer=dealer,
        )

    @property
    def kontra_key(self):
        return "betli"

    @property
    def training_mode(self):
        return "betli"


# ---------------------------------------------------------------------------
#  Ulti dojo
# ---------------------------------------------------------------------------

# Trump count distribution for biased ulti deals.
_ULTI_TRUMP_COUNTS = [3,    4,    5,    6,    7,    8]
_ULTI_TRUMP_WEIGHTS = [0.05, 0.40, 0.30, 0.15, 0.08, 0.02]


class UltiDojo(ContractDojo):

    def deal(self, rng, alpha, suit_sigma):
        deck = make_deck()
        alpha = self._randomize_alpha(rng, alpha)
        # Pick trump suit
        trump = rng.choice(ALL_SUITS)
        self._last_trump = trump

        # Choose trump count from distribution
        n_trump = rng.choices(_ULTI_TRUMP_COUNTS, _ULTI_TRUMP_WEIGHTS, k=1)[0]

        # --- Deal trump cards (biased toward high strength) ---
        trump_cards = [c for c in deck if c.suit == trump]
        n_trump = min(n_trump, len(trump_cards))
        # Weight by exp(alpha * normal_strength) — high cards preferred
        trump_weights = [
            math.exp(alpha * int(c.rank)) for c in trump_cards
        ]
        sol_trumps, remaining_trumps = _weighted_sample_without_replacement(
            trump_cards, trump_weights, n_trump, rng,
        )

        # --- Deal remaining soloist cards (hajtó + filler) ---
        n_rest = 10 - n_trump
        non_trump = [c for c in deck if c.suit != trump]
        # Use suit_sigma to concentrate into one or two suits (hajtó)
        suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in ALL_SUITS}
        suit_mult[trump] = 0.0  # already dealt trump cards
        rest_weights = [
            math.exp(alpha * int(c.rank)) * suit_mult[c.suit]
            for c in non_trump
        ]
        sol_rest, remaining_non_trump = _weighted_sample_without_replacement(
            non_trump, rest_weights, n_rest, rng,
        )

        sol_hand = sol_trumps + sol_rest
        all_remaining = remaining_trumps + remaining_non_trump
        def_hands, talon = _deal_rest(all_remaining, rng)
        return [sol_hand] + def_hands, talon

    def hand_quality(self, hand):
        # Quality based on trump count (normalised 3-8 → 0-1)
        # plus average strength of all cards
        trump = getattr(self, "_last_trump", None)
        if trump is None:
            return 0.5
        n_trump = sum(1 for c in hand if c.suit == trump)
        trump_score = min((n_trump - 3) / 5.0, 1.0)  # 3→0, 8→1
        avg_strength = sum(int(c.rank) for c in hand) / (7.0 * 10)  # 0-1
        return 0.6 * trump_score + 0.4 * avg_strength

    def best_discard(self, gs, soloist, wrapper, game):
        hand = gs.hands[soloist]
        assert len(hand) == 12
        trump = self._last_trump
        is_piros = (trump == Suit.HEARTS)
        cdef = CONTRACT_DEFS["ulti"]
        trump_7 = Card(trump, Rank.SEVEN)

        best_val = -float("inf")
        best_pair = (hand[0], hand[1])
        for c1, c2 in combinations(hand, 2):
            # Never discard the trump 7 (needed for ulti endgame)
            if c1 == trump_7 or c2 == trump_7:
                continue
            node = _make_eval_state(
                gs, soloist, trump=trump, discards=(c1, c2),
                contract_def=cdef, is_piros=is_piros, dealer=gs.dealer,
            )
            feats = game.encode_state(node, soloist)
            val = wrapper.predict_value(feats)
            if val > best_val:
                best_val = val
                best_pair = (c1, c2)
        return list(best_pair)

    def setup_node(self, gs, soloist, dealer):
        trump = self._last_trump
        is_piros = (trump == Suit.HEARTS)
        set_contract(gs, soloist, trump=trump, betli=False)
        gs.has_ulti = True
        gs.training_mode = "ulti"
        declare_all_marriages(gs, soloist_marriage_restrict=None)
        return UltiNode(
            gs=gs,
            known_voids=(frozenset(), frozenset(), frozenset()),
            bid_rank=0, is_red=is_piros,
            contract_components=frozenset({"parti", "ulti"}),
            dealer=dealer,
        )

    @property
    def kontra_key(self):
        return "ulti"

    @property
    def training_mode(self):
        return "ulti"


# ---------------------------------------------------------------------------
#  Durchmars (Duri) dojo
# ---------------------------------------------------------------------------


class DurchmarsDojo(ContractDojo):

    def deal(self, rng, alpha, suit_sigma):
        deck = make_deck()
        alpha = self._randomize_alpha(rng, alpha)
        suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in ALL_SUITS}
        # Bias toward HIGH betli-strength cards (soloist wants to win all tricks)
        weights = [
            math.exp(alpha * BETLI_STRENGTH[c.rank]) * suit_mult[c.suit]
            for c in deck
        ]
        sol_hand, rest = _weighted_sample_without_replacement(deck, weights, 10, rng)
        def_hands, talon = _deal_rest(rest, rng)
        return [sol_hand] + def_hands, talon

    def hand_quality(self, hand):
        return sum(BETLI_STRENGTH[c.rank] for c in hand) / 70.0

    def best_discard(self, gs, soloist, wrapper, game):
        hand = gs.hands[soloist]
        assert len(hand) == 12
        cdef = CONTRACT_DEFS["durchmars"]
        best_val = -float("inf")
        best_pair = (hand[0], hand[1])
        for c1, c2 in combinations(hand, 2):
            node = _make_eval_state(
                gs, soloist, trump=None, discards=(c1, c2),
                contract_def=cdef, is_piros=False, dealer=gs.dealer,
            )
            feats = game.encode_state(node, soloist)
            val = wrapper.predict_value(feats)
            if val > best_val:
                best_val = val
                best_pair = (c1, c2)
        return list(best_pair)

    def setup_node(self, gs, soloist, dealer):
        set_contract(gs, soloist, trump=None, betli=True)
        gs.training_mode = "durchmars"
        declare_all_marriages(gs, soloist_marriage_restrict=None)
        return UltiNode(
            gs=gs,
            known_voids=(frozenset(), frozenset(), frozenset()),
            bid_rank=0, is_red=False,
            contract_components=frozenset({"durchmars"}),
            dealer=dealer,
        )

    @property
    def kontra_key(self):
        return "durchmars"

    @property
    def training_mode(self):
        return "durchmars"


# ---------------------------------------------------------------------------
#  Dojo registry
# ---------------------------------------------------------------------------

DOJO_REGISTRY: dict[str, type[ContractDojo]] = {
    "betli": BetliDojo,
    "ulti": UltiDojo,
    "durchmars": DurchmarsDojo,
}


# ---------------------------------------------------------------------------
#  Play one dojo game (generic)
# ---------------------------------------------------------------------------

def play_one_dojo_game(
    game: UltiGame,
    wrapper,
    sol_player: HybridPlayer,
    def_player: HybridPlayer,
    rng: random.Random,
    alpha: float,
    suit_sigma: float,
    kontra: bool,
    dojo: ContractDojo,
) -> tuple[list[_Sample], float, float, list[Card]]:
    """Play one biased game, return (samples, quality, sol_eval, sol_hand_10).

    Each sample: (state_feats, action_mask, policy, reward, is_soloist).
    sol_eval: soloist value head prediction before play.
    sol_hand_10: soloist's 10-card hand after discard.
    """
    # 1. Biased deal
    hands, talon = dojo.deal(rng, alpha, suit_sigma)
    dealer = 2  # arbitrary; soloist is player 0
    soloist = 0

    quality = dojo.hand_quality(hands[soloist])

    # Build GameState with 10-card hands
    gs = _make_gs(hands, dealer, soloist)

    # 2. Pickup talon (hand goes 10 → 12) and smart discard
    pickup_talon(gs, soloist, talon)
    discards = dojo.best_discard(gs, soloist, wrapper, game)
    discard_talon(gs, discards)

    # Capture 10-card hand after discard (for hand evaluator training)
    sol_hand_10 = list(gs.hands[soloist])

    # 3. Set contract and build UltiNode
    node = dojo.setup_node(gs, soloist, dealer)

    # Soloist eval score (value head prediction before play)
    sol_feats = game.encode_state(node, soloist)
    sol_eval = wrapper.predict_value(sol_feats)

    # Kontra decision (simple: defenders kontra if their value > 0.4)
    if kontra:
        for def_p in range(3):
            if def_p == soloist:
                continue
            feats = game.encode_state(node, def_p)
            def_val = wrapper.predict_value(feats)
            if def_val > 0.4:
                node.component_kontras[dojo.kontra_key] = 1
                node.gs.kontra_level = 1
                break

    # 4. Play the game
    samples: list[_Sample] = []

    state = node
    while not game.is_terminal(state):
        player = game.current_player(state)
        is_sol = (player == soloist)
        hp = sol_player if is_sol else def_player

        pi, action, _sv = hp.choose_action_with_policy(state, player, rng)

        feats = game.encode_state(state, player)
        mask = game.legal_action_mask(state)
        samples.append((feats, mask, pi, 0.0, is_sol))

        state = game.apply(state, action)

    # 5. Label with terminal outcome
    labeled: list[_Sample] = []
    for feats, mask, pi, _, is_sol in samples:
        player_for_reward = soloist if is_sol else (1 if soloist != 1 else 2)
        reward = simple_outcome(state, player_for_reward)
        labeled.append((feats, mask, pi, reward, is_sol))

    return labeled, quality, sol_eval, sol_hand_10


# ---------------------------------------------------------------------------
#  Worker process support
# ---------------------------------------------------------------------------

# Per-worker globals (initialised once per process)
_W_GAME: UltiGame | None = None
_W_NET: UltiNet | None = None
_W_WRAPPER = None


def _init_worker(net_kwargs: dict) -> None:
    """Called once per worker process to create game + network."""
    global _W_GAME, _W_NET, _W_WRAPPER
    _W_GAME = UltiGame(restrictions=[])
    _W_NET = UltiNet(**net_kwargs)
    _W_NET.eval()
    _W_WRAPPER = make_wrapper(_W_NET, device="cpu")


def _play_batch_in_worker(
    args: tuple,
) -> list[tuple[list[_Sample], float, float, list[Card]]]:
    """Worker entry-point: play a batch of games with one weight load."""
    (weights, sol_cfg, def_cfg, game_seeds, alpha, suit_sigma,
     kontra, endgame_tricks, pimc_dets, solver_temp, contract) = args

    global _W_NET, _W_WRAPPER, _W_GAME

    # Load weights once per batch (not per game)
    _W_NET.load_state_dict(weights, strict=False)
    _W_NET.eval()
    _W_WRAPPER = make_wrapper(_W_NET, device="cpu")

    dojo = DOJO_REGISTRY[contract]()

    sol_player = HybridPlayer(
        _W_GAME, _W_WRAPPER, mcts_config=sol_cfg,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )
    def_player = HybridPlayer(
        _W_GAME, _W_WRAPPER, mcts_config=def_cfg,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=solver_temp,
    )

    results = []
    for seed in game_seeds:
        rng = random.Random(seed)
        results.append(play_one_dojo_game(
            _W_GAME, _W_WRAPPER, sol_player, def_player, rng,
            alpha, suit_sigma, kontra, dojo,
        ))
    return results


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

@dataclass
class DojoConfig:
    # Source / target
    source: str = "scout"
    save_as: str | None = None
    contract: str = "betli"

    # Training budget
    steps: int = 200
    games_per_step: int = 8
    train_steps: int = 50
    batch_size: int = 64
    buffer_size: int = 30_000

    # Learning rate
    lr_start: float = 5e-4
    lr_end: float = 1e-4

    # Dealing bias
    alpha: float = 0.5
    suit_sigma: float = 1.0

    # MCTS
    sol_sims: int = 40
    sol_dets: int = 2
    def_sims: int = 20
    def_dets: int = 2
    endgame_tricks: int = 6
    pimc_dets: int = 20
    solver_temp: float = 0.5

    # Kontra
    kontra: bool = True

    # Start from random weights (ignore checkpoint)
    from_scratch: bool = False

    # Freeze value head (only train policy)
    freeze_value: bool = False

    # Workers
    num_workers: int = 1

    # Device
    device: str = "cpu"
    seed: int = 42


def train_dojo(cfg: DojoConfig) -> None:
    """Run focused dojo training."""

    if cfg.contract not in DOJO_REGISTRY:
        print(f"Error: no dojo for contract '{cfg.contract}'. "
              f"Available: {', '.join(DOJO_REGISTRY)}")
        sys.exit(1)

    dojo = DOJO_REGISTRY[cfg.contract]()

    # ── Load model ────────────────────────────────────────────────
    game = UltiGame(restrictions=[])

    if cfg.from_scratch:
        tier = TIERS.get(cfg.source)
        if tier is None:
            print(f"Error: unknown tier '{cfg.source}' for --from-scratch. "
                  f"Available: {', '.join(TIERS)}")
            sys.exit(1)
        net = UltiNet(
            input_dim=game.state_dim,
            body_units=tier.body_units,
            body_layers=tier.body_layers,
            action_dim=game.action_space_size,
        )
        print(f"  Starting from random weights ({tier.body_units}×{tier.body_layers})")
    else:
        model_dir = Path(f"models/ulti/{cfg.source}/final/{cfg.contract}")
        model_pt = model_dir / "model.pt"
        if not model_pt.exists():
            print(f"Error: no model found at {model_dir}")
            sys.exit(1)

        cp = torch.load(model_pt, weights_only=False, map_location=cfg.device)
        net = UltiNet(
            input_dim=cp.get("input_dim", game.state_dim),
            body_units=cp.get("body_units", 256),
            body_layers=cp.get("body_layers", 4),
            action_dim=cp.get("action_dim", game.action_space_size),
        )
        net.load_state_dict(cp["model_state_dict"], strict=False)

    net.to(cfg.device)
    wrapper = make_wrapper(net, device=cfg.device)

    # ── BetliNet hand evaluator (betli dojo only) ───────────────
    betli_he = None
    if cfg.contract == "betli":
        from trickster.betli.hand_evaluator import BetliHandEvaluator
        from trickster.betli.model import BetliNet as BetliNetModel
        from trickster.betli.model import load_model as load_betli, save_model as save_betli

        betli_eval_path = Path(f"models/ulti/{cfg.source}/final/betli/betli_hand_eval.pt")
        if betli_eval_path.exists() and not cfg.from_scratch:
            betli_net = load_betli(betli_eval_path)
            print(f"  BetliNet: loaded from {betli_eval_path}")
        else:
            betli_net = BetliNetModel()
            print(f"  BetliNet: fresh weights (online training enabled)")
        betli_he = BetliHandEvaluator(betli_net, device=cfg.device)

    # ── Setup ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr_start)
    buf = ReplayBuffer(capacity=cfg.buffer_size, seed=cfg.seed)
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    sol_mcts = MCTSConfig(
        simulations=cfg.sol_sims,
        determinizations=cfg.sol_dets,
        use_value_head=True,
        use_policy_priors=True,
    )
    def_mcts = MCTSConfig(
        simulations=cfg.def_sims,
        determinizations=cfg.def_dets,
        use_value_head=True,
        use_policy_priors=True,
    )

    # Quality tier tracking
    TIERS_N = 5
    tier_wins = [0] * TIERS_N
    tier_total = [0] * TIERS_N
    tier_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    save_as = cfg.save_as or cfg.source

    # ── Header ────────────────────────────────────────────────────
    print(f"  Dojo: {cfg.contract} training")
    print(f"  Source: {cfg.source} → Save: {save_as}")
    print(f"  Steps: {cfg.steps} × {cfg.games_per_step} games")
    print(f"  Alpha: {cfg.alpha}, Suit σ: {cfg.suit_sigma}")
    print(f"  LR: {cfg.lr_start:.1e} → {cfg.lr_end:.1e}")
    if cfg.freeze_value:
        print(f"  Freeze: value head (policy-only training)")
    if cfg.num_workers > 1:
        print(f"  Self-play: {cfg.num_workers} workers (process pool)")
    else:
        print(f"  Self-play: sequential")
    print(f"  Device: {cfg.device}")
    print()

    def _fmt_time(s: float) -> str:
        if s < 60:
            return f"{s:.0f}s"
        m, sec = divmod(int(s), 60)
        if m < 60:
            return f"{m}m {sec:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"

    total_games = 0
    total_wins = 0
    total_kontras = 0
    t0 = time.perf_counter()

    # ── Worker pool ───────────────────────────────────────────────
    executor = None
    body_layers = len([m for m in net.backbone if isinstance(m, torch.nn.Linear)])
    net_kwargs = {
        "input_dim": net.input_dim,
        "body_units": net.body_units,
        "body_layers": body_layers,
        "action_dim": net.action_dim,
    }

    if cfg.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            initializer=_init_worker,
            initargs=(net_kwargs,),
        )

    try:
        for step in range(1, cfg.steps + 1):
            # ── Learning rate schedule ───────────────────────────────
            frac = (step - 1) / max(cfg.steps - 1, 1)
            lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ── Generate games ───────────────────────────────────────
            net.eval()

            step_wins = 0
            step_qualities: list[float] = []
            step_kontras = 0
            step_samples = 0

            def _collect_result(
                samples: list[_Sample],
                quality: float,
                sol_hand_10: list[Card] | None = None,
            ) -> None:
                nonlocal step_wins, step_kontras, step_samples

                step_qualities.append(quality)

                sol_reward = next(r for _, _, _, r, is_sol in samples if is_sol)
                won = sol_reward > 0
                if won:
                    step_wins += 1

                tier_idx = min(int(quality * TIERS_N), TIERS_N - 1)
                tier_total[tier_idx] += 1
                if won:
                    tier_wins[tier_idx] += 1

                if abs(sol_reward) > 1.1:
                    step_kontras += 1

                # Feed BetliNet hand evaluator
                if betli_he is not None and sol_hand_10 is not None:
                    betli_he.record_outcome(sol_hand_10, sol_reward)

                for feats, mask, pi, reward, is_sol in samples:
                    buf.push(feats, mask, pi, reward, is_sol)
                    step_samples += 1

            if executor is not None:
                # --- Parallel self-play (batched per worker) ---
                all_weights = {k: v.cpu() for k, v in net.state_dict().items()}
                all_seeds = [cfg.seed + step * 1000 + g
                             for g in range(cfg.games_per_step)]

                batches = []
                base = 0
                per_w = cfg.games_per_step // cfg.num_workers
                extra = cfg.games_per_step % cfg.num_workers
                for w in range(cfg.num_workers):
                    n = per_w + (1 if w < extra else 0)
                    if n == 0:
                        continue
                    batches.append((
                        all_weights, sol_mcts, def_mcts,
                        all_seeds[base:base + n],
                        cfg.alpha, cfg.suit_sigma, cfg.kontra,
                        cfg.endgame_tricks, cfg.pimc_dets, cfg.solver_temp,
                        cfg.contract,
                    ))
                    base += n

                for batch_results in executor.map(_play_batch_in_worker, batches):
                    for samples, quality, _sol_eval, sol_hand_10 in batch_results:
                        _collect_result(samples, quality, sol_hand_10)
            else:
                # --- Sequential self-play ---
                wrapper = make_wrapper(net, device=cfg.device)
                sol_player = HybridPlayer(
                    game, wrapper, mcts_config=sol_mcts,
                    endgame_tricks=cfg.endgame_tricks,
                    pimc_determinizations=cfg.pimc_dets,
                    solver_temperature=cfg.solver_temp,
                )
                def_player = HybridPlayer(
                    game, wrapper, mcts_config=def_mcts,
                    endgame_tricks=cfg.endgame_tricks,
                    pimc_determinizations=cfg.pimc_dets,
                    solver_temperature=cfg.solver_temp,
                )
                for g in range(cfg.games_per_step):
                    samples, quality, _sol_eval, sol_hand_10 = play_one_dojo_game(
                        game, wrapper, sol_player, def_player, rng,
                        cfg.alpha, cfg.suit_sigma, cfg.kontra, dojo,
                    )
                    _collect_result(samples, quality, sol_hand_10)

            total_games += cfg.games_per_step
            total_wins += step_wins
            total_kontras += step_kontras

            # ── SGD ──────────────────────────────────────────────────
            v_loss_avg = 0.0
            p_loss_avg = 0.0
            sgd_count = 0

            if len(buf) >= cfg.batch_size:
                net.train()
                for _ in range(cfg.train_steps):
                    states, masks, policies, rewards, is_sol, _on_pol = buf.sample(
                        cfg.batch_size, np_rng,
                    )
                    s_t = torch.from_numpy(states).float().to(cfg.device)
                    m_t = torch.from_numpy(masks).bool().to(cfg.device)
                    pi_t = torch.from_numpy(policies).float().to(cfg.device)
                    z_t = torch.from_numpy(rewards).float().to(cfg.device)
                    is_sol_t = torch.from_numpy(is_sol).bool().to(cfg.device)

                    log_probs, values = net.forward_dual(s_t, m_t, is_sol_t)
                    policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
                    if cfg.freeze_value:
                        value_loss = torch.tensor(0.0)
                        loss = policy_loss
                    else:
                        value_loss = F.huber_loss(values, z_t, delta=1.0)
                        loss = value_loss + policy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimizer.step()

                    v_loss_avg += value_loss.item()
                    p_loss_avg += policy_loss.item()
                    sgd_count += 1

            if sgd_count > 0:
                v_loss_avg /= sgd_count
                p_loss_avg /= sgd_count

            # ── BetliNet SGD ─────────────────────────────────────────
            if betli_he is not None:
                betli_he.train_step()

            # ── Progress bar ─────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            gps = total_games / elapsed if elapsed > 0 else 0
            overall_wr = total_wins / total_games if total_games > 0 else 0

            frac = step / cfg.steps
            bar_w = 30
            filled = int(bar_w * frac)
            bar = "█" * filled + "░" * (bar_w - filled)
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0

            print(
                f"\r  {bar} {frac*100:5.1f}%  "
                f"win={overall_wr:.0%}  "
                f"v={v_loss_avg:.3f} p={p_loss_avg:.3f}  "
                f"{gps:.1f} g/s  "
                f"{_fmt_time(elapsed)} / {_fmt_time(eta)}   ",
                end="", flush=True,
            )

            if step == cfg.steps:
                print()

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    # ── Save ──────────────────────────────────────────────────────
    out_dir = Path(f"models/ulti/{save_as}/final/{cfg.contract}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": net.state_dict(),
        "body_units": net.body_units,
        "body_layers": body_layers,
        "input_dim": game.state_dim,
        "action_dim": game.action_space_size,
        "training_mode": dojo.training_mode,
        "method": "dojo",
        "dojo_contract": cfg.contract,
        "dojo_alpha": cfg.alpha,
        "dojo_suit_sigma": cfg.suit_sigma,
        "dojo_steps": cfg.steps,
        "dojo_total_games": total_games,
    }, out_dir / "model.pt")

    # Save BetliNet if trained
    if betli_he is not None:
        betli_eval_out = out_dir / "betli_hand_eval.pt"
        save_betli(betli_he.net, betli_eval_out)

    elapsed = time.perf_counter() - t0
    overall_wr = total_wins / total_games if total_games > 0 else 0

    print()
    print(f"  ┌─ DOJO COMPLETE ─────────────────────────────────")
    print(f"  │  Contract:   {cfg.contract}")
    print(f"  │  Games:      {total_games:,}  ({_fmt_time(elapsed)}, {total_games / elapsed:.1f} g/s)")
    print(f"  │  Win rate:   {overall_wr:.1%}  (kontras: {total_kontras})")
    print(f"  │")
    print(f"  │  Quality tiers:")
    for i in range(TIERS_N):
        if tier_total[i] > 0:
            tw = tier_wins[i] / tier_total[i]
            print(f"  │    {tier_labels[i]}: {tw:>4.0%}  ({tier_wins[i]}/{tier_total[i]})")
    print(f"  │")
    print(f"  │  Saved: {out_dir}/model.pt")
    if betli_he is not None:
        print(f"  │  BetliNet: {out_dir}/betli_hand_eval.pt "
              f"(buffer: {betli_he.buffer_size} samples)")
    print(f"  └─────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dojo — focused contract training with biased dealing",
    )
    parser.add_argument("model", help="Source model (e.g. trinity, bishop)")
    parser.add_argument("--contract", default="betli",
                        help=f"Contract to train (available: {', '.join(DOJO_REGISTRY)})")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--games-per-step", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--alpha", type=float, default=0.5, help="Rank bias strength")
    parser.add_argument("--suit-sigma", type=float, default=1.0, help="Suit concentration variance")
    parser.add_argument("--lr-start", type=float, default=5e-4)
    parser.add_argument("--lr-end", type=float, default=1e-4)
    parser.add_argument("--sol-sims", type=int, default=40)
    parser.add_argument("--sol-dets", type=int, default=2)
    parser.add_argument("--def-sims", type=int, default=20)
    parser.add_argument("--def-dets", type=int, default=2)
    parser.add_argument("--endgame-tricks", type=int, default=6)
    parser.add_argument("--pimc-dets", type=int, default=20)
    parser.add_argument("--solver-temp", type=float, default=0.5)
    parser.add_argument("--kontra", action="store_true", default=True)
    parser.add_argument("--no-kontra", dest="kontra", action="store_false")
    parser.add_argument("--from-scratch", action="store_true", default=False,
                        help="Start from random weights (uses tier architecture)")
    parser.add_argument("--freeze-value", action="store_true", default=False,
                        help="Freeze value head, only train policy")
    parser.add_argument("--save-as", default=None, help="Target model name (default: same as source)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Detect device
    if args.device:
        device = args.device
    else:
        tier = TIERS.get(args.model)
        body_units = tier.body_units if tier else 256
        body_layers = tier.body_layers if tier else 4
        device = auto_device(body_units, body_layers)

    cfg = DojoConfig(
        source=args.model,
        save_as=args.save_as,
        contract=args.contract,
        steps=args.steps,
        games_per_step=args.games_per_step,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        alpha=args.alpha,
        suit_sigma=args.suit_sigma,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
        sol_sims=args.sol_sims,
        sol_dets=args.sol_dets,
        def_sims=args.def_sims,
        def_dets=args.def_dets,
        endgame_tricks=args.endgame_tricks,
        pimc_dets=args.pimc_dets,
        solver_temp=args.solver_temp,
        kontra=args.kontra,
        from_scratch=args.from_scratch,
        freeze_value=args.freeze_value,
        num_workers=args.workers,
        device=device,
        seed=args.seed,
    )

    train_dojo(cfg)


if __name__ == "__main__":
    main()
