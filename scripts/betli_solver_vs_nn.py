#!/usr/bin/env python3
"""Betli experiment: pure PIMC solver vs NN (HybridPlayer).

Plays biased betli deals and compares:
  1. Solver soloist vs NN defenders
  2. NN soloist vs Solver defenders
  3. Solver soloist vs Solver defenders (upper bound)

Usage:
    python3 scripts/betli_solver_vs_nn.py trinity --games 200
    python3 scripts/betli_solver_vs_nn.py rook --games 500 --pimc-dets 40
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import ALL_SUITS, BETLI_STRENGTH, Card, make_deck
from trickster.games.ulti.game import (
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
)
from trickster.games.ulti.hybrid import HybridPlayer, _solve_root
from trickster.games.ulti.rewards import simple_outcome
from trickster.mcts import MCTSConfig
from trickster.model import make_wrapper
from trickster.training.model_io import load_net
from trickster.training.tiers import TIERS

# Re-use dojo helpers
from dojo import (
    BetliDojo,
    _deal_rest,
    _make_gs,
    _weighted_sample_without_replacement,
    play_one_dojo_game,
)


# ---------------------------------------------------------------------------
#  PIMC solver player (full game, no NN)
# ---------------------------------------------------------------------------


class SolverPlayer:
    """Pure PIMC solver — determinize + exact solve from trick 0."""

    def __init__(self, game: UltiGame, num_dets: int = 30):
        self.game = game
        self.num_dets = num_dets

    def choose_action(
        self, state: UltiNode, player: int, rng: random.Random,
    ) -> Card:
        legal = self.game.legal_actions(state)
        if len(legal) <= 1:
            return legal[0]

        totals = {c: 0.0 for c in legal}
        counts = {c: 0 for c in legal}

        for _ in range(self.num_dets):
            det = self.game.determinize(state, player, rng)
            vals = _solve_root(det.gs, contract="betli")
            for card in legal:
                if card in vals:
                    totals[card] += vals[card]
                    counts[card] += 1

        avg = {c: totals[c] / max(1, counts[c]) for c in legal}

        if player == state.gs.soloist:
            return max(avg, key=avg.__getitem__)
        else:
            return min(avg, key=avg.__getitem__)

    def choose_action_with_policy(
        self, state: UltiNode, player: int, rng: random.Random,
    ):
        """Compatibility shim for play_one_game."""
        action = self.choose_action(state, player, rng)
        action_space = self.game.action_space_size
        pi = np.zeros(action_space, dtype=np.float64)
        pi[self.game.action_to_index(action)] = 1.0
        return pi, action, None


# ---------------------------------------------------------------------------
#  Play one game with configurable soloist/defender players
# ---------------------------------------------------------------------------


def play_one_game(
    game: UltiGame,
    sol_player,
    def_player,
    rng: random.Random,
    alpha: float = 0.5,
    suit_sigma: float = 1.0,
) -> tuple[bool, float, int]:
    """Play one biased betli game. Returns (soloist_won, reward, tricks_survived)."""
    dojo = BetliDojo()

    # Deal
    hands, talon = dojo.deal(rng, alpha, suit_sigma)
    dealer = 2
    soloist = 0
    gs = _make_gs(hands, dealer, soloist)

    # Pickup and random discard (no NN discard — keep it fair)
    pickup_talon(gs, soloist, talon)
    discards = rng.sample(gs.hands[soloist], 2)
    discard_talon(gs, discards)

    # Setup
    set_contract(gs, soloist, trump=None, betli=True)
    gs.training_mode = "betli"
    declare_all_marriages(gs)

    state = UltiNode(
        gs=gs,
        known_voids=(frozenset(), frozenset(), frozenset()),
        bid_rank=0, is_red=False,
        contract_components=frozenset({"betli"}),
        dealer=dealer,
    )

    # Play
    while not game.is_terminal(state):
        player = game.current_player(state)
        is_sol = (player == soloist)
        hp = sol_player if is_sol else def_player

        pi, action, _sv = hp.choose_action_with_policy(state, player, rng)
        state = game.apply(state, action)

    # Result
    from trickster.games.ulti.game import soloist_lost_betli, soloist_tricks
    sol_won = not soloist_lost_betli(state.gs)
    tricks = soloist_tricks(state.gs)
    survived = state.gs.trick_no - tricks

    return sol_won, survived


def main():
    parser = argparse.ArgumentParser(description="Betli: PIMC solver vs NN")
    parser.add_argument("model", help="Model source (e.g. trinity, rook)")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--pimc-dets", type=int, default=30,
                        help="PIMC determinizations for solver")
    parser.add_argument("--nn-sims", type=int, default=60,
                        help="MCTS simulations for NN player")
    parser.add_argument("--nn-endgame", type=int, default=6,
                        help="Endgame tricks for NN hybrid player")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dealing bias strength")
    parser.add_argument("--suit-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load NN model
    model_path = Path(f"models/ulti/{args.model}/final/betli")
    net = load_net(model_path, device="cpu")
    if net is None:
        print(f"Error: no betli model at {model_path}")
        sys.exit(1)
    net.eval()
    wrapper = make_wrapper(net, device="cpu")

    game = UltiGame(restrictions=[])

    # Create players
    solver = SolverPlayer(game, num_dets=args.pimc_dets)

    nn_mcts = MCTSConfig(
        simulations=args.nn_sims,
        determinizations=2,
        use_value_head=True,
        use_policy_priors=True,
    )
    nn_player = HybridPlayer(
        game, wrapper, mcts_config=nn_mcts,
        endgame_tricks=args.nn_endgame,
        pimc_determinizations=20,
        solver_temperature=0.5,
    )

    # Configurations to test
    configs = [
        ("Solver sol vs NN def", solver, nn_player),
        ("NN sol vs Solver def", nn_player, solver),
        ("Solver sol vs Solver def", solver, solver),
    ]

    print(f"\n  Betli: PIMC Solver vs {args.model} NN")
    print(f"  Games: {args.games} per config")
    print(f"  Solver: {args.pimc_dets} PIMC dets (full game)")
    print(f"  NN: {args.nn_sims} MCTS sims, endgame={args.nn_endgame}")
    print(f"  Alpha: {args.alpha}, Suit σ: {args.suit_sigma}")
    print()

    for label, sol_p, def_p in configs:
        wins = 0
        total_survived = 0
        t0 = time.perf_counter()

        for g in range(args.games):
            rng = random.Random(args.seed + g)
            won, survived = play_one_game(
                game, sol_p, def_p, rng,
                alpha=args.alpha, suit_sigma=args.suit_sigma,
            )
            if won:
                wins += 1
            total_survived += survived

            # Progress
            if (g + 1) % 20 == 0 or g == args.games - 1:
                elapsed = time.perf_counter() - t0
                wr = wins / (g + 1)
                avg_surv = total_survived / (g + 1)
                gps = (g + 1) / elapsed
                print(
                    f"\r  {label}: {g+1}/{args.games}  "
                    f"win={wr:.0%}  avg_survived={avg_surv:.1f}  "
                    f"({gps:.1f} g/s)   ",
                    end="", flush=True,
                )

        elapsed = time.perf_counter() - t0
        wr = wins / args.games
        avg_surv = total_survived / args.games
        print(
            f"\r  {label}: "
            f"win={wr:.1%} ({wins}/{args.games})  "
            f"avg_survived={avg_surv:.1f}/10  "
            f"({elapsed:.0f}s)                    "
        )

    print()


if __name__ == "__main__":
    main()
