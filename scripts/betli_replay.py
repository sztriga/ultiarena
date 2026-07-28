#!/usr/bin/env python3
"""Play out a single good betli hand with trinity and print every trick."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.bidding.evaluator import _make_eval_state
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame
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
    declare_all_marriages,
    discard_talon,
    next_player,
    pickup_talon,
    set_contract,
)
from trickster.games.ulti.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig
from trickster.model import UltiNet, make_wrapper
from trickster.training.model_io import auto_device


def biased_betli_deal(rng, alpha=0.3, suit_sigma=1.5):
    deck = make_deck()
    suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in ALL_SUITS}
    weights = []
    for card in deck:
        w = math.exp(-alpha * BETLI_STRENGTH[card.rank]) * suit_mult[card.suit]
        weights.append(w)

    remaining = list(range(len(deck)))
    remaining_weights = list(weights)
    sol_indices = []
    for _ in range(10):
        total = sum(remaining_weights)
        r = rng.random() * total
        cum = 0.0
        chosen = 0
        for i, w in enumerate(remaining_weights):
            cum += w
            if cum >= r:
                chosen = i
                break
        sol_indices.append(remaining[chosen])
        remaining.pop(chosen)
        remaining_weights.pop(chosen)

    sol_hand = [deck[i] for i in sol_indices]
    rest = [deck[i] for i in remaining]
    rng.shuffle(rest)
    hands = [sol_hand, rest[:10], rest[10:20]]
    talon = rest[20:22]
    return hands, talon


def hand_quality(hand):
    return sum(7 - BETLI_STRENGTH[c.rank] for c in hand) / 70.0


def best_betli_discard(gs, soloist, wrapper, game):
    from itertools import combinations
    hand = gs.hands[soloist]
    cdef = CONTRACT_DEFS["betli"]
    best_val = -float("inf")
    best_pair = (hand[0], hand[1])
    for c1, c2 in combinations(hand, 2):
        node = _make_eval_state(gs, soloist, trump=None, discards=(c1, c2),
                                contract_def=cdef, is_piros=False, dealer=gs.dealer)
        feats = game.encode_state(node, soloist)
        val = wrapper.predict_value(feats)
        if val > best_val:
            best_val = val
            best_pair = (c1, c2)
    return list(best_pair), best_val


def _run_auction_mode(game, net, wrapper, mcts_cfg, base_seed):
    """Deal hands normally and find cases where betli gets selected via auction."""
    import copy
    import itertools
    import numpy as np
    from trickster.bidding.auction_runner import evaluate_pickup, run_auction
    from trickster.bidding.evaluator import evaluate_all_contracts
    from trickster.games.ulti.game import deal
    from trickster.model import make_wrapper as _mw

    # Build wrappers dict (contract_key → wrapper)
    wrappers = {}
    for ck in ["parti", "ulti", "40-100", "betli"]:
        model_pt = Path(f"models/ulti/trinity/final/{ck}/model.pt")
        if model_pt.exists():
            cp = torch.load(model_pt, weights_only=False, map_location="cpu")
            n = UltiNet(
                input_dim=cp.get("input_dim", game.state_dim),
                body_units=cp.get("body_units", 256),
                body_layers=cp.get("body_layers", 4),
                action_dim=cp.get("action_dim", game.action_space_size),
            )
            n.load_state_dict(cp["model_state_dict"], strict=False)
            n.eval()
            wrappers[ck] = _mw(n, device="cpu")

    print(f"  Loaded wrappers: {list(wrappers.keys())}")
    print()

    from trickster.training.model_io import load_talon_prior
    tp = load_talon_prior("trinity")

    betli_played = 0
    betli_wins = 0
    rng = random.Random(base_seed)

    for deal_idx in range(2000):
        seed = base_seed + deal_idx
        gs, talon = deal(seed=seed, dealer=deal_idx % 3)
        dealer = deal_idx % 3

        # Run the full auction — does it actually pick betli?
        seat_wrappers = [wrappers, wrappers, wrappers]
        result = run_auction(
            gs, talon, dealer, seat_wrappers,
            pickup_quantile=0.5,
            rng=random.Random(seed),
            talon_prior=tp,
        )
        if result.bid is None or result.bid.contract_key != "betli":
            continue

        # Betli was actually selected by the auction!
        soloist = result.soloist
        betli_played += 1

        print(f"  ═══ Deal {deal_idx}, Soloist P{soloist} — BETLI (auction selected) ═══")
        print(f"  10-card hand: {fmt_hand(gs.hands[soloist])}")
        print(f"  Talon: {' '.join(c.short() for c in talon)}")

        from trickster.bidding.auction_runner import setup_bid_game, extract_player_bid_ranks
        state = setup_bid_game(
            game, gs, soloist, dealer, result.bid,
            initial_bidder=result.initial_bidder,
            player_bid_ranks=extract_player_bid_ranks(result.auction),
        )

        print(f"  Final hand: {fmt_hand(state.gs.hands[soloist])}")

        # Danger analysis
        final_hand = state.gs.hands[soloist]
        suits_in_hand = {}
        for c in final_hand:
            suits_in_hand.setdefault(c.suit, []).append(c)
        for suit, cards in suits_in_hand.items():
            max_str = max(BETLI_STRENGTH[c.rank] for c in cards)
            n_cards = len(cards)
            if max_str >= 5:
                danger_cards = [c for c in cards if BETLI_STRENGTH[c.rank] >= 5]
                print(f"    {suit.name}: {n_cards} cards, DANGER — {' '.join(c.short() for c in danger_cards)}")
            elif max_str >= 3 and n_cards <= 2:
                print(f"    {suit.name}: {n_cards} cards, RISK (thin, max str {max_str})")
            else:
                print(f"    {suit.name}: ok ({n_cards} cards, max str {max_str})")

        # Show defender hands
        for dp in range(3):
            if dp != soloist:
                print(f"  DEF{dp}: {fmt_hand(state.gs.hands[dp])}")

        # Play it out
        sol_player = HybridPlayer(game, wrappers.get("betli", wrapper),
                                  mcts_config=mcts_cfg,
                                  endgame_tricks=6, pimc_determinizations=20,
                                  solver_temperature=0.1)
        def_player = HybridPlayer(game, wrappers.get("betli", wrapper),
                                  mcts_config=mcts_cfg,
                                  endgame_tricks=6, pimc_determinizations=20,
                                  solver_temperature=0.1)

        trick_buf = []
        trick_no = 0
        print(f"  ┌─ PLAY ──────────────────────────────────────────")
        while not game.is_terminal(state):
            p = game.current_player(state)
            hp = sol_player if p == soloist else def_player
            actions = game.legal_actions(state)
            action = actions[0] if len(actions) <= 1 else hp.choose_action(state, p, rng)
            role = "SOL" if p == soloist else f"DEF{p}"
            trick_buf.append((p, role, action))
            state = game.apply(state, action)
            if len(trick_buf) == 3:
                trick_no += 1
                cards_str = "  ".join(f"{role}:{c.short()}" for _, role, c in trick_buf)
                history = state.gs.trick_history
                winner = history[trick_no - 1][1] if history and len(history) >= trick_no else state.gs.leader
                winner_role = "SOL" if winner == soloist else f"DEF{winner}"
                sol_marker = " *** SOLOIST TOOK TRICK ***" if winner == soloist else ""
                print(f"  │  Trick {trick_no:2d}: {cards_str}  → {winner_role}{sol_marker}")
                trick_buf = []

        print(f"  └─────────────────────────────────────────────────")
        from trickster.games.ulti.game import soloist_lost_betli, soloist_tricks
        lost = soloist_lost_betli(state.gs)
        n_tricks = soloist_tricks(state.gs)
        result_str = "LOSS" if lost else "WIN"
        if not lost:
            betli_wins += 1
        print(f"  Result: {result_str} (soloist took {n_tricks} tricks)")
        print()

        if betli_played >= 10:
            break

    print(f"  Summary: {betli_played} betli games found in {deal_idx + 1} deals, "
          f"{betli_wins} wins ({betli_wins * 100 // max(1, betli_played)}%)")


def fmt_hand(hand):
    by_suit = {}
    for c in sorted(hand, key=lambda c: (c.suit.value, BETLI_STRENGTH[c.rank])):
        by_suit.setdefault(c.suit.name[0], []).append(c.short())
    return "  ".join(f"{s}: {' '.join(cards)}" for s, cards in by_suit.items())


def main():
    # Load trinity betli model
    model_pt = Path("models/ulti/trinity/final/betli/model.pt")
    if not model_pt.exists():
        print(f"No model at {model_pt}")
        sys.exit(1)

    cp = torch.load(model_pt, weights_only=False, map_location="cpu")
    game = UltiGame()
    net = UltiNet(
        input_dim=cp.get("input_dim", game.state_dim),
        body_units=cp.get("body_units", 256),
        body_layers=cp.get("body_layers", 4),
        action_dim=cp.get("action_dim", game.action_space_size),
    )
    net.load_state_dict(cp["model_state_dict"], strict=False)
    net.eval()
    wrapper = make_wrapper(net, device="cpu")

    mcts_cfg = MCTSConfig(simulations=80, determinizations=4,
                          use_value_head=True, use_policy_priors=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "biased"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else random.randint(0, 999999)

    if mode == "auction":
        # Simulate real auction — deal normally and see if betli gets picked
        print(f"  Mode: auction (real deals, betli pickup evaluation)")
        print(f"  Seed: {seed}")
        _run_auction_mode(game, net, wrapper, mcts_cfg, seed)
        return

    rng = random.Random(seed)
    print(f"  Mode: biased deal")
    print(f"  Seed: {seed}")

    # Find a good betli hand (quality >= 0.7)
    for attempt in range(200):
        hands, talon = biased_betli_deal(rng, alpha=0.3, suit_sigma=1.5)
        q = hand_quality(hands[0])
        if q >= 0.7:
            break
    else:
        print("Could not find a quality >= 0.7 hand in 200 attempts")
        sys.exit(1)

    soloist = 0
    dealer = 2

    print(f"  Hand quality: {q:.2f}")
    print(f"  Soloist (P0): {fmt_hand(hands[0])}")
    print(f"  Def 1  (P1): {fmt_hand(hands[1])}")
    print(f"  Def 2  (P2): {fmt_hand(hands[2])}")
    print(f"  Talon:        {' '.join(c.short() for c in talon)}")
    print()

    # Pickup + discard
    gs = GameState(
        hands=hands, trump=None, betli=False, soloist=soloist,
        dealer=dealer, captured=[[], [], []], scores=[0, 0, 0],
        leader=next_player(dealer), trick_no=0,
        trick_cards=[], last_trick=None,
    )
    pickup_talon(gs, soloist, talon)
    print(f"  After pickup (12): {fmt_hand(gs.hands[0])}")

    discards, disc_val = best_betli_discard(gs, soloist, wrapper, game)
    print(f"  Discards: {' '.join(c.short() for c in discards)}  (value head: {disc_val:+.3f})")
    discard_talon(gs, discards)
    print(f"  Final hand (10):   {fmt_hand(gs.hands[0])}")
    print()

    # Set contract
    set_contract(gs, soloist, trump=None, betli=True)
    gs.training_mode = "betli"
    declare_all_marriages(gs, soloist_marriage_restrict=None)

    from trickster.games.ulti.adapter import EMPTY_VOIDS, UltiNode
    node = UltiNode(
        gs=gs,
        known_voids=(frozenset(), frozenset(), frozenset()),
        bid_rank=0, is_red=False,
        contract_components=frozenset({"betli"}),
        dealer=dealer,
    )

    # Create players
    sol_player = HybridPlayer(game, wrapper, mcts_config=mcts_cfg,
                              endgame_tricks=6, pimc_determinizations=20,
                              solver_temperature=0.1)
    def_player = HybridPlayer(game, wrapper, mcts_config=mcts_cfg,
                              endgame_tricks=6, pimc_determinizations=20,
                              solver_temperature=0.1)

    # Play out the game, printing each trick
    state = node
    trick_cards_buf = []
    trick_no = 0

    print("  ┌─ PLAY ──────────────────────────────────────────")
    while not game.is_terminal(state):
        player = game.current_player(state)
        is_sol = player == soloist
        hp = sol_player if is_sol else def_player

        actions = game.legal_actions(state)
        if len(actions) <= 1:
            action = actions[0]
        else:
            action = hp.choose_action(state, player, rng)

        # Action IS a Card object
        card = action
        role = "SOL" if is_sol else f"DEF{player}"
        trick_cards_buf.append((player, role, card))

        state = game.apply(state, action)

        # Check if trick is complete (3 cards played)
        if len(trick_cards_buf) == 3:
            trick_no += 1
            cards_str = "  ".join(f"{role}:{c.short()}" for _, role, c in trick_cards_buf)

            # Winner from trick_history
            history = state.gs.trick_history
            if history and len(history) >= trick_no:
                _, winner = history[trick_no - 1]
            else:
                winner = state.gs.leader  # fallback

            winner_role = "SOL" if winner == soloist else f"DEF{winner}"
            sol_marker = " *** SOLOIST TOOK TRICK ***" if winner == soloist else ""
            print(f"  │  Trick {trick_no:2d}: {cards_str}  → {winner_role}{sol_marker}")
            trick_cards_buf = []

    print("  └─────────────────────────────────────────────────")
    print()

    # Result
    from trickster.games.ulti.game import soloist_lost_betli, soloist_tricks
    lost = soloist_lost_betli(state.gs)
    n_tricks = soloist_tricks(state.gs)
    print(f"  Result: {'LOSS' if lost else 'WIN'} (soloist took {n_tricks} tricks)")


if __name__ == "__main__":
    main()
