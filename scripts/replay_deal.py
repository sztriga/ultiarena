#!/usr/bin/env python3
"""Replay a specific deal seed with full auction + play diagnostics.

Shows the dealt hands, auction flow with evaluation details,
and trick-by-trick play. Uses run_auction directly with an on_step
callback to guarantee identical RNG flow as betli_pickup_diag.py.

Usage:
    python scripts/replay_deal.py trinity 48
    python scripts/replay_deal.py rook 45 --speed deep
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from trickster.bidding.auction_runner import (
    AuctionResult,
    extract_player_bid_ranks,
    run_auction,
    setup_bid_game,
)
from trickster.bidding.registry import BID_TO_CONTRACT
from trickster.games.ulti.adapter import UltiGame
from trickster.games.ulti.cards import BETLI_STRENGTH, Card, Suit, make_deck
from trickster.games.ulti.game import GameState, next_player
from trickster.games.ulti.rewards import simple_outcome
from trickster.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig
from trickster.training.model_io import load_talon_prior, load_wrappers


# ---------------------------------------------------------------------------
#  Formatting helpers
# ---------------------------------------------------------------------------

def fmt_hand(hand: list[Card]) -> str:
    by_suit: dict[Suit, list[Card]] = {}
    for c in hand:
        by_suit.setdefault(c.suit, []).append(c)
    parts = []
    for suit in Suit:
        if suit in by_suit:
            cards = sorted(by_suit[suit], key=lambda c: c.rank.value)
            parts.append(" ".join(c.short() for c in cards))
    return "  ".join(parts)


def betli_quality(hand: list[Card]) -> float:
    return sum(7 - BETLI_STRENGTH[c.rank] for c in hand) / (7.0 * len(hand))


# ---------------------------------------------------------------------------
#  Auction step logger
# ---------------------------------------------------------------------------

def make_step_logger() -> tuple[list[dict], callable]:
    """Return (log_list, callback) for run_auction's on_step."""
    log: list[dict] = []
    def callback(info: dict) -> None:
        log.append(info)
    return log, callback


def print_auction_log(log: list[dict]) -> None:
    """Pretty-print the captured auction steps."""
    for i, step in enumerate(log, 1):
        player = step["player"]

        if step["type"] == "bid":
            bid_obj = step["bid"]
            discards = step["discards"]
            ev = step["eval"]
            all_evals = step["all_evals"]
            hand = step["hand"]

            bid_label = bid_obj.label() if hasattr(bid_obj, "label") else str(bid_obj)
            print(f"  [{i}] P{player}: BID {bid_label}")
            print(f"       12-card hand: {fmt_hand(hand)}")

            if all_evals:
                print(f"       Evaluations (top 5):")
                for ev_item in all_evals[:5]:
                    prefix = "p." if ev_item.is_piros else ""
                    print(f"         {prefix}{ev_item.contract_key}: "
                          f"mean={ev_item.game_pts:+.3f}  "
                          f"best_discard={ev_item.best_discard.game_pts:+.3f}  "
                          f"val={ev_item.best_discard.value:+.3f}  "
                          f"discard=[{', '.join(c.short() for c in ev_item.best_discard.discard)}]")

            print(f"       → discard: [{', '.join(c.short() for c in discards)}]")
            post_hand = [c for c in hand if c not in discards]
            print(f"       10-card hand: {fmt_hand(post_hand)}")

        elif step["type"] == "pickup_decision":
            hand = step["hand"]
            talon = step["talon"]
            current_bid = step["current_bid"]
            holder = step["holder"]
            pe = step["pickup_eval"]
            picked_up = step["picked_up"]

            bid_label = current_bid.label() if current_bid else "none"
            print(f"  [{i}] P{player}: {'PICKUP' if picked_up else 'PASS'}")
            print(f"       10-card hand: {fmt_hand(hand)}")
            print(f"       Current bid: {bid_label} by P{holder}")
            print(f"       Talon on offer: [{', '.join(c.short() for c in talon)}]")

            if pe is not None:
                prefix = "p." if pe.is_piros else ""
                print(f"       Pickup eval: {prefix}{pe.contract_key} "
                      f"rank={pe.bid_rank}  "
                      f"value={pe.value:+.3f}  "
                      f"def_value={pe.def_value:+.3f}  "
                      f"trump={pe.trump}")
                if picked_up:
                    print(f"       → 12-card hand: {fmt_hand(hand + talon)}")
            elif not picked_up:
                print(f"       No qualifying contract above current bid")


# ---------------------------------------------------------------------------
#  Search speed presets
# ---------------------------------------------------------------------------

SPEEDS = {
    "fast":   (40, 2, 20, 6),
    "normal": (80, 3, 25, 6),
    "deep":   (200, 6, 50, 7),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a deal with full diagnostics")
    parser.add_argument("model", help="Model source (e.g. rook, trinity)")
    parser.add_argument("seed", type=int, help="Deal seed to replay")
    parser.add_argument("--speed", choices=SPEEDS, default="fast")
    parser.add_argument("--pickup-threshold", type=float, default=0.0)
    parser.add_argument("--pickup-quantile", type=float, default=0.5)
    parser.add_argument("--n-talon-samples", type=int, default=20)
    args = parser.parse_args()

    source = args.model
    seed = args.seed
    sims, dets, pimc_dets, endgame_tricks = SPEEDS[args.speed]

    print(f"\n  Replaying seed {seed} with model {source} (speed={args.speed})")
    print()

    # Load model
    wrappers = load_wrappers(source)
    tp = load_talon_prior(source)
    if not wrappers:
        print(f"  ERROR: No models found for {source}")
        sys.exit(1)
    print(f"  Loaded: {list(wrappers.keys())}")
    if tp:
        print(f"  Talon prior: {len(tp.contract_keys)} contracts")
    print()

    # Load betli hand evaluator if available
    hand_evaluators = None
    try:
        from trickster.betli.hand_evaluator import BetliHandEvaluator
        from trickster.betli.model import load_model as load_betli
        p = Path(f"models/ulti/{source}/final/betli/betli_hand_eval.pt")
        if p.exists():
            net = load_betli(p)
            net.eval()
            hand_evaluators = {"betli": BetliHandEvaluator(net, device="cpu")}
            print(f"  Betli hand evaluator: loaded from {p}")
    except ImportError:
        pass

    game = UltiGame()
    thresholds = {"betli": args.pickup_threshold} if args.pickup_threshold != 0.0 else None

    # 1. Deal — exactly as betli_pickup_diag.py does
    rng = random.Random(seed)
    deck = make_deck()
    rng.shuffle(deck)
    hands = [deck[:10], deck[10:20], deck[20:30]]
    talon = deck[30:32]
    dealer = seed % 3
    first_bidder = next_player(dealer)

    print(f"  ═══ DEAL (seed={seed}, dealer=P{dealer}) ═══")
    for p in range(3):
        marker = " ← first bidder" if p == first_bidder else ""
        q = betli_quality(hands[p])
        print(f"  P{p}: {fmt_hand(hands[p])}  (betli q={q:.2f}){marker}")
    print(f"  Talon: {' '.join(c.short() for c in talon)}")
    print()

    # 2. Run auction — same call as the diagnostic, same rng object
    gs = GameState(
        hands=[list(h) for h in hands],
        trump=None, betli=False, soloist=0, dealer=dealer,
        captured=[[], [], []], scores=[0, 0, 0],
        leader=first_bidder, trick_no=0,
        trick_cards=[], last_trick=None,
    )

    seat_wrappers = [wrappers, wrappers, wrappers]
    auction_log, on_step = make_step_logger()

    result = run_auction(
        gs, talon, dealer, seat_wrappers,
        pickup_quantile=args.pickup_quantile,
        n_talon_samples=args.n_talon_samples,
        rng=rng,
        hand_evaluators=hand_evaluators,
        pickup_thresholds=thresholds,
        talon_prior=tp,
        on_step=on_step,
    )

    # 3. Display auction with full evaluation details
    print(f"  ═══ AUCTION ═══")
    print_auction_log(auction_log)

    soloist = result.soloist
    print()
    print(f"  Winner: P{soloist}")
    if result.auction.current_bid:
        print(f"  Winning bid: {result.auction.current_bid.label()}")

    if result.bid is not None:
        ck = result.bid.contract_key
        bd = result.bid.best_discard
        print(f"  Contract: {'p.' if result.bid.is_piros else ''}{ck}")
        print(f"  Eval: game_pts={result.bid.game_pts:+.3f}  "
              f"best_discard.game_pts={bd.game_pts:+.3f}  "
              f"best_discard.value={bd.value:+.3f}")
        print(f"  Discard: [{', '.join(c.short() for c in bd.discard)}]")
        print(f"  Soloist hand: {fmt_hand(gs.hands[soloist])}")
    print()

    # Check for all-pass
    from trickster.games.ulti.auction import BID_PASSZ
    if result.auction.current_bid and result.auction.current_bid.rank <= BID_PASSZ.rank:
        print(f"  All-pass: P{soloist} pays pass penalty.")
        return

    if result.bid is None:
        print(f"  No bid — cannot replay play phase.")
        return

    bid = result.bid
    ck = bid.contract_key

    # 4. Play phase
    print(f"  ═══ PLAY ({ck}, soloist=P{soloist}) ═══")
    print()

    pbr = extract_player_bid_ranks(result.auction)
    state = setup_bid_game(
        game, gs, soloist, dealer, bid,
        initial_bidder=result.initial_bidder,
        player_bid_ranks=pbr,
    )

    sol_mcts = MCTSConfig(
        simulations=sims, determinizations=dets,
        use_value_head=True, use_policy_priors=True,
    )
    def_mcts = MCTSConfig(
        simulations=sims // 2, determinizations=dets,
        use_value_head=True, use_policy_priors=True,
    )

    wrapper = wrappers.get(ck) or next(iter(wrappers.values()))
    sol_player = HybridPlayer(
        game, wrapper, mcts_config=sol_mcts,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=0.5,
    )
    def_player = HybridPlayer(
        game, wrapper, mcts_config=def_mcts,
        endgame_tricks=endgame_tricks,
        pimc_determinizations=pimc_dets,
        solver_temperature=0.5,
    )

    play_rng = random.Random(seed + 9999)
    trick_no = 0
    trick_cards: list[tuple[int, Card]] = []

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)

        if len(actions) <= 1:
            action = actions[0]
        else:
            hp = sol_player if player == soloist else def_player
            action = hp.choose_action(state, player, play_rng)

        trick_cards.append((player, action))

        if len(trick_cards) == 3:
            trick_no += 1
            cards_str = "  ".join(f"P{p}:{c.short()}" for p, c in trick_cards)
            next_state = game.apply(state, action)
            winner = next_state.gs.leader
            sol_marker = " ← SOL" if winner == soloist else ""
            print(f"  Trick {trick_no}: {cards_str}  → P{winner}{sol_marker}")
            trick_cards = []
            state = next_state
        else:
            state = game.apply(state, action)

    # 5. Outcome
    sol_reward = simple_outcome(state, soloist)
    print()
    print(f"  ═══ RESULT ═══")
    print(f"  Soloist reward: {sol_reward:+.3f}")
    won = sol_reward > 0
    print(f"  {'WIN' if won else 'LOSS'}")


if __name__ == "__main__":
    main()
