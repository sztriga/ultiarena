#!/usr/bin/env python3
"""Diagnose betli bleeding: is it selection or play?

Two tests:
  1. AUCTION TEST — Run N deals through the full auction, track which become
     betli, measure hand quality, and report win rates by quality tier.
  2. FORCED-PLAY TEST — Force-deal known-good betli hands (using the dojo's
     biased dealer) and make the AI play them.  Isolates play skill from
     selection skill.

Usage:
    python scripts/diagnose_betli.py scout --deals 300
    python scripts/diagnose_betli.py scout --deals 500 --speed fast
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from trickster.bidding.auction_runner import (
    extract_player_bid_ranks,
    run_auction,
    setup_bid_game,
)
from trickster.bidding.constants import (
    MIN_BID_PTS,
    PASS_PENALTY,
)
from trickster.bidding.evaluator import evaluate_all_contracts, _make_eval_state
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import (
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
    soloist_lost_betli,
    soloist_tricks,
)
from trickster.games.ulti.rewards import simple_outcome
from trickster.hybrid import HybridPlayer
from trickster.mcts import MCTSConfig
from trickster.model import UltiNetWrapper
from trickster.training.model_io import load_talon_prior, load_wrappers


# ---------------------------------------------------------------------------
#  Hand quality (same as dojo)
# ---------------------------------------------------------------------------

def hand_quality(hand: list[Card]) -> float:
    """Betli hand quality: 0.0 (all Aces) to 1.0 (all 7s)."""
    return sum(7 - BETLI_STRENGTH[c.rank] for c in hand) / 70.0


def hand_summary(hand: list[Card]) -> str:
    """One-line summary: suit counts + highest per suit."""
    by_suit: dict[Suit, list[Card]] = defaultdict(list)
    for c in hand:
        by_suit[c.suit].append(c)
    parts = []
    suit_sym = {Suit.HEARTS: "P", Suit.BELLS: "T", Suit.LEAVES: "Z", Suit.ACORNS: "M"}
    for s in [Suit.HEARTS, Suit.BELLS, Suit.LEAVES, Suit.ACORNS]:
        cards = sorted(by_suit.get(s, []), key=lambda c: BETLI_STRENGTH[c.rank])
        if cards:
            rank_names = {
                Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "X",
                Rank.JACK: "J", Rank.QUEEN: "Q", Rank.KING: "K", Rank.ACE: "A",
            }
            ranks = "".join(rank_names[c.rank] for c in cards)
            parts.append(f"{suit_sym[s]}:{ranks}")
    return " ".join(parts)


def max_betli_strength(hand: list[Card]) -> int:
    """Highest betli-strength card in hand (7=worst for betli)."""
    return max(BETLI_STRENGTH[c.rank] for c in hand)


def count_high_cards(hand: list[Card], threshold: int = 5) -> int:
    """Count cards with betli strength >= threshold (K=6, A=7)."""
    return sum(1 for c in hand if BETLI_STRENGTH[c.rank] >= threshold)


# ---------------------------------------------------------------------------
#  Search preset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchPreset:
    sims: int
    dets: int
    pimc_dets: int
    endgame_tricks: int

    def mcts_config(self) -> MCTSConfig:
        return MCTSConfig(
            simulations=self.sims,
            determinizations=self.dets,
            c_puct=1.5,
            dirichlet_alpha=0.0,
            dirichlet_weight=0.0,
            use_value_head=True,
            use_policy_priors=True,
            visit_temp=0.1,
            leaf_batch_size=8,
        )


SPEEDS = {
    "fast":   SearchPreset(sims=40,  dets=2, pimc_dets=20, endgame_tricks=6),
    "normal": SearchPreset(sims=80,  dets=3, pimc_dets=25, endgame_tricks=6),
}


# ---------------------------------------------------------------------------
#  Play a game to completion
# ---------------------------------------------------------------------------

def play_game(
    game: UltiGame,
    state: UltiNode,
    wrapper: UltiNetWrapper,
    search: SearchPreset,
    rng: random.Random,
) -> UltiNode:
    """Play a game to completion, return terminal state."""
    players = {}
    for seat in range(3):
        players[seat] = HybridPlayer(
            game, wrapper,
            mcts_config=search.mcts_config(),
            endgame_tricks=search.endgame_tricks,
            pimc_determinizations=search.pimc_dets,
            solver_temperature=0.1,
        )

    while not game.is_terminal(state):
        player = game.current_player(state)
        actions = game.legal_actions(state)
        if len(actions) == 1:
            state = game.apply(state, actions[0])
            continue
        action = players[player].choose_action(state, player, rng)
        state = game.apply(state, action)

    return state


# ---------------------------------------------------------------------------
#  TEST 1: Auction test — what betli hands does the AI select?
# ---------------------------------------------------------------------------

@dataclass
class AuctionBetliResult:
    seed: int
    hand_quality: float
    hand_10card: list[Card]
    hand_10card_summary: str
    max_strength: int
    high_cards: int
    soloist_won: bool
    is_piros: bool
    tricks_won: int
    sol_value_pred: float  # value head prediction at game start


def run_auction_test(
    game: UltiGame,
    wrappers: dict[str, UltiNetWrapper],
    search: SearchPreset,
    n_deals: int,
    seed: int,
    talon_prior: object | None = None,
) -> tuple[list[AuctionBetliResult], dict]:
    """Run N deals through the auction, collect betli outcomes."""

    deal_rng = random.Random(seed)
    betli_results: list[AuctionBetliResult] = []
    contract_counts: dict[str, int] = defaultdict(int)
    total_passes = 0

    for i in range(n_deals):
        deal_seed = deal_rng.randint(0, 2**31)
        rng = random.Random(deal_seed)
        dealer = i % 3

        gs, talon = deal(seed=deal_seed, dealer=dealer)
        seat_wrappers = [wrappers, wrappers, wrappers]

        result = run_auction(
            gs, talon, dealer, seat_wrappers,
            min_bid_pts=MIN_BID_PTS,
            pickup_quantile=0.5,
            rng=rng,
            talon_prior=talon_prior,
        )

        if result.bid is None:
            total_passes += 1
            contract_counts["__pass__"] += 1
            continue

        ck = result.bid.contract_key
        is_piros = result.bid.is_piros
        dkey = f"p.{ck}" if is_piros else ck
        contract_counts[dkey] += 1

        if ck != "betli":
            continue

        # This is a betli game — analyze it
        soloist = result.soloist
        hand_10 = list(gs.hands[soloist])  # 10-card hand after discard
        quality = hand_quality(hand_10)

        # Get value head prediction
        state = setup_bid_game(
            game, gs, soloist, dealer, result.bid,
            initial_bidder=result.initial_bidder,
            player_bid_ranks=extract_player_bid_ranks(result.auction),
        )

        betli_w = wrappers.get("betli")
        sol_val = 0.0
        if betli_w is not None:
            feats = game.encode_state(state, soloist)
            sol_val = float(betli_w.predict_value(feats))

        # Play it out
        terminal = play_game(game, state, betli_w, search, rng)
        won = not soloist_lost_betli(terminal.gs)
        tricks = soloist_tricks(terminal.gs)

        betli_results.append(AuctionBetliResult(
            seed=deal_seed,
            hand_quality=quality,
            hand_10card=hand_10,
            hand_10card_summary=hand_summary(hand_10),
            max_strength=max_betli_strength(hand_10),
            high_cards=count_high_cards(hand_10),
            soloist_won=won,
            is_piros=is_piros,
            tricks_won=tricks,
            sol_value_pred=sol_val,
        ))

        if (i + 1) % 50 == 0:
            print(f"\r  Auction test: {i+1}/{n_deals} deals, "
                  f"{len(betli_results)} betli games...", end="", flush=True)

    print()
    summary = {
        "total_deals": n_deals,
        "total_passes": total_passes,
        "contract_counts": dict(contract_counts),
    }
    return betli_results, summary


# ---------------------------------------------------------------------------
#  TEST 2: Forced betli — isolate play skill
# ---------------------------------------------------------------------------

@dataclass
class ForcedBetliResult:
    seed: int
    hand_quality: float
    hand_summary: str
    max_strength: int
    high_cards: int
    soloist_won: bool
    tricks_won: int
    sol_value_pred: float


def biased_betli_deal(rng: random.Random, alpha: float = 0.5) -> tuple[list[list[Card]], list[Card]]:
    """Deal biased betli hands (same as dojo)."""
    import math
    deck = make_deck()
    weights = []
    for card in deck:
        w = math.exp(-alpha * BETLI_STRENGTH[card.rank])
        weights.append(w)

    remaining = list(range(len(deck)))
    remaining_weights = list(weights)
    soloist_indices: list[int] = []

    for _ in range(10):
        total = sum(remaining_weights)
        r = rng.random() * total
        cumulative = 0.0
        chosen = 0
        for j, w in enumerate(remaining_weights):
            cumulative += w
            if cumulative >= r:
                chosen = j
                break
        soloist_indices.append(remaining[chosen])
        remaining.pop(chosen)
        remaining_weights.pop(chosen)

    soloist_hand = [deck[j] for j in soloist_indices]
    rest = [deck[j] for j in remaining]
    rng.shuffle(rest)
    hands = [soloist_hand, rest[:10], rest[10:20]]
    talon = rest[20:22]
    return hands, talon


def run_forced_betli_test(
    game: UltiGame,
    wrapper: UltiNetWrapper,
    search: SearchPreset,
    n_games: int,
    seed: int,
    alpha: float = 0.5,
) -> list[ForcedBetliResult]:
    """Force-deal biased betli hands and play them."""

    results: list[ForcedBetliResult] = []
    rng = random.Random(seed)

    for i in range(n_games):
        game_rng = random.Random(seed + i * 7919)

        # Biased deal
        hands, talon = biased_betli_deal(game_rng, alpha=alpha)
        dealer = 2
        soloist = 0

        gs = GameState(
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

        # Pickup and smart discard
        pickup_talon(gs, soloist, talon)

        # Find best discard using value head
        hand = gs.hands[soloist]
        cdef = CONTRACT_DEFS["betli"]
        best_val = -float("inf")
        best_pair = (hand[0], hand[1])

        for c1, c2 in combinations(hand, 2):
            node = _make_eval_state(
                gs, soloist, trump=None, discards=(c1, c2),
                contract_def=cdef, is_piros=False, dealer=dealer,
            )
            feats = game.encode_state(node, soloist)
            val = float(wrapper.predict_value(feats))
            if val > best_val:
                best_val = val
                best_pair = (c1, c2)

        discard_talon(gs, list(best_pair))
        set_contract(gs, soloist, trump=None, betli=True)
        gs.training_mode = "betli"
        declare_all_marriages(gs)

        hand_10 = list(gs.hands[soloist])
        quality = hand_quality(hand_10)

        node = UltiNode(
            gs=gs,
            known_voids=(frozenset(), frozenset(), frozenset()),
            bid_rank=5,
            is_red=False,
            contract_components=frozenset({"betli"}),
            dealer=dealer,
        )

        # Value prediction at game start
        feats = game.encode_state(node, soloist)
        sol_val = float(wrapper.predict_value(feats))

        # Play it
        terminal = play_game(game, node, wrapper, search, game_rng)
        won = not soloist_lost_betli(terminal.gs)
        tricks = soloist_tricks(terminal.gs)

        results.append(ForcedBetliResult(
            seed=seed + i,
            hand_quality=quality,
            hand_summary=hand_summary(hand_10),
            max_strength=max_betli_strength(hand_10),
            high_cards=count_high_cards(hand_10),
            soloist_won=won,
            tricks_won=tricks,
            sol_value_pred=sol_val,
        ))

        if (i + 1) % 50 == 0:
            wins = sum(1 for r in results if r.soloist_won)
            print(f"\r  Forced betli: {i+1}/{n_games}, "
                  f"win rate {wins}/{len(results)} = {wins/len(results):.0%}",
                  end="", flush=True)

    print()
    return results


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------

def print_auction_report(
    betli_results: list[AuctionBetliResult],
    summary: dict,
):
    print()
    print("=" * 72)
    print("  TEST 1: AUCTION — What betli hands does the AI select?")
    print("=" * 72)

    # Contract distribution
    cc = summary["contract_counts"]
    total = summary["total_deals"]
    print()
    print(f"  Contract distribution ({total} deals):")
    for dk in sorted(cc, key=lambda k: -cc[k]):
        label = dk if dk != "__pass__" else "Pass"
        pct = cc[dk] / total * 100
        print(f"    {label:<15} {cc[dk]:>5} ({pct:>5.1f}%)")

    if not betli_results:
        print("\n  No betli games were bid!")
        return

    n = len(betli_results)
    wins = sum(1 for r in betli_results if r.soloist_won)
    print()
    print(f"  Betli games: {n}   Win rate: {wins}/{n} = {wins/n:.0%}")

    # Quality tiers
    tiers = [(0.0, 0.3, "Bad  (0.0-0.3)"),
             (0.3, 0.5, "Meh  (0.3-0.5)"),
             (0.5, 0.7, "OK   (0.5-0.7)"),
             (0.7, 1.01, "Good (0.7-1.0)")]

    print()
    print(f"  {'Quality tier':<20} {'N':>5} {'Wins':>5} {'Win%':>6} {'Avg V-pred':>11} {'Avg tricks':>11}")
    print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*6} {'─'*11} {'─'*11}")

    for lo, hi, label in tiers:
        tier = [r for r in betli_results if lo <= r.hand_quality < hi]
        if not tier:
            print(f"  {label:<20} {'n/a':>5}")
            continue
        tw = sum(1 for r in tier if r.soloist_won)
        avg_v = np.mean([r.sol_value_pred for r in tier])
        avg_t = np.mean([r.tricks_won for r in tier])
        print(f"  {label:<20} {len(tier):>5} {tw:>5} {tw/len(tier):>5.0%} "
              f"{avg_v:>+10.3f} {avg_t:>10.1f}")

    # High card analysis
    print()
    print(f"  {'High cards (K/A)':>20} {'N':>5} {'Wins':>5} {'Win%':>6}")
    print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*6}")
    for hc in range(5):
        tier = [r for r in betli_results if r.high_cards == hc]
        if not tier:
            continue
        tw = sum(1 for t in tier if t.soloist_won)
        print(f"  {f'{hc} high cards':<20} {len(tier):>5} {tw:>5} {tw/len(tier):>5.0%}")

    # Show some example hands (wins and losses)
    losses = [r for r in betli_results if not r.soloist_won]
    wins_list = [r for r in betli_results if r.soloist_won]

    if losses:
        print()
        print("  LOST betli hands (up to 10):")
        print(f"    {'Q':>5} {'V-pred':>7} {'Tricks':>6}  Hand")
        for r in sorted(losses, key=lambda r: -r.hand_quality)[:10]:
            print(f"    {r.hand_quality:>5.2f} {r.sol_value_pred:>+7.3f} {r.tricks_won:>6}  {r.hand_10card_summary}")

    if wins_list:
        print()
        print("  WON betli hands (up to 10):")
        print(f"    {'Q':>5} {'V-pred':>7} {'Tricks':>6}  Hand")
        for r in sorted(wins_list, key=lambda r: r.hand_quality)[:10]:
            print(f"    {r.hand_quality:>5.2f} {r.sol_value_pred:>+7.3f} {r.tricks_won:>6}  {r.hand_10card_summary}")


def print_forced_report(results: list[ForcedBetliResult]):
    print()
    print("=" * 72)
    print("  TEST 2: FORCED BETLI — Can the AI play good betli hands?")
    print("=" * 72)

    n = len(results)
    wins = sum(1 for r in results if r.soloist_won)
    print()
    print(f"  Games: {n}   Win rate: {wins}/{n} = {wins/n:.0%}")

    tiers = [(0.0, 0.3, "Bad  (0.0-0.3)"),
             (0.3, 0.5, "Meh  (0.3-0.5)"),
             (0.5, 0.7, "OK   (0.5-0.7)"),
             (0.7, 1.01, "Good (0.7-1.0)")]

    print()
    print(f"  {'Quality tier':<20} {'N':>5} {'Wins':>5} {'Win%':>6} {'Avg V-pred':>11} {'Avg tricks':>11}")
    print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*6} {'─'*11} {'─'*11}")

    for lo, hi, label in tiers:
        tier = [r for r in results if lo <= r.hand_quality < hi]
        if not tier:
            print(f"  {label:<20} {'n/a':>5}")
            continue
        tw = sum(1 for r in tier if r.soloist_won)
        avg_v = np.mean([r.sol_value_pred for r in tier])
        avg_t = np.mean([r.tricks_won for r in tier])
        print(f"  {label:<20} {len(tier):>5} {tw:>5} {tw/len(tier):>5.0%} "
              f"{avg_v:>+10.3f} {avg_t:>10.1f}")

    # Show lost high-quality hands
    good_losses = [r for r in results if not r.soloist_won and r.hand_quality >= 0.5]
    if good_losses:
        print()
        print(f"  LOST despite good hand (q >= 0.5): {len(good_losses)} games")
        print(f"    {'Q':>5} {'V-pred':>7} {'Tricks':>6}  Hand")
        for r in sorted(good_losses, key=lambda r: -r.hand_quality)[:15]:
            print(f"    {r.hand_quality:>5.2f} {r.sol_value_pred:>+7.3f} {r.tricks_won:>6}  {r.hand_summary}")

    good_wins = [r for r in results if r.soloist_won and r.hand_quality >= 0.5]
    if good_wins:
        print()
        print(f"  WON with good hand (q >= 0.5): {len(good_wins)}/{len([r for r in results if r.hand_quality >= 0.5])}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose betli bleeding")
    parser.add_argument("model", help="Model source (e.g. scout)")
    parser.add_argument("--deals", type=int, default=300, help="Auction deals")
    parser.add_argument("--forced", type=int, default=200, help="Forced betli games")
    parser.add_argument("--speed", default="fast", choices=list(SPEEDS.keys()))
    parser.add_argument("--alpha", type=float, default=0.5, help="Forced deal bias")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    search = SPEEDS[args.speed]

    print()
    print(f"  Loading model: {args.model}")
    game = UltiGame()
    wrappers = load_wrappers(args.model)
    betli_wrapper = wrappers.get("betli")
    print(f"    Loaded {len(wrappers)} contract models")
    if betli_wrapper is None:
        print("  ERROR: No betli model found!")
        sys.exit(1)
    print()

    t0 = time.perf_counter()

    # Test 1: Auction
    print(f"  Running auction test ({args.deals} deals)...")
    tp = load_talon_prior(args.model)
    betli_results, summary = run_auction_test(
        game, wrappers, search, args.deals, args.seed,
        talon_prior=tp,
    )
    t1 = time.perf_counter()
    print_auction_report(betli_results, summary)
    print(f"\n  Auction test: {t1 - t0:.0f}s")

    # Test 2: Forced betli
    print()
    print(f"  Running forced betli test ({args.forced} games, alpha={args.alpha})...")
    forced_results = run_forced_betli_test(
        game, betli_wrapper, search, args.forced, args.seed + 99999,
        alpha=args.alpha,
    )
    t2 = time.perf_counter()
    print_forced_report(forced_results)
    print(f"\n  Forced betli test: {t2 - t1:.0f}s")

    # Verdict
    print()
    print("=" * 72)
    print("  VERDICT")
    print("=" * 72)
    if betli_results:
        auction_wr = sum(1 for r in betli_results if r.soloist_won) / len(betli_results)
    else:
        auction_wr = 0
    forced_good = [r for r in forced_results if r.hand_quality >= 0.5]
    forced_good_wr = (
        sum(1 for r in forced_good if r.soloist_won) / len(forced_good)
        if forced_good else 0
    )
    forced_all_wr = sum(1 for r in forced_results if r.soloist_won) / len(forced_results)

    print(f"  Auction betli win rate:        {auction_wr:.0%} ({len(betli_results)} games)")
    print(f"  Forced betli win rate (all):   {forced_all_wr:.0%} ({len(forced_results)} games)")
    print(f"  Forced betli win rate (q>=0.5): {forced_good_wr:.0%} ({len(forced_good)} games)")
    print()

    if forced_good_wr < 0.5:
        print("  >>> PLAY PROBLEM: AI can't win even with good betli hands.")
        print("      The policy/value head needs more betli training (dojo).")
    elif auction_wr < 0.3:
        if len(betli_results) < 5:
            print("  >>> Too few auction betli games to judge selection.")
            print(f"      Only {len(betli_results)} betli bids in {args.deals} deals.")
        else:
            avg_q = np.mean([r.hand_quality for r in betli_results])
            if avg_q < 0.4:
                print("  >>> SELECTION PROBLEM: AI picks bad betli hands.")
                print(f"      Average hand quality: {avg_q:.2f} (should be > 0.5)")
                print("      The pickup quantile gate may need tightening.")
            else:
                print("  >>> MIXED: Decent hand selection but poor play outcomes.")
                print("      May need both better selection AND better play.")
    else:
        print("  >>> Betli seems OK! Win rate is reasonable.")
    print()


if __name__ == "__main__":
    main()
