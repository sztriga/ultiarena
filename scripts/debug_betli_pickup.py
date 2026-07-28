#!/usr/bin/env python3
"""Diagnose betli pickup — instrument the NEW per-sample-best logic.

Runs the real auction and catches betli selections, showing:
- The per-sample-best values that led to betli being selected
- How many talons had betli as the best contract
- The value head's predictions
"""
from __future__ import annotations

import copy
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trickster.bidding.evaluator import evaluate_all_contracts
from trickster.bidding.registry import CONTRACT_DEFS
from trickster.games.ulti.adapter import UltiGame
from trickster.games.ulti.cards import BETLI_STRENGTH, make_deck
from trickster.games.ulti.game import deal, next_player
from trickster.model import UltiNet, make_wrapper
from trickster.train_utils import _GAME_PTS_MAX


def fmt_hand(hand):
    by_suit = {}
    for c in sorted(hand, key=lambda c: (c.suit.value, BETLI_STRENGTH[c.rank])):
        by_suit.setdefault(c.suit.name[0], []).append(c.short())
    return "  ".join(f"{s}: {' '.join(cards)}" for s, cards in by_suit.items())


def instrumented_new_pickup(gs, player, dealer, wrappers, rng):
    """Run the NEW per-sample-best pickup logic with full instrumentation."""
    hand = gs.hands[player]
    assert len(hand) == 10

    all_cards = frozenset(make_deck())
    unknown = list(all_cards - set(hand))
    n_unknown = len(unknown)
    k = 20

    talon_samples = []
    seen = set()
    while len(talon_samples) < k:
        pair = tuple(sorted(rng.sample(range(n_unknown), 2)))
        if pair not in seen:
            seen.add(pair)
            talon_samples.append((unknown[pair[0]], unknown[pair[1]]))

    # Per-sample best + full detail per sample
    sample_details = []

    for talon_pair in talon_samples:
        gs_copy = copy.deepcopy(gs)
        gs_copy.hands[player] = list(hand) + list(talon_pair)

        evals = evaluate_all_contracts(
            gs_copy, player, dealer, wrappers=wrappers, min_bid_rank=0,
        )
        if not evals:
            continue

        best_ev = evals[0]
        label = f"{'p.' if best_ev.is_piros else ''}{best_ev.contract_key}"

        # Also get betli's value for this talon (if present)
        betli_val = None
        for ev in evals:
            if ev.contract_key == "betli" and not ev.is_piros:
                betli_val = ev.game_pts
                break

        sample_details.append(dict(
            best_label=label,
            best_pts=best_ev.game_pts,
            betli_pts=betli_val,
            talon=[c.short() for c in talon_pair],
        ))

    if not sample_details:
        return None

    # Sort by best_pts for quantile
    sample_details.sort(key=lambda x: x["best_pts"])
    idx = min(int(len(sample_details) * 0.5), len(sample_details) - 1)
    qval = sample_details[idx]["best_pts"]
    value = qval / (_GAME_PTS_MAX / 2)

    # Vote
    vote_counts = {}
    for s in sample_details:
        vote_counts[s["best_label"]] = vote_counts.get(s["best_label"], 0) + 1
    winner = max(vote_counts, key=vote_counts.get)

    return dict(
        winner=winner, qval=qval, value=value, passes=value > 0.0,
        votes=vote_counts, samples=sample_details, idx=idx,
    )


def main():
    game = UltiGame()

    wrappers = {}
    for ck in ["parti", "ulti", "40-100", "betli"]:
        model_pt = Path(f"models/ulti/trinity/final/{ck}/model.pt")
        if not model_pt.exists():
            continue
        cp = torch.load(model_pt, weights_only=False, map_location="cpu")
        n = UltiNet(
            input_dim=cp.get("input_dim", game.state_dim),
            body_units=cp.get("body_units", 256),
            body_layers=cp.get("body_layers", 4),
            action_dim=cp.get("action_dim", game.action_space_size),
        )
        n.load_state_dict(cp["model_state_dict"], strict=False)
        n.eval()
        wrappers[ck] = make_wrapper(n, device="cpu")

    print(f"  Loaded: {list(wrappers.keys())}")

    from trickster.bidding.auction_runner import run_auction
    from trickster.training.model_io import load_talon_prior
    _tp = load_talon_prior("trinity")

    base_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    max_betli = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    max_deals = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

    betli_found = 0

    for deal_idx in range(max_deals):
        seed = base_seed + deal_idx
        gs_orig, talon = deal(seed=seed, dealer=deal_idx % 3)
        dealer = deal_idx % 3

        # Run real auction with NEW logic
        gs_auction = copy.deepcopy(gs_orig)
        seat_wrappers = [wrappers, wrappers, wrappers]
        result = run_auction(
            gs_auction, talon, dealer, seat_wrappers,
            pickup_quantile=0.5,
            rng=random.Random(seed),
            talon_prior=_tp,
        )
        if result.bid is None or result.bid.contract_key != "betli":
            continue

        soloist = result.soloist
        betli_found += 1

        # Re-run instrumented pickup on the original 10-card hand
        gs_instr = copy.deepcopy(gs_orig)
        rng2 = random.Random(seed + soloist * 7919)
        info = instrumented_new_pickup(gs_instr, soloist, dealer, wrappers, rng2)

        print(f"\n  ═══ Deal {deal_idx} (seed {seed}), Soloist P{soloist} ═══")
        print(f"  10-card hand: {fmt_hand(gs_orig.hands[soloist])}")
        print(f"  Real talon:   {' '.join(c.short() for c in talon)}")

        if info:
            # Per-sample bests
            bv = [s["best_pts"] for s in info["samples"]]
            print(f"\n  Per-sample bests (sorted):")
            print(f"    {' '.join(f'{v:+.1f}' for v in bv)}")
            print(f"  Median (idx={info['idx']}): qval={info['qval']:+.2f}, "
                  f"value={info['value']:+.3f} → {'PICKUP' if info['passes'] else 'PASS'}")
            print(f"  Vote: {dict(sorted(info['votes'].items(), key=lambda x: -x[1]))}")

            # Show each sample's winner + betli value
            print(f"\n  Per-sample breakdown:")
            print(f"  {'#':>3} {'Talon':<10} {'Best contract':<14} {'Best pts':>9} {'Betli pts':>10}")
            print(f"  {'─' * 52}")
            for i, s in enumerate(info["samples"]):
                talon_str = " ".join(s["talon"])
                betli_str = f"{s['betli_pts']:+.2f}" if s["betli_pts"] is not None else "n/a"
                marker = " ← median" if i == info["idx"] else ""
                print(f"  {i+1:>3} {talon_str:<10} {s['best_label']:<14} {s['best_pts']:>+9.2f} {betli_str:>10}{marker}")
        else:
            print("  (instrumented pickup returned None)")

        # Danger analysis
        hand = gs_orig.hands[soloist]
        dangers = []
        for suit_name in "AHLB":
            suit_cards = [c for c in hand if c.suit.name.startswith(suit_name)]
            if not suit_cards:
                continue
            max_str = max(BETLI_STRENGTH[c.rank] for c in suit_cards)
            if max_str >= 5:
                danger_cards = [c for c in suit_cards if BETLI_STRENGTH[c.rank] >= 5]
                dangers.append(f"{suit_name}: {' '.join(c.short() for c in danger_cards)}")
        if dangers:
            print(f"\n  DANGER: {', '.join(dangers)}")

        # Show the bid that was actually made
        print(f"  Actual bid: {'p.' if result.bid.is_piros else ''}{result.bid.contract_key}")
        print(f"  Bid game_pts: {result.bid.game_pts:+.2f}")

        if betli_found >= max_betli:
            break

    print(f"\n  ══════════════════════════════════════════════════")
    print(f"  Found {betli_found} betli games in {min(deal_idx + 1, max_deals)} deals")


if __name__ == "__main__":
    main()
