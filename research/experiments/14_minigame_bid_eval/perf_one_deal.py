"""Perf test for the minigame bid evaluator on ONE deal.

For one randomly dealt 12-card sol hand, evaluate every (discard, contract,
trump) combination via PIMC32 and time the whole thing.

Counts both the upper bound (all 660 calls) and the actual number of
calls made (skipping ulti when sol doesn't hold the trump-7 in the
post-discard 10-card hand).
"""
from __future__ import annotations

import itertools, random, sys, time
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.card import Card, SUITS, fresh_deck
from solvers import pis, pimc as _pimc

PIMC_N = 32
SEED   = 42


def deal_12(seed):
    rng = random.Random(seed)
    deck = fresh_deck()
    rng.shuffle(deck)
    sol = deck[:12]
    def1 = deck[12:22]
    def2 = deck[22:32]
    return sol, def1, def2


def eval_one_deal(sol12, def1_10, def2_10, pimc_n, seed):
    upper_bound_calls = 0
    actual_calls = 0
    results = {}   # (discard_pair, contract, trump_or_none) -> max_avg

    rng_seed = seed
    discards = list(itertools.combinations(sol12, 2))
    for discard_pair in discards:
        remaining = [c for c in sol12 if c not in discard_pair]
        assert len(remaining) == 10
        talon = list(discard_pair)

        # parti × 4 trumps  +  ulti × 4 trumps  =  8 trump-contract calls
        for trump in SUITS:
            for contract in ('parti', 'ulti'):
                upper_bound_calls += 1
                if contract == 'ulti':
                    # Skip if sol doesn't hold the trump-7
                    has_t7 = any(c.suit == trump and c.rank == '7' for c in remaining)
                    if not has_t7:
                        continue
                pos = pis.build_position(
                    hands=[remaining, def1_10, def2_10], soloist=0,
                    leader=0, contract=contract, trump=trump, talon=talon,
                )
                rng_seed += 1
                _, avg = _pimc.pimc_decision(
                    true_pos=pos, contract=contract, n_samples=pimc_n,
                    seed=rng_seed,
                )
                actual_calls += 1
                results[(discard_pair, contract, trump)] = max(avg.values()) if avg else None

        # betli + duri (trumpless)
        for contract in ('betli', 'durchmars'):
            upper_bound_calls += 1
            pos = pis.build_position(
                hands=[remaining, def1_10, def2_10], soloist=0,
                leader=0, contract=contract, trump=None, talon=talon,
            )
            rng_seed += 1
            _, avg = _pimc.pimc_decision(
                true_pos=pos, contract=contract, n_samples=pimc_n,
                seed=rng_seed,
            )
            actual_calls += 1
            results[(discard_pair, contract, None)] = max(avg.values()) if avg else None

    return upper_bound_calls, actual_calls, results


def main():
    sol12, def1_10, def2_10 = deal_12(SEED)
    n_trump_7 = sum(1 for c in sol12 if c.rank == '7')
    print(f"Sol 12-card hand:")
    by_suit = {}
    for c in sol12:
        by_suit.setdefault(c.suit, []).append(c.rank)
    for s in SUITS:
        if s in by_suit:
            print(f"  {s:>7}: {by_suit[s]}")
    print(f"Sol holds {n_trump_7} trump-7s in 12-card hand.")
    print()

    t0 = time.perf_counter()
    ub, actual, results = eval_one_deal(sol12, def1_10, def2_10, PIMC_N, SEED)
    wall = time.perf_counter() - t0

    print(f"=== Perf summary (PIMC{PIMC_N}, 1 deal) ===")
    print(f"  upper-bound calls:    {ub}")
    print(f"  actual calls made:    {actual}  ({actual/ub*100:.1f}% of upper)")
    print(f"  wall:                 {wall:.1f}s")
    print(f"  ms/call:              {wall/actual*1000:.1f}")
    print(f"  estimated wall/deal at upper bound: {wall/actual*ub:.1f}s")


if __name__ == "__main__":
    main()
