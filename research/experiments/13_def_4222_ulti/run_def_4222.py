"""Sidequest: how often does sol's ulti fail when one defender holds
exactly 4 trumps + 2-2-2 of the non-trump suits?

Setup:
  - Sol holds the trump-7 (else no ulti possible).
  - One defender (the "tank") gets exactly 4 trumps (no 7) + 2 of each
    of the 3 non-trump suits.
  - Other defender + talon split the remaining cards arbitrarily.
  - Sol's remaining 9 cards are random from what's left.

Solve `contract='ulti'` via god solver (perfect info, alpha-beta).
Report fail rate broken down by which defender is the tank.
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.card import Card, RANKS, SUITS, fresh_deck
from ulti.solvers import pis

N            = 500
SEED_BASE    = 700_000_000
N_WORKERS    = 4


def _sample_deal(seed: int):
    rng = random.Random(seed)
    deck = fresh_deck()
    trump = rng.choice(SUITS)
    non_trumps = [s for s in SUITS if s != trump]
    tank = rng.choice([1, 2])  # which defender is the 4-2-2-2 tank

    # Partition deck by suit
    by_suit = {s: [c for c in deck if c.suit == s] for s in SUITS}

    trump_seven = Card(trump, '7')
    trump_others = [c for c in by_suit[trump] if c.rank != '7']  # 7 cards
    # Tank's 4 trumps (no 7)
    tank_trumps = rng.sample(trump_others, 4)
    # Tank's 2-2-2 non-trumps
    tank_non = []
    for s in non_trumps:
        tank_non.extend(rng.sample(by_suit[s], 2))
    tank_hand = tank_trumps + tank_non
    assert len(tank_hand) == 10

    # Remaining: 32 - 10 = 22 cards (including trump-7)
    used = set(tank_hand)
    remaining = [c for c in deck if c not in used]

    # Sol must have trump-7. Pick sol's other 9 from remaining \ {trump-7}.
    sol_others = [c for c in remaining if c != trump_seven]
    sol_rest = rng.sample(sol_others, 9)
    sol_hand = [trump_seven] + sol_rest

    used.update(sol_hand)
    leftover = [c for c in deck if c not in used]
    rng.shuffle(leftover)
    other_def_hand = leftover[:10]
    talon = leftover[10:12]

    if tank == 1:
        hands = [sol_hand, tank_hand, other_def_hand]
    else:
        hands = [sol_hand, other_def_hand, tank_hand]
    return hands, trump, tank, talon


def worker(seed):
    hands, trump, tank, talon = _sample_deal(seed)
    pos = pis.build_position(
        hands=hands, soloist=0, leader=0, contract='ulti',
        trump=trump, talon=talon,
    )
    vals = pis.solve_all(pos, contract='ulti')
    # Sol picks the move with max value. Ulti is binary: max value > 0
    # ⟹ sol can make ulti against god defense.
    sol_can_make = max(vals.values()) > 0
    return (seed, tank, trump, sol_can_make)


def main():
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(worker, seeds, chunksize=4):
            rows.append(r)
            if len(rows) % 50 == 0:
                wall = time.perf_counter() - t0
                rate = len(rows) / wall
                eta = (len(seeds) - len(rows)) / rate if rate else 0
                print(f"  {len(rows)}/{N}  wall={wall:.0f}s  eta={eta:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0

    total_won = sum(1 for _, _, _, w in rows if w)
    total_failed = N - total_won
    print()
    print(f"=== Overall (N={N}) ===")
    print(f"  sol made ulti:   {total_won:4d}  ({total_won/N:.3f})")
    print(f"  sol failed:      {total_failed:4d}  ({total_failed/N:.3f})")

    for tank in (1, 2):
        sub = [r for r in rows if r[1] == tank]
        ns = len(sub)
        if ns == 0:
            continue
        nw = sum(1 for _, _, _, w in sub if w)
        nf = ns - nw
        print()
        print(f"=== Tank = def{tank} (N={ns}) ===")
        print(f"  sol made ulti:   {nw:4d}  ({nw/ns:.3f})")
        print(f"  sol failed:      {nf:4d}  ({nf/ns:.3f})")

    print()
    print(f"Wall: {wall:.0f}s, {N_WORKERS} workers")


if __name__ == "__main__":
    main()
