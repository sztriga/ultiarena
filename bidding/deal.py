"""Shared minigame eval logic.

Per deal: sol gets 12 cards, defs 10 each. For each of 66 talon
discards × {4 trumps × {parti, ulti}, betli, duri} run PIMC32 and
convert the per-card averaged value into expected GP per defender.

GP rates come from `scoring_oracle.GPTable` defaults (single source
of truth across the codebase). Piros (hearts trump) is a flat ×2
multiplier per the oracle's convention.

    parti        : +1 made / -1 lost                 piros: +2/-2
    ulti         : +4 made / -8 bukott               piros: +8/-16
    duri         : +6 made / -6 lost   (colorless, no piros)
    betli        : +5 made / -5 lost   (colorless, no piros)
    pass         : 0
"""
from __future__ import annotations

import itertools, random, sys
from pathlib import Path


from ulti.card import SUITS, fresh_deck
from solvers import pis, pimc as _pimc
from scoring.oracle import GPTable

_GP = GPTable()


def deal_12_10_10(seed):
    rng = random.Random(seed)
    deck = fresh_deck()
    rng.shuffle(deck)
    return deck[:12], deck[12:22], deck[22:32]


def _pmake_from_avg(contract: str, avg: float) -> float:
    """Convert PIMC averaged value into P(make/win) ∈ [0,1]. All
    binary contracts (parti, ulti, betli, durchmars) use the uniform
    0/1 solver scale, so the PIMC average is already a probability."""
    return max(0.0, min(1.0, avg))


def _ev_per_def(contract: str, piros: bool, p_make: float) -> float:
    """Expected GP per defender = (R+L)·P − L, with piros ×2 multiplier
    on colored contracts (parti, ulti). Rates come from GPTable so the
    eval stays consistent with the post-play oracle scoring."""
    mult = 2 if piros else 1
    if contract == 'parti':
        made, lost = _GP.parti, _GP.parti
    elif contract == 'ulti':
        made, lost = _GP.ulti_bid, _GP.ulti_bid_bukott  # 4 / 8
    elif contract == 'durchmars':
        made, lost = _GP.durchmars_bid, _GP.durchmars_bid  # 6 / 6
        mult = 1  # colorless
    elif contract == 'betli':
        made, lost = _GP.betli, _GP.betli  # 5 / 5
        mult = 1  # colorless
    else:
        raise ValueError(contract)
    return mult * (made * p_make - lost * (1.0 - p_make))


def eval_one_deal(sol12, def1_10, def2_10, *, pimc_n: int, seed: int):
    """Return list of records: (discard_pair, contract, trump_or_None,
    p_make, ev_per_def)."""
    records = []
    rng_seed = seed
    discards = list(itertools.combinations(sol12, 2))

    for discard_pair in discards:
        remaining = [c for c in sol12 if c not in discard_pair]
        talon = list(discard_pair)

        # parti × 4 trumps + ulti × 4 trumps (ulti only if sol has trump-7)
        for trump in SUITS:
            piros = (trump == 'hearts')
            for contract in ('parti', 'ulti'):
                if contract == 'ulti':
                    has_t7 = any(c.suit == trump and c.rank == '7'
                                 for c in remaining)
                    if not has_t7:
                        records.append((discard_pair, contract, trump,
                                        0.0, _ev_per_def(contract, piros, 0.0)))
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
                if not avg:
                    p = 0.0
                else:
                    p = _pmake_from_avg(contract, max(avg.values()))
                records.append((discard_pair, contract, trump, p,
                                _ev_per_def(contract, piros, p)))

        # betli + duri (trumpless, colorless)
        for contract in ('betli', 'durchmars'):
            pos = pis.build_position(
                hands=[remaining, def1_10, def2_10], soloist=0,
                leader=0, contract=contract, trump=None, talon=talon,
            )
            rng_seed += 1
            _, avg = _pimc.pimc_decision(
                true_pos=pos, contract=contract, n_samples=pimc_n,
                seed=rng_seed,
            )
            if not avg:
                p = 0.0
            else:
                p = _pmake_from_avg(contract, max(avg.values()))
            records.append((discard_pair, contract, None, p,
                            _ev_per_def(contract, piros=False, p_make=p)))

    return records


def best_record(records):
    """Pick the highest-EV (discard, contract, trump?) and apply the
    pass rule (return None if best EV < 0)."""
    if not records:
        return None
    best = max(records, key=lambda r: r[4])
    if best[4] < 0:
        return None  # pass
    return best
