"""Are the hands the v18a auction commits to actually god-winnable?

For each of the 3000 eval deals, resolve the auction (no PIMC32 playout)
and god-label the winner's *actual* final layout. Reports, per contract:
god-win% (could the soloist make it under perfect double-dummy play) vs
the PIMC32-realized won% from the bleeder run.

If god-win% >> realized-won% → the pickup judgement is fine; the bleed is
PIMC32 execution failure. If god-win% ≈ realized-won% → the net is
genuinely picking up un-winnable hands (overconfident vs god).

Usage: N_DEALS=3000 python god_check.py
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import simulate
from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.vnet.pickup.v18 import Exp18Pickup

EXP_DIR   = Path(__file__).parent
WEIGHTS   = EXP_DIR / "multihead_v18a.pt"
N         = int(os.environ.get("N_DEALS", 3000))
SEED_BASE = 100_000
N_WORKERS = 8

# PIMC32-realized won% from bleeders.py (same seeds/auction) for contrast.
REALIZED = {
    'ulti/hearts': 72.9, 'parti/hearts': 17.5, 'ulti/bells': 73.8,
    'ulti/leaves': 68.0, 'ulti/acorns': 70.7, 'betli/colorless': 12.1,
    'durchmars/colorless': 40.0,
}

_PICKER = None


def _get():
    global _PICKER
    if _PICKER is None:
        _PICKER = Exp18Pickup.load(WEIGHTS)
    return _PICKER


def _run_one(seed):
    p = _get()
    r = simulate(seed, [p, p, p], play_out=False)
    if r['winner_pid'] is None:
        return ('PASS_PENALTY', None)
    pos = pis.build_position(
        hands=[r['sol_hand'], r['def1'], r['def2']], soloist=0, leader=0,
        contract=r['contract'], trump=r['trump'], talon=r['talon'],
    )
    god_win = god_says_soloist_wins(pos, contract=r['contract'])
    bid = f"{r['contract']}/{r['trump'] or 'colorless'}"
    return (bid, bool(god_win))


def main():
    print(f"=== v18a god-winnability of committed hands: N={N} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    by_bid = defaultdict(list)
    done = 0
    with Pool(N_WORKERS) as pool:
        for bid, god_win in pool.imap_unordered(_run_one, seeds, chunksize=8):
            if god_win is not None:
                by_bid[bid].append(god_win)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{N}  wall={time.perf_counter()-t0:.0f}s",
                      flush=True)
    print(f"  wall: {time.perf_counter()-t0:.0f}s\n")

    print("God-winnability of the hands the auction actually committed to:")
    print(f"  {'contract':>22}  {'n':>5}  {'god-win%':>9}  "
          f"{'PIMC32 won%':>11}  {'gap':>6}")
    order = sorted(by_bid, key=lambda b: -len(by_bid[b]))
    for bid in order:
        wins = by_bid[bid]
        n = len(wins)
        gw = 100.0 * sum(wins) / n
        realized = REALIZED.get(bid)
        gap = f"{gw - realized:+.1f}" if realized is not None else "  —"
        rstr = f"{realized:>10.1f}" if realized is not None else "         —"
        print(f"  {bid:>22}  {n:>5}  {gw:>8.1f}  {rstr}  {gap:>6}")

    print("\nReading: god-win% = could a perfect soloist make this exact "
          "hand;\n  PIMC32 won% = did the real (handicapped) soloist make it.\n"
          "  big positive gap ⇒ winnable hands, lost to PIMC32 execution; \n"
          "  gap ≈ 0 ⇒ the net genuinely picked un-winnable hands.")


if __name__ == "__main__":
    main()
