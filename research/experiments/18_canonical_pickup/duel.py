"""v17 vs v18a duel: same deal, both forced as P0 opener, adjudicated
by the god solver (pure pickup judgement, no PIMC32 play handicap).

For each seed both nets make the forced-opener decision on the identical
12-card hand. Each net's chosen (contract, trump, discard) is scored by
the *god* solver on the true layout — i.e. the GP the soloist would get
playing that contract under perfect double-dummy play. This removes the
PIMC32-vs-god confound that muddies the tournament: it asks only "whose
contract choice is actually better on this deal."

We report, on the deals where they DISAGREE, whose pick scores higher
under god — that's the cleanest "who has better pickup judgement" signal.

Usage: N_DEALS=3000 python duel.py
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import (
    _best_bid_above_rank, BID_FLOOR, PASS_PENALTY, contract_rank,
)
from _lib import deal_12_10_10, _ev_per_def
from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.vnet.pickup.v17 import Exp17Pickup
from ulti.vnet.pickup.v18 import Exp18Pickup

EXP_DIR   = Path(__file__).parent
EXP17_W   = EXP_DIR.parent / "17_clean_pickup_net" / "multihead_v17.pt"
EXP18_W   = EXP_DIR / "multihead_v18a.pt"

N         = int(os.environ.get("N_DEALS", 3000))
SEED_BASE = 100_000
N_WORKERS = 8

_NETS = None


def _get_nets():
    global _NETS
    if _NETS is None:
        _NETS = {'v17': Exp17Pickup.load(EXP17_W),
                 'v18': Exp18Pickup.load(EXP18_W)}
    return _NETS


def _god_gp_soloist(pick, sol12, d1, d2):
    """Soloist GP of a net's opener decision under god play on the true
    layout. ``pick`` = (ev, discard, contract, trump, p) or None (pass)."""
    if pick is None:
        return 2 * PASS_PENALTY, 'PASS', False
    _, discard, contract, trump, _ = pick
    final10 = [c for c in sol12 if c not in discard]
    pos = pis.build_position(
        hands=[final10, list(d1), list(d2)], soloist=0, leader=0,
        contract=contract, trump=trump, talon=list(discard),
    )
    win = god_says_soloist_wins(pos, contract=contract)
    piros = (trump == 'hearts')
    gp_per_def = _ev_per_def(contract, piros, 1.0 if win else 0.0)
    return 2 * gp_per_def, f"{contract}/{trump or 'colorless'}", win


def _run_one(seed):
    nets = _get_nets()
    sol12, d1, d2 = deal_12_10_10(seed)
    p17 = _best_bid_above_rank(sol12, picker=nets['v17'],
                               min_rank=0, ev_floor=BID_FLOOR)
    p18 = _best_bid_above_rank(sol12, picker=nets['v18'],
                               min_rank=0, ev_floor=BID_FLOOR)
    gp17, bid17, win17 = _god_gp_soloist(p17, sol12, d1, d2)
    gp18, bid18, win18 = _god_gp_soloist(p18, sol12, d1, d2)
    return {
        'seed': seed,
        'bid17': bid17, 'bid18': bid18,
        'gp17': gp17, 'gp18': gp18,
        'win17': win17, 'win18': win18,
        'agree': bid17 == bid18,
    }


def main():
    print(f"=== v17 vs v18a opener duel (god-adjudicated), N={N} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_run_one, seeds, chunksize=8):
            rows.append(r)
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{N}  wall={time.perf_counter()-t0:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0
    print(f"\n  wall: {wall:.0f}s\n")

    agree = [r for r in rows if r['agree']]
    disagree = [r for r in rows if not r['agree']]
    print(f"Agreement (same contract+trump): {len(agree)}/{N} "
          f"({len(agree)/N*100:.1f}%)")
    print(f"Disagreement:                    {len(disagree)}/{N} "
          f"({len(disagree)/N*100:.1f}%)\n")

    # Overall god-GP per net (forced opener every deal)
    g17 = sum(r['gp17'] for r in rows) / N
    g18 = sum(r['gp18'] for r in rows) / N
    print("Overall god-GP / deal (forced opener, soloist view):")
    print(f"  v17: {g17:+.3f}    v18: {g18:+.3f}    Δ(v18−v17): {g18-g17:+.3f}\n")

    # The clean signal: on disagreements, who scores higher under god
    if disagree:
        d17 = sum(r['gp17'] for r in disagree) / len(disagree)
        d18 = sum(r['gp18'] for r in disagree) / len(disagree)
        v18_better = sum(1 for r in disagree if r['gp18'] > r['gp17'] + 1e-9)
        v17_better = sum(1 for r in disagree if r['gp17'] > r['gp18'] + 1e-9)
        tie = len(disagree) - v18_better - v17_better
        print(f"On the {len(disagree)} disagreement deals (god-adjudicated):")
        print(f"  mean god-GP   v17: {d17:+.3f}    v18: {d18:+.3f}    "
              f"Δ: {d18-d17:+.3f}")
        print(f"  v18 picked better: {v18_better} "
              f"({v18_better/len(disagree)*100:.1f}%)")
        print(f"  v17 picked better: {v17_better} "
              f"({v17_better/len(disagree)*100:.1f}%)")
        print(f"  tie (both win / both lose same): {tie} "
              f"({tie/len(disagree)*100:.1f}%)\n")

        # Where do disagreements come from? cross-tab the contract choices
        print("Disagreement contract pairs (v17 → v18), top 12 by count:")
        pairs = Counter((r['bid17'].split('/')[0], r['bid18'].split('/')[0])
                        for r in disagree)
        for (a, b), n in pairs.most_common(12):
            sub = [r for r in disagree
                   if r['bid17'].split('/')[0] == a
                   and r['bid18'].split('/')[0] == b]
            s17 = sum(r['gp17'] for r in sub) / len(sub)
            s18 = sum(r['gp18'] for r in sub) / len(sub)
            print(f"  {a:>10} → {b:<10}  n={n:>4}  "
                  f"god-GP v17={s17:+.2f}  v18={s18:+.2f}")


if __name__ == "__main__":
    main()
