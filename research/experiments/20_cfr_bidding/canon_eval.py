"""Comprehensive eval of the CANON bidder (vnet greedy + debias, default on).

All 3 seats use CompositePickup through auction_h2h.simulate. Per deal we resolve
the auction, then BOTH god-solve the committed hand and run the PIMC32-vs-god
play-out — so we report bidding economics, the PIMC32 execution handicap, and
auction structure together.

Reports: seat GP, per-contract (freq / PIMC-won% / god-win% / GP-def), auction
length distribution, who wins the auction, overtake patterns.

Usage: N_EVAL=30000 python canon_eval.py     (DEBIAS_BID defaults on = canon)
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import (simulate, _play_pimc_vs_god, _score,   # noqa: E402
                         DEBIAS_BID, DEBIAS_PCTL)
from ulti.solvers import pis                                         # noqa: E402
from ulti.eval.pimc_matchup import god_says_soloist_wins             # noqa: E402
from ulti.vnet.pickup.composite import CompositePickup               # noqa: E402

EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent.parent / "19_colorless_split"

N_EVAL    = int(os.environ.get("N_EVAL", 30000))
SEED_BASE = int(os.environ.get("SEED_BASE", 100_000))
N_WORKERS = int(os.environ.get("N_WORKERS", 8))

_P = None


def _picker():
    global _P
    if _P is None:
        _P = CompositePickup.load(
            trump_weights=EXP18 / "multihead_v18a.pt",
            betli_weights=EXP19 / "colorless_betli.pt",
            durchmars_weights=EXP19 / "colorless_durchmars.pt")
    return _P


def _worker(seed):
    p = _picker()
    r = simulate(seed, [p, p, p], play_out=False)
    if r['winner_pid'] is None:                       # pass-out
        return {'passout': True, 'gps': [-4.0, 2.0, 2.0], 'n_pickups': 0,
                'winner': None, 'bid': 'passout', 'seat_win': None,
                'pimc_won': None, 'god_won': None, 'n_bids': 0}
    pos = pis.build_position(
        hands=[r['sol_hand'], r['def1'], r['def2']], soloist=0, leader=0,
        contract=r['contract'], trump=r['trump'], talon=r['talon'])
    god = god_says_soloist_wins(pos, contract=r['contract'])
    final = _play_pimc_vs_god(sol10=r['sol_hand'], d1=r['def1'], d2=r['def2'],
                              talon=r['talon'], contract=r['contract'],
                              trump=r['trump'], seed=seed * 919)
    gpd = _score(final, contract=r['contract'], piros=(r['trump'] == 'hearts'))
    w = r['winner_pid']
    gps = [-gpd, -gpd, -gpd]
    gps[w] = 2 * gpd
    return {'passout': False, 'gps': gps, 'n_pickups': r['n_pickups'],
            'winner': w, 'bid': r['winning_bid'], 'seat_win': w,
            'pimc_won': gpd > 0, 'god_won': bool(god),
            'n_bids': len(r['bid_seq']), 'bid_seq': r['bid_seq']}


def main():
    print(f"=== CANON bidder eval  N={N_EVAL}  "
          f"DEBIAS_BID={DEBIAS_BID} PCTL={DEBIAS_PCTL} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N_EVAL)]
    rows = []
    t0 = time.perf_counter()
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_worker, seeds, chunksize=16):
            rows.append(r)
            if len(rows) % 5000 == 0:
                print(f"  {len(rows)}/{N_EVAL}  {time.perf_counter()-t0:.0f}s",
                      flush=True)
    dt = time.perf_counter() - t0
    print(f"  wall {dt:.0f}s  ({N_EVAL/dt:.0f}/s)\n")
    N = len(rows)
    played = [r for r in rows if not r['passout']]

    # ── seat GP ──────────────────────────────────────────────────────────────
    print("=== seat GP/deal (P0 = forced opener) ===")
    for s in range(3):
        g = np.array([r['gps'][s] for r in rows])
        print(f"  P{s}: {g.mean():+.4f}  ± {1.96*g.std(ddof=1)/np.sqrt(N):.4f}")
    print(f"  sum: {np.sum([np.mean([r['gps'][s] for r in rows]) for s in range(3)]):+.4f}"
          f"  (≈0, zero-sum check)")

    # ── auction structure ────────────────────────────────────────────────────
    print("\n=== auction structure ===")
    po = sum(r['passout'] for r in rows)
    print(f"  pass-out rate          : {po/N*100:.2f}%  ({po})")
    nb = Counter(r['n_pickups'] for r in rows)
    print(f"  bids per auction (n_pickups = #commits):")
    labels = {0: 'pass-out', 1: 'P0 uncontested', 2: '1 overtake',
              3: '2 overtakes', 4: '3 overtakes'}
    for k in sorted(nb):
        lab = labels.get(k, f'{k-1} overtakes')
        print(f"    {k}: {nb[k]/N*100:5.1f}%  ({nb[k]:>6})  {lab}")
    contested = sum(1 for r in played if r['n_pickups'] > 1)
    print(f"  contested (>1 commit)  : {contested/N*100:.1f}%")
    mean_pick = np.mean([r['n_pickups'] for r in played])
    print(f"  mean commits | played  : {mean_pick:.2f}")
    wseat = Counter(r['winner'] for r in played)
    print(f"  auction winner seat    : " +
          "  ".join(f"P{s} {wseat.get(s,0)/len(played)*100:.1f}%" for s in range(3)))

    # ── per-contract economics + the PIMC32-vs-god handicap ─────────────────
    print("\n=== per-contract (freq / PIMC-won% / god-win% / handicap / GP-def) ===")
    by = defaultdict(list)
    for r in played:
        by[r['bid']].append(r)
    print(f"  {'contract':>22}  {'freq%':>6}  {'PIMCw%':>7}  {'godw%':>6}  "
          f"{'gap':>5}  {'GP/def':>7}")
    for k in sorted(by, key=lambda x: -len(by[x])):
        v = by[k]
        f = len(v) / N * 100
        pw = np.mean([r['pimc_won'] for r in v]) * 100
        gw = np.mean([r['god_won'] for r in v]) * 100
        gpd = np.mean([r['gps'][r['winner']] / 2 for r in v])  # back to per-def
        print(f"  {k:>22}  {f:>6.2f}  {pw:>7.1f}  {gw:>6.1f}  "
              f"{gw-pw:>+5.1f}  {gpd:>+7.2f}")

    # ── overall handicap ─────────────────────────────────────────────────────
    gw = np.mean([r['god_won'] for r in played]) * 100
    pw = np.mean([r['pimc_won'] for r in played]) * 100
    print(f"\n  OVERALL committed contract: god-makeable {gw:.1f}%  "
          f"PIMC32-made {pw:.1f}%  → execution handicap {gw-pw:+.1f}pp")

    # ── who overtakes whom ───────────────────────────────────────────────────
    print("\n=== overtakes (contested auctions only) ===")
    ot = Counter()
    for r in played:
        bs = r.get('bid_seq', [])
        for i in range(1, len(bs)):
            ot[(bs[i-1][0], bs[i][0])] += 1
    tot = sum(ot.values())
    if tot:
        for (a, b), n in ot.most_common(6):
            print(f"  P{a} → overtaken by P{b}: {n/tot*100:.1f}%  ({n})")


if __name__ == "__main__":
    main()
