"""Precompute the god-exact leaf payoff table over a fixed deal pool.

For each deal and each (player, action): the value-model picks its best
realization (suit, discard) from the real 12, then ONE god solve gives the
exact win/loss → EV/def. Also stores deployable availability and the raw-10
bucket per player. Cached to npz; the CFR trainer reuses it across all
iterations.

Usage: N_DEALS=50000 SEED_BASE=300000000 python leaves.py
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))

from _lib import _ev_per_def                          # noqa: E402
from solvers import pis                               # noqa: E402
from eval.pimc_matchup import god_says_soloist_wins   # noqa: E402
from vnet.pickup.composite import CompositePickup     # noqa: E402

from common import (ACTIONS, N_ACTIONS, deal_10_10_10_2,           # noqa: E402
                    action_realization)
from buckets import bucket_of_hand                    # noqa: E402

EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent.parent / "19_colorless_split"

N_DEALS   = int(os.environ.get("N_DEALS", 50000))
SEED_BASE = int(os.environ.get("SEED_BASE", 300_000_000))
N_WORKERS = int(os.environ.get("N_WORKERS", 8))
OUT = Path(__file__).parent / f"leaves_{N_DEALS}.npz"

_PICKER = None


def _picker():
    global _PICKER
    if _PICKER is None:
        _PICKER = CompositePickup.load(
            trump_weights=EXP18 / "multihead_v18a.pt",
            betli_weights=EXP19 / "colorless_betli.pt",
            durchmars_weights=EXP19 / "colorless_durchmars.pt",
        )
    return _PICKER


def _leaf_one(seed):
    pk = _picker()
    h = list(deal_10_10_10_2(seed))
    talon = h[3]
    players = h[:3]
    ev = np.full((3, N_ACTIONS), np.nan, dtype=np.float32)
    avail = np.zeros((3, N_ACTIONS), dtype=bool)
    bucket = np.zeros(3, dtype=np.int32)
    for p in range(3):
        solh = players[p]
        defA = players[(p + 1) % 3]
        defB = players[(p + 2) % 3]
        bucket[p] = bucket_of_hand(pk, solh)
        for ai, action in enumerate(ACTIONS):
            r = action_realization(pk, solh, talon, action)
            if r is None:
                continue
            pos = pis.build_position(
                hands=[r['sol10'], list(defA), list(defB)], soloist=0,
                leader=0, contract=r['contract'], trump=r['trump'],
                talon=r['discard'])
            win = god_says_soloist_wins(pos, contract=r['contract'])
            piros = (r['trump'] == 'hearts')
            ev[p, ai] = _ev_per_def(r['contract'], piros, 1.0 if win else 0.0)
            avail[p, ai] = True
    return ev, avail, bucket


def main():
    print(f"=== leaf precompute  N={N_DEALS}  base={SEED_BASE} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N_DEALS)]
    EV = np.empty((N_DEALS, 3, N_ACTIONS), dtype=np.float32)
    AV = np.empty((N_DEALS, 3, N_ACTIONS), dtype=bool)
    BK = np.empty((N_DEALS, 3), dtype=np.int32)
    t0 = time.perf_counter()
    with Pool(N_WORKERS) as pool:
        for i, (ev, av, bk) in enumerate(
                pool.imap(_leaf_one, seeds, chunksize=64)):
            EV[i], AV[i], BK[i] = ev, av, bk
            if (i + 1) % 5000 == 0:
                el = time.perf_counter() - t0
                print(f"  {i+1}/{N_DEALS}  {el:.0f}s  "
                      f"({(i+1)/el:.0f} deals/s)", flush=True)
    el = time.perf_counter() - t0
    print(f"  done {el:.0f}s  ({N_DEALS/el:.0f} deals/s)")

    np.savez_compressed(OUT, ev=EV, avail=AV, bucket=BK,
                        seed_base=SEED_BASE, n_deals=N_DEALS,
                        actions=np.array(ACTIONS))
    print(f"  saved → {OUT.name}  ({OUT.stat().st_size/1e6:.1f} MB)")

    # quick sanity: action availability + god-makeable rates
    print("\naction availability / god-makeable (per player-deal):")
    for ai, a in enumerate(ACTIONS):
        av = AV[:, :, ai]
        made = (EV[:, :, ai] > 0) & av
        print(f"  {a:>11}: avail {av.mean()*100:5.1f}%   "
              f"god-win {made.sum()/max(av.sum(),1)*100:5.1f}% of available")
    n_bk = len(np.unique(BK))
    print(f"\ndistinct buckets realized: {n_bk}")


if __name__ == "__main__":
    main()
