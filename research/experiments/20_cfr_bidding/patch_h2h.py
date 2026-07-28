"""Head-to-head in the REAL harness: the debiased (patched) bidder vs the
deployed bidder, on auction_h2h.simulate (P0-gets-12 deal, PIMC32 vs god
play-out). Confirms the inflation-removal GP edge in production.

Patched picker = deployed CompositePickup wrapped with a ``debias_pctl``
attribute, which auction_h2h reads per-seat — so patched and deployed sit at
the same table. Lone-patched-hero vs 2-deployed, seat-rotated, paired.

Usage: N_EVAL=2000 PCTL=0.80 python patch_h2h.py
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import simulate                       # noqa: E402
from vnet.pickup.composite import CompositePickup      # noqa: E402

EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent.parent / "19_colorless_split"

N_EVAL    = int(os.environ.get("N_EVAL", 2000))
SEED_BASE = int(os.environ.get("SEED_BASE", 100_000))
N_WORKERS = int(os.environ.get("N_WORKERS", 8))
PCTL      = float(os.environ.get("PCTL", 0.80))

_DEP = _PATCH = None


class PatchedPicker:
    """Deployed picker + a debias_pctl attribute (read by auction_h2h)."""
    def __init__(self, base, pctl):
        self._base = base
        self.debias_pctl = pctl

    def predict(self, X, contract):
        return self._base.predict(X, contract)


def _init():
    global _DEP, _PATCH
    if _DEP is None:
        _DEP = CompositePickup.load(
            trump_weights=EXP18 / "multihead_v18a.pt",
            betli_weights=EXP19 / "colorless_betli.pt",
            durchmars_weights=EXP19 / "colorless_durchmars.pt")
        _PATCH = PatchedPicker(_DEP, PCTL)
    return _DEP, _PATCH


def _worker(seed):
    dep, patch = _init()
    base = simulate(seed, [dep, dep, dep], play_out=True)
    rows = []
    for s in range(3):
        line = [dep, dep, dep]
        line[s] = patch
        hero = simulate(seed, line, play_out=True)
        rows.append({'seat': s,
                     'dep_gp': base['gps'][s], 'patch_gp': hero['gps'][s],
                     'dep_bid': base['winning_bid'] if base['winner_pid'] == s else None,
                     'patch_bid': hero['winning_bid'] if hero['winner_pid'] == s else None})
    return rows


def main():
    print(f"=== real-harness patch h2h  N={N_EVAL}  PCTL={PCTL} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N_EVAL)]
    rows = []
    t0 = time.perf_counter()
    with Pool(N_WORKERS) as pool:
        for rs in pool.imap_unordered(_worker, seeds, chunksize=8):
            rows.extend(rs)
            if len(rows) % 1500 == 0:
                print(f"  {len(rows)//3}/{N_EVAL}  {time.perf_counter()-t0:.0f}s",
                      flush=True)
    print(f"  wall {time.perf_counter()-t0:.0f}s\n")

    d = np.array([r['patch_gp'] - r['dep_gp'] for r in rows])
    se = d.std(ddof=1) / np.sqrt(len(d))
    nd = int((d != 0).sum())
    print(f"  patched − deployed GP/seat-deal: {d.mean():+.4f} ± {1.96*se:.4f}"
          f"  t={d.mean()/se:.1f}")
    print(f"  bids differ on {nd}/{len(d)} ({nd/len(d)*100:.1f}%);  "
          f"Δ on those = {d[d!=0].mean():+.3f}")
    for s in range(3):
        ds = np.array([r['patch_gp'] - r['dep_gp'] for r in rows if r['seat'] == s])
        ses = ds.std(ddof=1) / np.sqrt(len(ds))
        print(f"    seat {s}: Δ {ds.mean():+.3f} ± {1.96*ses:.3f}  t={ds.mean()/ses:.1f}")

    # what the patched hero stops bidding
    from collections import Counter
    dep_bids = Counter(r['dep_bid'] for r in rows if r['dep_bid'])
    pat_bids = Counter(r['patch_bid'] for r in rows if r['patch_bid'])
    print("\n  hero-seat winning-bid counts (deployed → patched):")
    for k in sorted(set(dep_bids) | set(pat_bids)):
        print(f"    {k:>22}: {dep_bids.get(k,0):>4} → {pat_bids.get(k,0):>4}")


if __name__ == "__main__":
    main()
