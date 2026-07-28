"""Head-to-head tournament: exp 16 calibrated v2 vs exp 17 multi-head.

Runs all 6 mixed seat assignments × N seeds. Aggregates per-model GP/deal
across the 1-vs-2 and 2-vs-1 configurations.
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from auction_h2h import simulate
from ulti.vnet.pickup.calibration import CalibratedPickupNet
from ulti.vnet.pickup.v17 import Exp17Pickup

EXP15_DIR = Path(__file__).parent.parent / "15_vnet_pickup"
EXP17_WEIGHTS = Path(__file__).parent / "multihead_v17.pt"

N         = int(os.environ.get("N_DEALS", 500))
SEED_BASE = 100_000
N_WORKERS = 8

# Six mixed seat assignments. 'A' = exp16, 'B' = exp17.
CONFIGS = [
    ('B', 'A', 'A'),   # 1 exp17 in P0
    ('A', 'B', 'A'),   # 1 exp17 in P1
    ('A', 'A', 'B'),   # 1 exp17 in P2
    ('A', 'B', 'B'),   # 2 exp17 (exp16 in P0)
    ('B', 'A', 'B'),   # 2 exp17 (exp16 in P1)
    ('B', 'B', 'A'),   # 2 exp17 (exp16 in P2)
]

_PICKERS = None   # lazy per-worker init


def _get_pickers():
    global _PICKERS
    if _PICKERS is None:
        _PICKERS = {
            'A': CalibratedPickupNet.load(EXP15_DIR),
            'B': Exp17Pickup.load(EXP17_WEIGHTS),
        }
    return _PICKERS


def _run_one(args):
    seed, config = args
    pickers_map = _get_pickers()
    pickers = [pickers_map[c] for c in config]
    res = simulate(seed, pickers)
    res['config'] = ''.join(config)
    return res


def main():
    print(f"=== Head-to-head: exp16 (A) vs exp17 (B), N={N}/config ===")
    print(f"  configs: {[''.join(c) for c in CONFIGS]}")
    print(f"  total deal-runs: {N * len(CONFIGS)}", flush=True)

    jobs = []
    for cfg in CONFIGS:
        for i in range(N):
            jobs.append((SEED_BASE + i, cfg))

    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_run_one, jobs, chunksize=4):
            rows.append(r)
            if len(rows) % 200 == 0:
                wall = time.perf_counter() - t0
                print(f"  {len(rows)}/{len(jobs)}  wall={wall:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    print()
    print(f"Wall: {wall:.0f}s ({wall/60:.1f} min)")

    # Aggregate per (config, seat)
    print("\n=== Per-config breakdown ===")
    print(f"  {'config':>6}  {'n':>4}  "
          f"{'P0(GP/deal)':>12}  {'P1':>8}  {'P2':>8}  "
          f"{'sum':>6}  winners(P0/P1/P2/penalty)")
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r['config'], []).append(r)
    for cfg in sorted(by_cfg):
        cfg_rows = by_cfg[cfg]
        n = len(cfg_rows)
        gps = [sum(r['gps'][pid] for r in cfg_rows) / n for pid in range(3)]
        wd = Counter(r['winner_pid'] for r in cfg_rows)
        print(f"  {cfg:>6}  {n:>4}  "
              f"{gps[0]:>+11.3f}  {gps[1]:>+7.3f}  {gps[2]:>+7.3f}  "
              f"{sum(gps):>+5.3f}  "
              f"{wd[0]}/{wd[1]}/{wd[2]}/{wd.get(None,0)}")

    # Aggregate per model
    # For each row, sum GP that A picked up and GP that B picked up.
    def model_gp_per_deal(target_letter, only_configs=None):
        total = 0.0; n = 0
        for r in rows:
            if only_configs is not None and r['config'] not in only_configs:
                continue
            for pid in range(3):
                if r['config'][pid] == target_letter:
                    total += r['gps'][pid]
                    n += 1
        # n counts model-seat-deals; convert to per-deal by /N_deals or /n?
        # Standard convention: per-deal = total / number_of_deals_played.
        # Each deal has 3 seats; for a config with k A's, A played k seats.
        # We want per-seat-deal: total / n.
        return total / n if n else 0.0, n

    print("\n=== Per-model aggregate (per-seat GP/deal, averaged over seats × deals) ===")
    a_1, a_1n = model_gp_per_deal('A', only_configs={'BAA','ABA','AAB'})  # A is majority
    b_1, b_1n = model_gp_per_deal('B', only_configs={'BAA','ABA','AAB'})  # B is minority
    a_2, a_2n = model_gp_per_deal('A', only_configs={'ABB','BAB','BBA'})  # A is minority
    b_2, b_2n = model_gp_per_deal('B', only_configs={'ABB','BAB','BBA'})  # B is majority
    print()
    print(f"  Config group: 2 exp16 + 1 exp17  (BAA, ABA, AAB)")
    print(f"    exp16 (majority, 2 seats):  {a_1:+.3f} GP/deal  (n={a_1n} seat-deals)")
    print(f"    exp17 (minority, 1 seat):   {b_1:+.3f} GP/deal  (n={b_1n} seat-deals)")
    print(f"    sanity: 2*{a_1:+.3f} + {b_1:+.3f} = {2*a_1+b_1:+.3f}  (should be ~0)")
    print()
    print(f"  Config group: 1 exp16 + 2 exp17  (ABB, BAB, BBA)")
    print(f"    exp16 (minority, 1 seat):   {a_2:+.3f} GP/deal  (n={a_2n} seat-deals)")
    print(f"    exp17 (majority, 2 seats):  {b_2:+.3f} GP/deal  (n={b_2n} seat-deals)")
    print(f"    sanity: {a_2:+.3f} + 2*{b_2:+.3f} = {a_2+2*b_2:+.3f}  (should be ~0)")

    # Headline: model A vs model B overall
    a_all, a_alln = model_gp_per_deal('A')
    b_all, b_alln = model_gp_per_deal('B')
    print()
    print(f"=== Headline (averaged across all mixed configs) ===")
    print(f"  exp16 (A) overall: {a_all:+.3f} GP/seat-deal  (n={a_alln})")
    print(f"  exp17 (B) overall: {b_all:+.3f} GP/seat-deal  (n={b_alln})")
    print(f"  Δ (B − A):          {b_all - a_all:+.3f}")


if __name__ == "__main__":
    main()
