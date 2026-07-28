"""Head-to-head tournament: exp 17 (A) vs exp 18 winner variant (B).

Same harness as experiments/17_clean_pickup_net/tournament.py: all 6
mixed seat assignments × N seeds, per-seat-deal GP aggregation.

Usage: VARIANT=c N_DEALS=500 python tournament.py
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import simulate
from ulti.vnet.pickup.v17 import Exp17Pickup
from ulti.vnet.pickup.v18 import Exp18Pickup

EXP_DIR       = Path(__file__).parent
EXP17_WEIGHTS = EXP_DIR.parent / "17_clean_pickup_net" / "multihead_v17.pt"

N         = int(os.environ.get("N_DEALS", 500))
SEED_BASE = 100_000
N_WORKERS = 8
VARIANT   = os.environ.get("VARIANT", "c")

CONFIGS = [
    ('B', 'A', 'A'),
    ('A', 'B', 'A'),
    ('A', 'A', 'B'),
    ('A', 'B', 'B'),
    ('B', 'A', 'B'),
    ('B', 'B', 'A'),
]

_PICKERS = None


def _get_pickers():
    global _PICKERS
    if _PICKERS is None:
        weights = EXP_DIR / f"multihead_v18{VARIANT}.pt"
        loader = Exp17Pickup if VARIANT == 'b' else Exp18Pickup
        _PICKERS = {
            'A': Exp17Pickup.load(EXP17_WEIGHTS),
            'B': loader.load(weights),
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
    print(f"=== Head-to-head: exp17 (A) vs exp18 v18{VARIANT} (B), "
          f"N={N}/config ===")
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

    def model_gp_per_deal(target_letter, only_configs=None):
        total = 0.0; n = 0
        for r in rows:
            if only_configs is not None and r['config'] not in only_configs:
                continue
            for pid in range(3):
                if r['config'][pid] == target_letter:
                    total += r['gps'][pid]
                    n += 1
        return total / n if n else 0.0, n

    print("\n=== Per-model aggregate (per-seat GP/deal) ===")
    a_1, a_1n = model_gp_per_deal('A', only_configs={'BAA','ABA','AAB'})
    b_1, b_1n = model_gp_per_deal('B', only_configs={'BAA','ABA','AAB'})
    a_2, a_2n = model_gp_per_deal('A', only_configs={'ABB','BAB','BBA'})
    b_2, b_2n = model_gp_per_deal('B', only_configs={'ABB','BAB','BBA'})
    print()
    print(f"  Config group: 2 exp17 + 1 exp18  (BAA, ABA, AAB)")
    print(f"    exp17 (majority, 2 seats):  {a_1:+.3f} GP/deal  (n={a_1n})")
    print(f"    exp18 (minority, 1 seat):   {b_1:+.3f} GP/deal  (n={b_1n})")
    print(f"    sanity: 2*{a_1:+.3f} + {b_1:+.3f} = {2*a_1+b_1:+.3f}  (~0)")
    print()
    print(f"  Config group: 1 exp17 + 2 exp18  (ABB, BAB, BBA)")
    print(f"    exp17 (minority, 1 seat):   {a_2:+.3f} GP/deal  (n={a_2n})")
    print(f"    exp18 (majority, 2 seats):  {b_2:+.3f} GP/deal  (n={b_2n})")
    print(f"    sanity: {a_2:+.3f} + 2*{b_2:+.3f} = {a_2+2*b_2:+.3f}  (~0)")

    a_all, a_alln = model_gp_per_deal('A')
    b_all, b_alln = model_gp_per_deal('B')
    print()
    print(f"=== Headline (all mixed configs) ===")
    print(f"  exp17 (A) overall: {a_all:+.3f} GP/seat-deal  (n={a_alln})")
    print(f"  exp18 (B) overall: {b_all:+.3f} GP/seat-deal  (n={b_alln})")
    print(f"  Δ (B − A):          {b_all - a_all:+.3f}")


if __name__ == "__main__":
    main()
