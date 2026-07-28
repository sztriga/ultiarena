"""Sweep α from 0 to 2 (step 0.2), N seeds per α, Cython ulti solver.

For ulti, α biases the soloist's hand toward strong cards (the soloist
always also receives the trump-7 by mandatory_trump_ranks). At α=0 hands
are uniform random; higher α means a stronger soloist hand.

Records per-α: sol win rate (binary), mean / median / max solve time.

Usage:
    PYTHONPATH=. python3 experiments/08_ulti_alpha_sweep/alpha_sweep.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean, median

from eval.dojo import deal_ulti_biased
from solvers import pis as pis_bridge


ALPHAS = [round(0.2 * i, 1) for i in range(11)]   # 0.0, 0.2, ..., 2.0
N_PER_ALPHA = 50
SEED_BASE = 20260530
_ULTI_WIN_VAL = 10.0
OUT_DIR = Path(__file__).parent


def solve_one(seed: int, alpha: float) -> tuple[str, float]:
    deal = deal_ulti_biased(seed=seed, alpha=alpha)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract="ulti",
        trump=deal.trump, talon=list(deal.talon),
    )
    t0 = time.perf_counter()
    vals = pis_bridge.solve_all(pos, contract="ulti")
    dt = time.perf_counter() - t0
    best = max(vals.values())
    verdict = "soloist" if best >= _ULTI_WIN_VAL - 1e-6 else "defenders"
    return verdict, dt


def main() -> None:
    rng = random.Random(SEED_BASE)
    seeds = [rng.randint(0, 10**9) for _ in range(N_PER_ALPHA)]
    print(f"seeds: {N_PER_ALPHA} fixed across α values\n")

    rows = []
    for a in ALPHAS:
        t_alpha0 = time.perf_counter()
        verdicts: list[str] = []
        times: list[float] = []
        for s in seeds:
            v, dt = solve_one(s, a)
            verdicts.append(v)
            times.append(dt)
        sol_wr = verdicts.count("soloist") / len(verdicts)
        row = {
            "alpha": a,
            "n": len(verdicts),
            "sol_win_rate": sol_wr,
            "mean_s": mean(times),
            "median_s": median(times),
            "max_s": max(times),
            "total_s": sum(times),
        }
        rows.append(row)
        print(
            f"  α={a:>3}  sol_wr={sol_wr:>5.1%}  "
            f"mean={row['mean_s']:>7.3f}s  med={row['median_s']:>7.3f}s  "
            f"max={row['max_s']:>7.2f}s  total={row['total_s']:>7.2f}s  "
            f"wall={time.perf_counter() - t_alpha0:>6.1f}s",
            flush=True,
        )

    out = {"n_per_alpha": N_PER_ALPHA, "seed_base": SEED_BASE, "rows": rows}
    (OUT_DIR / "sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {OUT_DIR / 'sweep.json'}")


if __name__ == "__main__":
    main()
