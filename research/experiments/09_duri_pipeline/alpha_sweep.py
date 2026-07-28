"""Durchmars α sweep — god-solver win rate vs α for colored + colorless.

Mirrors experiments/08_ulti_alpha_sweep/alpha_sweep.py. Records per-α
soloist win rate (binary 10/0) plus solve timings, for both modes.

Usage:
    PYTHONPATH=. python3 experiments/09_duri_pipeline/alpha_sweep.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
from statistics import mean, median

from ulti.eval.dojo import deal_durchmars_colored, deal_durchmars_colorless
from ulti.solvers.pis import build_position, solve_all

# Colored uses parti/ulti-style α scaling; colorless uses the inverted-
# betli scale. We pick α ranges where each mode produces a real S-curve.
ALPHAS_COLORED   = [round(0.2 * i, 1) for i in range(11)]   # 0.0..2.0 step 0.2
ALPHAS_COLORLESS = [round(0.2 * i, 1) for i in range(16)]   # 0.0..3.0 step 0.2
N_PER_ALPHA      = 50
SEED_BASE        = 20260530
_WIN_VAL         = 10.0
OUT_DIR          = Path(__file__).parent


def solve_one(deal):
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = build_position(
        hands=hands, soloist=0, leader=0, contract="durchmars",
        trump=deal.trump, talon=list(deal.talon),
    )
    t0 = time.perf_counter()
    vals = solve_all(pos, contract="durchmars")
    dt = time.perf_counter() - t0
    best = max(vals.values())
    return ("soloist" if best >= _WIN_VAL - 1e-6 else "defenders"), dt


def sweep(mode: str, deal_fn, alphas):
    print(f"\n############  {mode}  ############")
    rows = []
    for a in alphas:
        t_alpha0 = time.perf_counter()
        verdicts, times = [], []
        for i in range(N_PER_ALPHA):
            d = deal_fn(seed=SEED_BASE + i, alpha=a)
            v, dt = solve_one(d)
            verdicts.append(v); times.append(dt)
        sol_wr = verdicts.count("soloist") / len(verdicts)
        row = {
            "mode": mode, "alpha": a, "n": N_PER_ALPHA,
            "sol_win_rate": sol_wr,
            "mean_s":  mean(times),  "median_s": median(times),
            "max_s":   max(times),   "total_s":  sum(times),
        }
        rows.append(row)
        print(
            f"  α={a:>3}  sol_wr={sol_wr:>5.1%}  "
            f"mean={row['mean_s']:>7.3f}s  med={row['median_s']:>7.3f}s  "
            f"max={row['max_s']:>7.2f}s  wall={time.perf_counter() - t_alpha0:>6.1f}s",
            flush=True,
        )
    return rows


def main() -> None:
    out = {
        "n_per_alpha": N_PER_ALPHA,
        "seed_base": SEED_BASE,
        "colored":   sweep("COLORED",   deal_durchmars_colored,   ALPHAS_COLORED),
        "colorless": sweep("COLORLESS", deal_durchmars_colorless, ALPHAS_COLORLESS),
    }
    (OUT_DIR / "alpha_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'alpha_sweep.json'}")


if __name__ == "__main__":
    main()
