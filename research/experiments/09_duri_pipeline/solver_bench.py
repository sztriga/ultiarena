"""Durchmars god solver benchmark — colored + colorless × úr ON/OFF A/B.

Mirrors the pattern from /tmp/parti_betli_bench.py: same seed across the
two A/B arms so any timing delta is attributable to the predicate.

Usage:
    PYTHONPATH=. python3 experiments/09_duri_pipeline/solver_bench.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
from statistics import mean, median

from ulti.eval.dojo import deal_durchmars_colored, deal_durchmars_colorless
from ulti.solvers.pis import build_position, solve_all
from trickster._solver_core import set_dm_proven_safe

N        = 200
OUT_DIR  = Path(__file__).parent


def bench(name: str, deal_fn, alpha: float):
    deals = [deal_fn(seed=10_000 + i, alpha=alpha) for i in range(N)]
    d0 = deals[0]
    pos0 = build_position(
        hands=[d0.sol_hand, d0.def1_hand, d0.def2_hand],
        soloist=0, leader=0, contract="durchmars",
        trump=d0.trump, talon=list(d0.talon))
    solve_all(pos0, contract="durchmars")          # warm

    vals, times = [], []
    t0_all = time.perf_counter()
    for d in deals:
        pos = build_position(
            hands=[d.sol_hand, d.def1_hand, d.def2_hand],
            soloist=0, leader=0, contract="durchmars",
            trump=d.trump, talon=list(d.talon))
        t0 = time.perf_counter()
        r = solve_all(pos, contract="durchmars")
        times.append((time.perf_counter() - t0) * 1000)
        vals.append(max(r.values()))
    wall = time.perf_counter() - t0_all
    st = sorted(times)
    win_rate = sum(1 for v in vals if v > 5) / N
    return {
        "name": name, "alpha": alpha, "n": N,
        "wall_s": wall, "win_rate": win_rate,
        "mean_ms": mean(times), "median_ms": median(times),
        "p90_ms": st[int(N * 0.90)], "p99_ms": st[int(N * 0.99)],
        "max_ms": max(times),
    }


def main() -> None:
    rows = []
    for proven_safe in (1, 0):
        set_dm_proven_safe(proven_safe)
        tag = "úr ON" if proven_safe else "úr OFF"
        print(f"\n############  proven_safe = {tag}  ############")
        for label, fn, alpha in [
            ("colored",   deal_durchmars_colored,   0.6),
            ("colorless", deal_durchmars_colorless, 1.5),
        ]:
            r = bench(f"{label} ({tag})", fn, alpha)
            rows.append(r)
            print(f"\n== {r['name']} (N={N}, α={alpha}) ==")
            print(f"  total : {r['wall_s']:.2f}s    win_rate={r['win_rate']:.1%}")
            print(f"  mean  : {r['mean_ms']:.1f} ms")
            print(f"  median: {r['median_ms']:.1f} ms")
            print(f"  p90   : {r['p90_ms']:.1f} ms")
            print(f"  p99   : {r['p99_ms']:.1f} ms")
            print(f"  max   : {r['max_ms']:.1f} ms")

    (OUT_DIR / "solver_bench.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT_DIR / 'solver_bench.json'}")


if __name__ == "__main__":
    main()
