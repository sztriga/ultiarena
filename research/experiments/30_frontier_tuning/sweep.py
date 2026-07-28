"""Config-sweep driver — runs run_one.py once per (FLOOR, DEBIAS_PCTL) in its own process
(FLOOR is a bidder.py import-time global). pimc scorer (realistic). Sequential to avoid
core contention. Results accumulate in sweep_results.tsv; ranked print at the end.
Env: N (per config), SCORER, WORKERS, GRID (coarse|fine).
"""
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = "/Users/milansimity/Cuccok/kodok/oldtawer/.venv/bin/python"

GRID = os.environ.get("GRID", "coarse")
if GRID == "fine":
    CONFIGS = [(f, d) for f in ("0.60", "0.65", "0.70", "0.75", "0.80")
               for d in ("0.75", "0.80", "0.85")]
else:
    CONFIGS = [(f, d) for f in ("0.60", "0.70", "0.80") for d in ("0.75", "0.80", "0.85")]

N = os.environ.get("N", "1500")
SCORER = os.environ.get("SCORER", "pimc")


def main():
    print(f"config sweep: {len(CONFIGS)} configs × N={N} ({SCORER} scorer)", flush=True)
    t0 = time.perf_counter()
    for i, (floor, debias) in enumerate(CONFIGS, 1):
        env = dict(os.environ)
        # KONTRA=1 matches the deployed bidder (passes weak hands) — FLOOR interacts with it.
        env.update({"FLOOR": floor, "DEBIAS_PCTL": debias, "N": N, "SCORER": SCORER, "KONTRA": "1"})
        print(f"\n[{i}/{len(CONFIGS)}] FLOOR={floor} DEBIAS={debias}  "
              f"(elapsed {(time.perf_counter()-t0)/60:.0f}m)", flush=True)
        subprocess.run([_PY, os.path.join(_HERE, "run_one.py")], env=env, cwd=_HERE)
    # ranked summary
    rows = []
    p = os.path.join(_HERE, "sweep_results.tsv")
    if os.path.exists(p):
        for line in open(p):
            d = dict(kv.split("=", 1) for kv in line.strip().split("\t") if "=" in kv)
            if d.get("scorer") == SCORER:
                rows.append(d)
    rows.sort(key=lambda d: float(d["metric"]), reverse=True)
    print(f"\n===== RANKED ({SCORER}, by metric) =====", flush=True)
    for d in rows:
        print(f"  FLOOR={d['FLOOR']} DEBIAS={d['DEBIAS']}  metric={d['metric']}  "
              f"nonfloor={d['nonfloor']} (n={d['n_nonfloor']})  pass={d['pass']}  P0={d.get('P0','?')}",
              flush=True)


if __name__ == "__main__":
    main()
