"""Duri-fix sweep — find the DURI_TERIT_MULT that best suppresses the terített-duri over-bid
leak (exp29: made 13-26%, −0.7 GP/deal, the #1 leak). Metric = harness.evaluate soloist GP vs
PIMC defenders (config-comparable, non-zero-sum). Run ON TOP of the RETUNED config from the
config sweep (FLOOR=0.80, DEBIAS=0.85 — best of 9). Sequential subprocesses (DURI_TERIT_MULT is
a bidder.py import-time global). Higher metric = better.
"""
import os
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = "/Users/milansimity/Cuccok/kodok/oldtawer/.venv/bin/python"

MULTS = ["1.0", "0.7", "0.5", "0.3", "0.15", "0.0"]
N = os.environ.get("N", "1500")
OUT = os.path.join(_HERE, "duri_results.tsv")


def main():
    print(f"duri-fix sweep: DURI_TERIT_MULT ∈ {MULTS} × N={N} (pimc, FLOOR=0.7 DEBIAS=0.80)", flush=True)
    open(OUT, "w").close()
    t0 = time.perf_counter()
    for i, mult in enumerate(MULTS, 1):
        env = dict(os.environ)
        env.update({"FLOOR": "0.80", "DEBIAS_PCTL": "0.85", "DURI_TERIT_MULT": mult,
                    "N": N, "SCORER": "pimc", "KONTRA": "1"})
        print(f"\n[{i}/{len(MULTS)}] DURI_TERIT_MULT={mult}  (elapsed {(time.perf_counter()-t0)/60:.0f}m)", flush=True)
        r = subprocess.run([_PY, os.path.join(_HERE, "run_one.py")], env=env, cwd=_HERE,
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("FLOOR=")), r.stdout.strip()[:200])
        with open(OUT, "a") as f:
            f.write(line + "\n")
        print("  " + line, flush=True)
    # ranked
    rows = []
    for line in open(OUT):
        d = dict(kv.split("=", 1) for kv in line.strip().split("\t") if "=" in kv)
        if "metric" in d:
            rows.append(d)
    rows.sort(key=lambda d: float(d["metric"]), reverse=True)
    print("\n===== RANKED (by soloist GP vs PIMC) =====", flush=True)
    for d in rows:
        print(f"  DURIMULT={d.get('DURIMULT','?')}  metric={d['metric']}  nonfloor={d['nonfloor']}  "
              f"pass={d['pass']}", flush=True)


if __name__ == "__main__":
    main()
