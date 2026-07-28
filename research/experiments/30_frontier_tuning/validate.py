"""Final validation — CURRENT deployed config vs RECOMMENDED combined config, SAME N & deals
(the two sweeps used different N, so measure the total improvement cleanly here).
Metric = harness.evaluate soloist GP/game vs PIMC defenders. Sequential (globals). N high.
"""
import os
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = "/Users/milansimity/Cuccok/kodok/oldtawer/.venv/bin/python"

CONFIGS = {
    "CURRENT (deployed)":     {"FLOOR": "0.70", "DEBIAS_PCTL": "0.80", "DURI_TERIT_MULT": "1.0"},
    "RECOMMENDED (retuned)":  {"FLOOR": "0.80", "DEBIAS_PCTL": "0.85", "DURI_TERIT_MULT": "0.3"},
}
N = os.environ.get("N", "2500")
OUT = os.path.join(_HERE, "validate_results.tsv")


def main():
    open(OUT, "w").close()
    results = {}
    for name, cfg in CONFIGS.items():
        env = dict(os.environ); env.update(cfg)
        env.update({"N": N, "SCORER": "pimc", "KONTRA": "1"})
        print(f"\n=== {name}: {cfg} ===", flush=True)
        r = subprocess.run([_PY, os.path.join(_HERE, "run_one.py")], env=env, cwd=_HERE,
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("FLOOR=")), r.stdout[:300])
        d = dict(kv.split("=", 1) for kv in line.split("\t") if "=" in kv)
        results[name] = d
        with open(OUT, "a") as f:
            f.write(name + "\t" + line + "\n")
        print("  " + line, flush=True)
    if len(results) == 2:
        cur = float(results["CURRENT (deployed)"]["metric"])
        rec = float(results["RECOMMENDED (retuned)"]["metric"])
        print(f"\n===== HEADLINE (N={N}, same deals, soloist GP/game vs PIMC) =====", flush=True)
        print(f"  CURRENT     {cur:+.3f}", flush=True)
        print(f"  RECOMMENDED {rec:+.3f}", flush=True)
        print(f"  Δ = {rec-cur:+.3f} GP/game  (FLOOR 0.70→0.80, DEBIAS 0.80→0.85, DURI_TERIT_MULT 1.0→0.3)", flush=True)


if __name__ == "__main__":
    main()
