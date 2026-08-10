"""exp47 — the overnight driver. Two tracks, one wall-clock budget, unattended.

TRACK A — the kontra research milan commissioned: what signals can a defender actually
get, per contract, and what are they worth in GP.
TRACK B — a full audit of the current model: every knob and promoted component, priced
individually against the deployed frontier.

Both are re-runs from scratch, because everything measured before 2026-08-02 was measured
against a bidder that read the talon before deciding whether to enter the auction, and
three of my own harnesses were additionally running it UNCALIBRATED with the betli heads
off. Nothing old is trusted here.

HOW IT PACES ITSELF. Each phase declares an estimated cost. The driver runs them in
priority order and SKIPS any phase whose estimate does not fit the remaining budget,
rather than starting something it cannot finish — a half-finished ablation is worse than
a missing one, because the partial file looks like a result. Everything is resumable, so
a skipped phase just runs next time. What was skipped is printed, loudly, in the report:
silent truncation reads as "we covered everything" when we did not.

Run:   HOURS=7 WORKERS=8 python3 run_night.py
       python3 run_night.py report        # re-print without running anything
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPS = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_EXPS, "43_kontra_signals"),
           os.path.join(_EXPS, "44_frontier_table")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HOURS = float(os.environ.get("HOURS", "7"))
WORKERS = int(os.environ.get("WORKERS", "8"))
LOG = os.path.join(_HERE, "night.log")
STATE = os.path.join(_HERE, "state.json")
REPORT = os.path.join(_HERE, "REPORT.md")

NATURAL = os.path.join(_HERE, "natural.jsonl")
FORCED = os.path.join(_HERE, "forced.jsonl")
GATE_DIR = os.path.join(_HERE, "gates")

# perf_counter, NOT time.time(): on macOS the wall clock keeps running while the machine
# sleeps, so an overnight run that gets suspended wakes up believing its whole budget is
# spent and skips every remaining phase. (2026-08-03: 9h14m wall clock, 1h47m of actual
# compute.) perf_counter excludes suspended time, which is the quantity a compute budget
# actually means. Run under `caffeinate -is` as well — not sleeping is better than
# accounting for it.
_t0 = time.perf_counter()
_results: dict = {}


def log(msg=""):
    el = time.perf_counter() - _t0
    line = f"[{el/3600:5.2f}h] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
        f.flush()


def remaining_h():
    return HOURS - (time.perf_counter() - _t0) / 3600.0


def _save():
    with open(STATE, "w") as f:
        json.dump({"results": _results, "elapsed_h": (time.perf_counter() - _t0) / 3600.0}, f,
                  indent=1, default=str)


def phase(name, est_h, fn, *args, **kw):
    """Run `fn` if it fits the remaining budget. Never starts what it cannot finish."""
    if name in _results and _results[name].get("status") == "done":
        log(f"SKIP {name}: already done")
        return _results[name].get("value")
    rem = remaining_h()
    if est_h > rem:
        log(f"SKIP {name}: needs ~{est_h:.1f}h, only {rem:.1f}h left")
        _results[name] = {"status": "skipped", "reason": f"needs {est_h:.1f}h, had {rem:.1f}h"}
        _save()
        return None
    log(f"START {name}  (est {est_h:.1f}h, {rem:.1f}h left)")
    t = time.perf_counter()
    try:
        val = fn(*args, **kw)
        _results[name] = {"status": "done", "hours": (time.perf_counter() - t) / 3600.0, "value": val}
        log(f"DONE  {name}  in {(time.perf_counter()-t)/3600:.2f}h")
        _save()
        return val
    except Exception as e:
        log(f"FAIL  {name}: {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1200:])
        _results[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        _save()
        return None


# ── Track A ─────────────────────────────────────────────────────────────────────

def _natural(n):
    import datagen
    datagen.PLAYED = NATURAL
    datagen.WORKERS = WORKERS
    datagen.build(n, out_path=NATURAL)
    kept = sum(1 for l in open(NATURAL) if '"kept": true' in l)
    return {"deals": n, "kept": kept}


def _forced(per_unit):
    import forced
    forced.OUT = FORCED
    return forced.build(per_unit, workers=WORKERS, log=log)


def _parti():
    import parti
    parti.OUT = os.path.join(_HERE, "parti.jsonl")
    parti.run([NATURAL, FORCED], workers=WORKERS, log=log)
    parti.report(log=log)
    return {"rows": sum(1 for _ in open(parti.OUT))}


def _signals():
    """exp43's featuriser + GP-priced signal search, over BOTH corpora.
    Natural and forced are analysed separately: forced rows have no auction, so an
    auction-feature rule cannot be compared across them."""
    import features
    import evaluate
    out = {}
    for path, label in ((NATURAL, "natural"), (FORCED, "forced")):
        if not os.path.exists(path):
            continue
        log(f"\n########## KONTRA SIGNALS — {label.upper()} corpus ##########")
        tables = features.build_table(path)
        features.assert_no_leak(tables)
        evaluate.report(tables)
        out[label] = {u: int(len(d["y"])) for u, d in tables.items()}
    return out


# ── Track B ─────────────────────────────────────────────────────────────────────

def _ablation(name, cand):
    import gate_lib
    from ablate import INCUMBENT
    os.makedirs(GATE_DIR, exist_ok=True)
    return gate_lib.run(name, INCUMBENT, cand, ABL_DEALS, 473_000_000,
                        os.path.join(GATE_DIR, f"{name}.jsonl"), workers=WORKERS, log=log)


ABL_DEALS = int(os.environ.get("ABL_DEALS", "400"))


# ── report ──────────────────────────────────────────────────────────────────────

def write_report():
    lines = ["# exp47 — overnight research report", ""]
    lines.append(f"Budget {HOURS:.1f}h · elapsed {(time.perf_counter()-_t0)/3600:.2f}h · "
                 f"{WORKERS} workers · {ABL_DEALS} deals/ablation")
    lines.append("")
    done = [k for k, v in _results.items() if v.get("status") == "done"]
    skipped = [k for k, v in _results.items() if v.get("status") == "skipped"]
    failed = [k for k, v in _results.items() if v.get("status") == "failed"]
    lines += ["## Coverage", "",
              f"- completed: {len(done)}",
              f"- **skipped (not measured): {len(skipped)}**" if skipped else "- skipped: none",
              f"- **FAILED: {len(failed)}**" if failed else "- failed: none", ""]
    for k in skipped:
        lines.append(f"  - `{k}` — {_results[k]['reason']}")
    for k in failed:
        lines.append(f"  - `{k}` — {_results[k]['error']}")
    lines.append("")

    lines += ["## Track B — what each moving part is worth", "",
              "Rotation gate vs the deployed frontier. Candidate rotates through all three "
              "seats, so **0.000 = parity** and a positive delta means the CHANGE is better "
              "than what ships. Aggregate only — per-seat numbers need a matched control "
              "(see gate_lib).", "",
              "| change | delta GP/seat-deal | se | t | verdict |", "|---|---|---|---|---|"]
    from ablate import ABLATIONS
    for name, _cand, _why in ABLATIONS:
        r = _results.get(f"ablate:{name}")
        if not r or r.get("status") != "done" or not r.get("value"):
            lines.append(f"| `{name}` | — | — | — | not measured |")
            continue
        v = r["value"]
        t = v.get("t", 0.0)
        verdict = ("**change is BETTER**" if t > 2 else
                   "**change is WORSE (component earns its place)**" if t < -2 else
                   "no detectable difference")
        lines.append(f"| `{name}` | {v['delta']:+.3f} | {v['se']:.3f} | {t:+.2f} | {verdict} |")
    lines += ["", "Interpretation: a strongly NEGATIVE delta means turning the component off "
              "costs GP, i.e. the component is earning its keep. A delta near zero on a knob "
              "means the knob is not binding and could be simplified away.", ""]

    lines += ["## Track A — kontra", "",
              "Full per-unit signal tables are in `night.log` (search for `KONTRA SIGNALS`); "
              "the parti deep-dive is under `PARTI DEEP-DIVE`.", ""]
    for k in ("natural", "forced", "parti", "signals"):
        r = _results.get(k)
        if r:
            lines.append(f"- `{k}`: {r.get('status')} — {r.get('value')}")
    lines.append("")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"report written: {REPORT}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        if os.path.exists(STATE):
            _results.update(json.load(open(STATE))["results"])
        write_report()
        return
    if os.path.exists(STATE):
        _results.update(json.load(open(STATE)).get("results", {}))
        log(f"resuming: {len(_results)} phases already recorded")

    log(f"exp47 overnight — budget {HOURS:.1f}h, {WORKERS} workers, "
        f"{ABL_DEALS} deals/ablation")
    log("Track A = kontra research; Track B = full model audit.")

    # ── Track A: corpora first, everything else depends on them ──
    phase("natural", 1.7, _natural, 12000)
    phase("forced", 1.2, _forced, 900)
    phase("parti", 0.5, _parti)
    phase("signals", 0.3, _signals)

    # ── Track B: ablations, in priority order, until the budget runs out ──
    from ablate import ABLATIONS
    per = max(0.1, ABL_DEALS / 27.0 / 60.0)          # ~27 deals/min at 8 workers
    for name, cand, why in ABLATIONS:
        log(f"\n--- ablation `{name}`: {why}")
        phase(f"ablate:{name}", per, _ablation, name, cand)

    write_report()
    log("ALL DONE")


if __name__ == "__main__":
    main()
