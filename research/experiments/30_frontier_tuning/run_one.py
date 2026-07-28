"""Evaluate ONE bidder config (FLOOR, DEBIAS_PCTL from env) with the fixed bidder.

Strength = harness.evaluate METRIC (mean soloist GP/game vs a FIXED PIMC defender) — config-
comparable (the defender is config-independent, so it's not zero-sum). Also the NON-FLOOR GP
(escalations above piros parti — the discriminating signal). Appends one row to sweep_results.tsv.
FLOOR is a bidder.py import-time global, so each config MUST run in its own process (this script).
Env: FLOOR, DEBIAS_PCTL, N, SCORER (pimc|kpimc|god), WORKERS.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (f"{_REPO}/experiments/24_bidding_loop",
           f"{_REPO}/experiments/23_bidding_integration",
           f"{_REPO}/experiments/14_minigame_bid_eval", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness import evaluate                      # noqa: E402
from net_bidder import make_net_bid_fn            # noqa: E402
from scorers import pimc_outcome, god_outcome, kontra_pimc_outcome  # noqa: E402

SCORER = {"pimc": pimc_outcome, "god": god_outcome, "kpimc": kontra_pimc_outcome}[
    os.environ.get("SCORER", "pimc")]
N = int(os.environ.get("N", "1200"))
FLOOR = os.environ.get("FLOOR", "?")
DEBIAS = os.environ.get("DEBIAS_PCTL", "?")

res = evaluate(make_net_bid_fn, n=N, workers=int(os.environ.get("WORKERS", "8")), scorer=SCORER)
row = (f"FLOOR={FLOOR}\tDEBIAS={DEBIAS}\tDURIMULT={os.environ.get('DURI_TERIT_MULT','1.0')}\t"
       f"scorer={os.environ.get('SCORER','pimc')}\t"
       f"N={N}\tmetric={res['metric']:+.3f}\tnonfloor={res['nonfloor']:+.3f}\t"
       f"n_nonfloor={res['n_nonfloor']}\tpass={res['pass_rate']:.3f}\t"
       f"P0={res['seat_gp'][0]:+.3f}\twall={res['wall']:.0f}s")
print(row, flush=True)
with open(os.path.join(_HERE, "sweep_results.tsv"), "a") as f:
    f.write(row + "\n")
