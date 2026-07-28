"""Simple-game eval WITH kontra vs WITHOUT — the 4-contract bidder.

KONTRA=0 → bidder ignores kontra (always declares ≥ piros parti), scored god (no kontra).
KONTRA=1 → bidder prices kontra (weak hands PASS), scored with the kontra decision.

Shows the behavioural change kontra introduces: pass rate rises, the never-pass
floor tax is contained. Env: N, KONTRA.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from harness import evaluate, print_report          # noqa: E402
from net_bidder import make_old4_bid_fn              # noqa: E402
from scorers import god_outcome, kontra_god_outcome  # noqa: E402

KONTRA = os.environ.get("KONTRA", "0").lower() in ("1", "true", "yes")
scorer = kontra_god_outcome if KONTRA else god_outcome
res = evaluate(make_old4_bid_fn, n=int(os.environ.get("N", "2000")), scorer=scorer)
print_report(res, f"4-contract simple game | KONTRA={'on' if KONTRA else 'off'}")
