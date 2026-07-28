"""Phase-1 ceiling measurement: h2h(god-agent, net-agent) → perception headroom.

god-agent = perfect-info perception; net-agent = current nets. Same composer,
auction, play (god defenders), scoring — ONLY perception differs. The god-agent's
GP edge = the maximum strength buyable by better perception. Env: N, SCORER.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_E24 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/24_bidding_loop"
_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for p in (_HERE, _E24, _E23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from h2h import h2h                          # noqa: E402  (exp24)
from net_bidder import make_net_bid_fn       # noqa: E402  (exp24)
from scorers import god_outcome, pimc_outcome  # noqa: E402  (exp24)
from god_provider import make_god_bid_fn      # noqa: E402  (exp25)

N = int(os.environ.get("N", "120"))
which = os.environ.get("SCORER", "god")
scorer = pimc_outcome if which == "pimc" else god_outcome

res = h2h(make_god_bid_fn, make_net_bid_fn, n=N, scorer=scorer)
print(f"\n=== PERCEPTION CEILING | god-agent vs net-agent | scorer={which} "
      f"N={res['n']} wall={res['wall']:.0f}s ===")
print(f"god-agent GP/deal {res['A']:+.3f}   net-agent GP/deal {res['B']:+.3f}")
print(f"PERCEPTION HEADROOM (god edge) {res['edge']:+.3f} GP/deal")
print("\ngod-agent wins on contracts the net-agent doesn't reach:")
for name, c in res["a_wins"].most_common(12):
    print(f"  {c:>5}  {name}")
