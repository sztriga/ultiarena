"""Phase 1b — split the perception headroom into TRAINABLE vs IRREDUCIBLE.

Same deals, same scorer: h2h(god, net) and h2h(pimc, net).
  net → PIMC (pimc edge)        = trainable headroom (better nets can buy this)
  PIMC → god (god − pimc edge)  = irreducible info gap (can't be trained away)
Env: N, N_DET.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/24_bidding_loop",
          "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration",
          "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from h2h import h2h                          # noqa: E402
from net_bidder import make_net_bid_fn       # noqa: E402
from scorers import god_outcome              # noqa: E402
from god_provider import make_god_bid_fn     # noqa: E402
from pimc_provider import make_pimc_bid_fn   # noqa: E402

N = int(os.environ.get("N", "100"))

print(f"[ceiling-split] N={N} N_DET={os.environ.get('N_DET','8')} — running god h2h...",
      flush=True)
g = h2h(make_god_bid_fn, make_net_bid_fn, n=N, scorer=god_outcome)
print(f"  god edge {g['edge']:+.3f}  ({g['wall']:.0f}s). running pimc h2h...", flush=True)
p = h2h(make_pimc_bid_fn, make_net_bid_fn, n=N, scorer=god_outcome)
print(f"  pimc edge {p['edge']:+.3f}  ({p['wall']:.0f}s)", flush=True)

total = g["edge"]
trainable = p["edge"]
irreducible = total - trainable
print(f"\n=== PERCEPTION HEADROOM SPLIT | N={N} ===")
print(f"  total (net → god, perfect info)     {total:+.3f} GP/deal")
print(f"  TRAINABLE (net → PIMC marginal)     {trainable:+.3f} GP/deal"
      f"  ({100*trainable/total:.0f}% of total)" if total else "")
print(f"  irreducible (PIMC → god, info gap)  {irreducible:+.3f} GP/deal"
      f"  ({100*irreducible/total:.0f}% of total)" if total else "")
