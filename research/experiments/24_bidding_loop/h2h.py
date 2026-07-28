"""Head-to-head: bidder A vs bidder B in the same auctions.

A is rotated through all 3 seats (B in the other two) to remove seat bias; each
deal is played 3 times. Reports A's mean GP/deal minus B's — the direct "is A a
better bidder than B" signal — plus what A wins that B can't.

Usage: set make_A / make_B (module-level factories), pick a scorer (god|pimc).
Default: full-ladder net bidder (A) vs the 4-contract-restricted bidder (B) →
the value of the 29 extra contracts.

Env: N, WORKERS, SCORER (god|pimc), PIMC_N.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from multiprocessing import get_context

_HERE = os.path.dirname(os.path.abspath(__file__))
_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for p in (_HERE, _E23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from auction import run_auction          # noqa: E402

_A = _B = _SCORER = None


def _init(make_A, make_B, scorer):
    global _A, _B, _SCORER
    _A, _B, _SCORER = make_A(), make_B(), scorer


def _job(args):
    seed, pos = args                       # pos = A's seat this round
    fns = [_B, _B, _B]
    fns[pos] = _A
    r = run_auction(seed, fns)
    if r["winner"] is None:
        gps = [-4.0, 2.0, 2.0]
        a_won = None
    else:
        gp, _ = _SCORER(r["rung"], r["trump"], r["sol"], r["def1"], r["def2"],
                        r["talon"], seed=seed)
        w = r["winner"]
        gps = [-gp, -gp, -gp]
        gps[w] = 2 * gp
        a_won = r["contract"] if w == pos else None
    a = gps[pos]
    b = sum(gps[s] for s in range(3) if s != pos) / 2.0
    return a, b, a_won


def h2h(make_A, make_B, n=600, seed_base=600_000_000, workers=8, scorer=None):
    if scorer is None:
        from scorers import god_outcome
        scorer = god_outcome
    jobs = [(seed_base + i, pos) for i in range(n) for pos in range(3)]
    A = B = 0.0
    a_wins = Counter()
    t0 = time.perf_counter()
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(make_A, make_B, scorer)) as pool:
        for a, b, won in pool.imap_unordered(_job, jobs, chunksize=4):
            A += a
            B += b
            if won is not None:
                a_wins[won] += 1
    m = len(jobs)
    wall = time.perf_counter() - t0
    return {"n": n, "A": A / m, "B": B / m, "edge": (A - B) / m,
            "a_wins": a_wins, "wall": wall}


if __name__ == "__main__":
    from net_bidder import make_net_bid_fn, make_old4_bid_fn
    from scorers import god_outcome, pimc_outcome
    which = os.environ.get("SCORER", "god")
    res = h2h(make_net_bid_fn, make_old4_bid_fn,
              n=int(os.environ.get("N", "600")),
              scorer=(pimc_outcome if which == "pimc" else god_outcome))
    print(f"\n=== H2H full-ladder (A) vs 4-contract (B) | scorer={which} "
          f"N={res['n']} wall={res['wall']:.0f}s ===")
    print(f"A GP/deal {res['A']:+.3f}   B GP/deal {res['B']:+.3f}   "
          f"A EDGE {res['edge']:+.3f}")
    print("A wins on contracts B can't bid:")
    for name, c in res["a_wins"].most_common(12):
        print(f"  {c:>5}  {name}")
