"""External-sampling MCCFR over the standard Ulti auction.

Reads the cached god-leaf table (leaves.py). Each iteration samples one deal
(chance), picks a traverser, and walks the bidding tree: traverser nodes
explore all actions and update counterfactual regret; opponent nodes sample
one action and accumulate the average strategy. Belief-updating is implicit —
the regrets at "(my bucket, opponents bid X)" are reach-weighted over exactly
the deals where that history occurs, i.e. over the posterior opponent hands.

Saves the average strategy → strategy_{N}.pkl for the CFR agent.

Usage: LEAVES=leaves_200000.npz ITERS=4000000 python cfr.py
"""
from __future__ import annotations

import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

import game as G                                  # noqa: E402
from common import ACTIONS                        # noqa: E402

HERE = Path(__file__).parent
LEAVES = HERE / os.environ.get("LEAVES", "leaves_200000.npz")
ITERS  = int(os.environ.get("ITERS", 4_000_000))
SEED   = int(os.environ.get("SEED", 0))
OUT    = HERE / os.environ.get("OUT", f"strategy.pkl")

# global tables: infoset key → {action: float}
regret = {}
strat_sum = {}


def regret_matching(I, actions):
    r = regret.get(I)
    if r is None:
        n = len(actions)
        return [1.0 / n] * n
    pos = [r.get(a, 0.0) for a in actions]
    pos = [x if x > 0 else 0.0 for x in pos]
    s = sum(pos)
    if s <= 0:
        n = len(actions)
        return [1.0 / n] * n
    return [x / s for x in pos]


def walk(hist, ctx, i, rng):
    if G.is_terminal(hist):
        return G.payoffs(hist, ctx)[i]
    actions = G.legal_actions(hist, ctx)
    if len(actions) == 1:                       # forced (holder auto-pass)
        return walk(G.apply(hist, actions[0]), ctx, i, rng)

    p = G.to_move(hist)
    I = G.infoset_key(hist, ctx)
    sigma = regret_matching(I, actions)

    if p == i:                                  # traverser: explore all
        util = [0.0] * len(actions)
        node_util = 0.0
        for k, a in enumerate(actions):
            util[k] = walk(G.apply(hist, a), ctx, i, rng)
            node_util += sigma[k] * util[k]
        reg = regret.setdefault(I, {})
        for k, a in enumerate(actions):
            reg[a] = reg.get(a, 0.0) + (util[k] - node_util)
        return node_util

    # opponent: accumulate avg strategy, sample one action
    ss = strat_sum.setdefault(I, {})
    for k, a in enumerate(actions):
        ss[a] = ss.get(a, 0.0) + sigma[k]
    r = rng.random()
    c = 0.0
    for k, a in enumerate(actions):
        c += sigma[k]
        if r <= c:
            return walk(G.apply(hist, a), ctx, i, rng)
    return walk(G.apply(hist, actions[-1]), ctx, i, rng)


def average_strategy():
    avg = {}
    for I, ss in strat_sum.items():
        s = sum(ss.values())
        if s <= 0:
            continue
        avg[I] = {a: v / s for a, v in ss.items()}
    return avg


def _open_report(avg):
    """P0's average opening distribution at the empty history (over buckets)."""
    from collections import defaultdict
    agg = defaultdict(float)
    cnt = 0
    for I, dist in avg.items():
        bucket, av, hist = I
        if hist == ():
            cnt += 1
            for a, p in dist.items():
                agg[a] += p
    if cnt:
        print(f"  P0 open infosets: {cnt}  mean action mix:")
        for a, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"    {a:>10}: {v/cnt:5.2f}")


def main():
    d = np.load(LEAVES, allow_pickle=True)
    EV, AV, BK = d['ev'], d['avail'], d['bucket']
    N = int(d['n_deals'])
    # NaN leaf EV (unavailable action) never indexed at a terminal, but make
    # it harmless if it ever is.
    EV = np.nan_to_num(EV, nan=0.0)
    print(f"=== MCCFR  leaves={LEAVES.name}  N={N}  iters={ITERS} ===",
          flush=True)

    rng = random.Random(SEED)
    t0 = time.perf_counter()
    for it in range(1, ITERS + 1):
        dseed = rng.randrange(N)
        ctx = {'ev': EV[dseed], 'avail': AV[dseed], 'bucket': BK[dseed]}
        walk((), ctx, it % 3, rng)
        if it % 500_000 == 0:
            el = time.perf_counter() - t0
            print(f"  {it/1e6:.1f}M iters  {el:.0f}s  "
                  f"({it/el/1e3:.0f}k/s)  infosets={len(strat_sum)}",
                  flush=True)
    print(f"  done {time.perf_counter()-t0:.0f}s  infosets={len(strat_sum)}")

    avg = average_strategy()
    _open_report(avg)
    with open(OUT, 'wb') as f:
        pickle.dump({'avg': avg, 'actions': ACTIONS,
                     'leaves': LEAVES.name, 'iters': ITERS}, f)
    print(f"  saved → {OUT.name}  ({len(avg)} infosets)")


if __name__ == "__main__":
    main()
