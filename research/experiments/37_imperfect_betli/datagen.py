"""exp37 — REALISTIC-defense betli make-probability datagen.

Mirrors the god betli head's pipeline EXACTLY (experiments/17_clean_pickup_net/gen_alpha0.py:
deal_betli → 12 cards → ONE random 2-card discard → featurize the 10-card post-discard hand),
swapping ONLY the label: instead of the double-dummy `god_says_soloist_wins`, the label is the
REALISTIC outcome — play the betli out with the deployed engine (PIMC soloist + PIMC defenders) and
record whether the soloist actually MADE it (took 0 tricks). This is exp36 run from the SOLOIST side:
P(make betli | realistic imperfect defense), the quantity the bidder needs to bid plain / bluff betli.

We sweep alpha ~ U(0, ALPHA_MAX) so the realistic make-rate spans the whole decision boundary
(alpha 0→~5% made, 0.5→~32%, 1.0→~55%, 1.5→~90%). The GOD label is recorded alongside (one extra
solve) so training + analysis can compare the two heads on IDENTICAL inputs and count the "dd-lost but
realistically-made" imperfect betlis the god bidder is blind to.

Features: `vnet.pickup.featurize(hand10, None, False)` — the 32-dim colorless encoding, byte-identical
to the deployed `betli` head input, so the trained head is a drop-in for `betli_baseline.pt`.

Env: N (kept records), WORKERS, ALPHA_MAX (1.5), PIMC_N (16), SEED_BASE, OUT.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/Users/milansimity/Cuccok/kodok/oldtawer"
for _p in (_HERE, f"{_REPO}/experiments/31_exploit_play", _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ulti.solvers import pis, determinize as _det                                 # noqa: E402
from ulti.eval.pimc_matchup import pimc_pick, defenders_won, god_says_soloist_wins  # noqa: E402
from ulti.eval.dojo import deal_betli                                             # noqa: E402
from ulti.vnet.pickup import featurize                                           # noqa: E402

N          = int(os.environ.get("N", "150000"))
WORKERS    = int(os.environ.get("WORKERS", "8"))
ALPHA_MAX  = float(os.environ.get("ALPHA_MAX", "1.5"))
PIMC_N     = int(os.environ.get("PIMC_N", "16"))
SEED_BASE  = int(os.environ.get("SEED_BASE", "370000000"))
OUT        = os.environ.get("OUT") or os.path.join(_HERE, "betli_real.npz")


def _play_betli(sol10, d1, d2, talon, seed):
    """Play the post-discard betli out: PIMC soloist vs PIMC defenders. 1 = soloist MADE it."""
    pos = pis.build_position(hands=[list(sol10), list(d1), list(d2)], soloist=0, leader=0,
                             contract="betli", trump=None, talon=list(talon))
    voids = _det.Voids(); vd = voids.as_dict(); rng = random.Random(seed); mi = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        mv = pimc_pick(pos=pos, contract="betli", n_samples=PIMC_N, seed=seed * 131 + mi, voids_dict=vd)
        if mv is None:
            mv = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, mv); vd.clear(); vd.update(voids.as_dict())
        pis.apply_move(pos, mv); mi += 1
    return 0 if defenders_won(pos, "betli") else 1


def worker(seed):
    rng = random.Random(seed ^ 0xA5A5A5A5)
    alpha = rng.uniform(0.0, ALPHA_MAX)
    deal = deal_betli(seed=seed, alpha=alpha)
    sol12 = list(deal.sol_hand) + list(deal.talon)
    idx = set(rng.sample(range(12), 2))                      # ONE random discard (god-head parity)
    sol10 = [sol12[i] for i in range(12) if i not in idx]
    disc = [sol12[i] for i in idx]
    d1, d2 = list(deal.def1_hand), list(deal.def2_hand)

    pos0 = pis.build_position(hands=[sol10, d1, d2], soloist=0, leader=0,
                              contract="betli", trump=None, talon=disc)
    y_god = 1 if god_says_soloist_wins(pos0, contract="betli") else 0
    y_real = _play_betli(sol10, d1, d2, disc, seed)
    x = featurize(sol10, None, False).astype(np.float32)
    return x, y_real, y_god, float(alpha)


def main():
    print(f"=== exp37 realistic-betli datagen ===", flush=True)
    print(f"  N={N}  ALPHA_MAX={ALPHA_MAX}  PIMC_N={PIMC_N}  workers={WORKERS}  out={os.path.basename(OUT)}",
          flush=True)
    seeds = (SEED_BASE + i for i in range(N))
    Xs, yr, yg, al = [], [], [], []
    t0 = time.perf_counter(); report_every = max(2000, N // 50)
    ctx = get_context("fork")
    with ctx.Pool(WORKERS) as pool:
        for r in pool.imap_unordered(worker, seeds, chunksize=64):
            Xs.append(r[0]); yr.append(r[1]); yg.append(r[2]); al.append(r[3])
            k = len(yr)
            if k % report_every == 0:
                wall = time.perf_counter() - t0; rate = k / wall
                mr = sum(yr) / k; gr = 1.0 - sum(yg) / k
                imperfect = sum(1 for i in range(k) if yr[i] == 1 and yg[i] == 0) / k
                print(f"  {k}/{N}  wall={wall:.0f}s eta={(N-k)/rate:.0f}s  "
                      f"real-made={mr:.3f}  dd-lost={gr:.3f}  dd-lost&made={imperfect:.3f}", flush=True)
    X = np.stack(Xs)
    y_real = np.array(yr, dtype=np.float32)
    y_god = np.array(yg, dtype=np.float32)
    alpha = np.array(al, dtype=np.float32)
    np.savez_compressed(OUT, X=X, y=y_real, y_god=y_god, alpha=alpha)
    wall = time.perf_counter() - t0
    imperfect = float(((y_real == 1) & (y_god == 0)).mean())
    print(f"\n  saved {X.shape[0]} records  wall={wall:.0f}s", flush=True)
    print(f"  realistic make-rate {y_real.mean():.4f}   dd-lost {1-y_god.mean():.4f}   "
          f"dd-lost & realistically-made {imperfect:.4f}  (= the imperfect-betli headroom)", flush=True)


if __name__ == "__main__":
    main()
