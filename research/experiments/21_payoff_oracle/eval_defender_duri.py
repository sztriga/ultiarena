"""Defender side of durchmars: will silent-duri-aware defenders ACTIVELY break a
sweep instead of letting it slide?

Durchmars-colored deals (dominant soloist → sweeps are on the table). Fixed
god-parti soloist (separate EV_PARTI evaluator, so it doesn't share the multi
global with the defenders). Defenders flip the duri objective:

  A  defenders god[parti]                 (silent_durchmars OFF — let it slide)
  B  defenders god[parti + silent_durchmars] (ON — minimise sol's value, so a
     sweep (+3 from the soloist's view) is something to DENY → grab one trick)

God-vs-god → deterministic, A vs B is purely the defender objective. Key output:
deals where A lets the sweep happen but B BREAKS it (milan's "at least one game"),
with seeds to inspect.

Env: CAND, WORKERS, SEED_BASE, ALPHA.
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from ulti.eval.dojo import deal_durchmars_colored
from ulti.eval.pimc_matchup import god_pick
from ulti.solvers import pis as pis_bridge
from trickster._solver_core import set_multi_weights
from trickster.games.ulti.game import (soloist_won_simple, soloist_points,
                                       soloist_won_durchmars)

from ulti.scoring.oracle import score as score_oracle, BidSet

CAND      = int(os.environ.get('CAND', '1500'))
WORKERS   = int(os.environ.get('WORKERS', '8'))
SEED_BASE = int(os.environ.get('SEED_BASE', '320000000'))
ALPHA     = float(os.environ.get('ALPHA', '0.6'))

_BID = BidSet(parti=True)


def _play(*, deal, def_contract, def_weights):
    """God-parti soloist; defenders play `def_contract` (parti or multi)."""
    if def_weights is not None:
        set_multi_weights(**def_weights)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract='parti',
        trump=deal.trump, talon=list(deal.talon), declare_marriages=True,
    )
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        contract = 'parti' if p == 0 else def_contract
        chosen = god_pick(pos=pos, contract=contract)
        pis_bridge.apply_move(pos, chosen)
    return pos


def _summary(final_pos):
    pv = score_oracle(final_pos=final_pos, bid=_BID)
    return {
        'sol_swept':   bool(soloist_won_durchmars(final_pos)),
        'sol_parti':   bool(soloist_won_simple(final_pos)),
        'total_gp':    pv.total_per_def,
    }


def worker(args):
    seed, = args
    deal = deal_durchmars_colored(seed=seed, alpha=ALPHA)
    a = _summary(_play(deal=deal, def_contract='parti', def_weights=None))
    b = _summary(_play(deal=deal, def_contract='multi',
                       def_weights=dict(parti_pts=1.0, silent_durchmars=3.0)))
    return (seed, a, b)


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def main():
    print(f"[def-duri] CAND={CAND}  god-parti sol vs god defenders (duri off/on)",
          flush=True)
    jobs = [(SEED_BASE + i,) for i in range(CAND)]
    t0 = time.perf_counter()
    rows = []
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(worker, jobs, chunksize=4):
            rows.append(r)
            if len(rows) % 200 == 0:
                broke = sum(1 for _, a, b in rows
                            if a['sol_swept'] and not b['sol_swept'])
                print(f"  {len(rows)}/{CAND} | A-sweeps "
                      f"{sum(a['sol_swept'] for _,a,_ in rows)} | "
                      f"B-breaks-it {broke} | {time.perf_counter()-t0:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0
    N = len(rows)

    a_sweep = sum(1 for _, a, _ in rows if a['sol_swept'])
    b_sweep = sum(1 for _, _, b in rows if b['sol_swept'])
    broke = [(s, a, b) for s, a, b in rows if a['sol_swept'] and not b['sol_swept']]
    forced = sum(1 for _, a, b in rows if a['sol_swept'] and b['sol_swept'])
    am, _ = _mean_std([a['total_gp'] for _, a, _ in rows])
    bm, _ = _mean_std([b['total_gp'] for _, _, b in rows])
    d = [b['total_gp'] - a['total_gp'] for _, a, b in rows]
    dm, ds = _mean_std(d)
    t = dm / (ds / N ** 0.5) if ds > 0 else 0.0

    print()
    print(f"N={N} durchmars-colored deals  wall={wall:.0f}s")
    print(f"  sweeps with A (parti defenders, let-slide):   {a_sweep} "
          f"({100*a_sweep/N:.1f}%)")
    print(f"  sweeps with B (duri-aware defenders):          {b_sweep} "
          f"({100*b_sweep/N:.1f}%)")
    print(f"  forced sweeps (happen under BOTH):             {forced}")
    print(f"  *** A let it slide but B BROKE the sweep:      {len(broke)} ***")
    print(f"  sol GP/def:  A {am:+.3f}  vs  B {bm:+.3f}   "
          f"(B−A {dm:+.3f}, t={t:+.2f}; negative = defenders gained)")
    if broke:
        print(f"  example seeds where B actively stopped the duri: "
              f"{[s for s, _, _ in broke[:12]]}")
        print("  → at least one game found: silent-duri defenders break a sweep "
              "that parti-only defenders allow.")
    else:
        print("  → no game found where B breaks a sweep A allows (all sweeps "
              "were forced / unbreakable, or none occurred).")


if __name__ == "__main__":
    main()
