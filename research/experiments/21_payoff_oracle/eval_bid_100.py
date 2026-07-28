"""Validate the BID 40-100 / 20-100 recipe: does pricing the declared 100 make
the soloist actually MAKE the bid (vs ignoring it and going bukott)?

The soloist has bid a 40-100 (or 20-100) — committed, scored +bid if it reaches
100 else -bid (bukott). We compare its PLAY objective:
  A  god-multi[parti_pts=1]            ignores the bid (just wins parti)
  B  god-multi[parti + bid score_geq_100]   plays to reach 100

Defenders god-parti (they resist the 100 for free — taking points helps their
parti). Both scored with the SAME BidSet (the bid is declared either way), so the
GP delta is purely the value of playing toward what you bid. god-vs-god →
deterministic.

Deals: ulti-biased (strong, parti-feasible), filtered to a SINGLE declared
marriage matching the bid (so the solver's total>=100 threshold is exact).

Env: BID ('40_100'|'20_100'), CAND, WORKERS, SEED_BASE, ALPHA.
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from eval.dojo import deal_ulti_biased
from eval.pimc_matchup import god_pick
from solvers import pis as pis_bridge
from trickster._solver_core import set_multi_weights
from trickster.games.ulti.game import soloist_won_simple, soloist_points

from scoring.oracle import score as score_oracle
from recipe import sol_marriages
from engine import solver_weights, oracle_bid

BID       = os.environ.get('BID', '40_100')
CAND      = int(os.environ.get('CAND', '4000'))
WORKERS   = int(os.environ.get('WORKERS', '8'))
SEED_BASE = int(os.environ.get('SEED_BASE', '330000000'))
ALPHA     = float(os.environ.get('ALPHA', '0.6'))

_BID = oracle_bid(bid=BID)
_KEY = BID                                   # oracle component key: '40_100'/'20_100'


def _wanted(has_40, has_20):
    """Single declared marriage matching the bid (exact threshold)."""
    if BID == '40_100':
        return has_40 and not has_20
    return has_20 and not has_40


def _play(*, deal, weights):
    set_multi_weights(**weights)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract='parti',
        trump=deal.trump, talon=list(deal.talon), declare_marriages=True,
    )
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        contract = 'multi' if p == 0 else 'parti'   # god-multi sol, god-parti def
        chosen = god_pick(pos=pos, contract=contract)
        pis_bridge.apply_move(pos, chosen)
    return pos


def _summary(final_pos):
    pv = score_oracle(final_pos=final_pos, bid=_BID)
    return {
        'made':      pv.components.get(_KEY, 0) > 0,    # reached 100
        'parti_won': bool(soloist_won_simple(final_pos)),
        'sol_pts':   soloist_points(final_pos),
        'total_gp':  pv.total_per_def,
    }


def worker(args):
    seed, = args
    deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
    has_40, has_20 = sol_marriages(deal.sol_hand, deal.trump)
    if not _wanted(has_40, has_20):
        return None
    a = _summary(_play(deal=deal, weights=solver_weights(parti=True)))
    b = _summary(_play(deal=deal, weights=solver_weights(parti=True, bid=BID)))
    return (seed, a, b)


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def main():
    print(f"[bid {BID}] CAND={CAND}  god-multi sol (A parti-only vs B bid) "
          f"vs god-parti def", flush=True)
    jobs = [(SEED_BASE + i,) for i in range(CAND)]
    t0 = time.perf_counter()
    rows = []
    scanned = 0
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(worker, jobs, chunksize=4):
            scanned += 1
            if r is not None:
                rows.append(r)
            if scanned % 500 == 0:
                print(f"  scanned {scanned}/{CAND}  kept {len(rows)}  "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    N = len(rows)
    print()
    print(f"bid {BID} | kept {N}/{CAND} single-marriage deals  wall={wall:.0f}s")
    if N == 0:
        return

    for label, idx in [("A play parti-only", 1), ("B play the bid", 2)]:
        s = [r[idx] for r in rows]
        made = sum(x['made'] for x in s) / N
        wr = sum(x['parti_won'] for x in s) / N
        pm, _ = _mean_std([x['sol_pts'] for x in s])
        gm, gs = _mean_std([x['total_gp'] for x in s])
        print(f"\n=== {label} ===")
        print(f"  bid MADE {made:.3f} (bukott {1-made:.3f}) | parti WR {wr:.3f} "
              f"| sol pts {pm:.1f} | GP/def {gm:+.3f} ± {gs:.3f}")

    a = [r[1]['total_gp'] for r in rows]
    b = [r[2]['total_gp'] for r in rows]
    d = [b[i] - a[i] for i in range(N)]
    dm, ds = _mean_std(d)
    t = dm / (ds / N ** 0.5) if ds > 0 else 0.0
    made_a = sum(r[1]['made'] for r in rows) / N
    made_b = sum(r[2]['made'] for r in rows) / N
    print(f"\n=== B − A (value of playing toward the {BID} you bid) ===")
    print(f"  made rate {made_a:.3f} → {made_b:.3f}  (+{made_b-made_a:.3f})")
    print(f"  GP/def {dm:+.3f}  t={t:+.2f}")


if __name__ == "__main__":
    main()
