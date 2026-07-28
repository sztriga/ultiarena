"""Defender side of silent ulti. Deals where a DEFENDER holds the trump-7,
fixed god-parti soloist, defenders flip objective:

  A  defenders god[parti]            (indifferent to silent ulti)
  B  defenders god[parti+silent_ulti] (aware: go for the 7, or dump it early)

God-vs-god → deterministic, so A vs B is PURELY the defender objective. The
silent_ulti component is scored from the soloist's view, so a DEFENDER win = −2
to sol, a defender bukott (plays the 7 in trick 10 and loses) = +4 to sol.
Hypothesis: aware defenders capture more silent ulti (−2) and avoid bukott (+4),
so the soloist's GP drops (B < A).

Env: CAND, WORKERS, SEED_BASE, ALPHA.
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from ulti.eval.dojo import deal_parti
from ulti.eval.pimc_matchup import god_pick
from ulti.solvers import pis as pis_bridge
from trickster._solver_core import set_multi_weights, set_multi_cull
from trickster.games.ulti.game import soloist_won_simple, soloist_points

from ulti.scoring.oracle import score as score_oracle, BidSet

CAND      = int(os.environ.get('CAND', '300'))
WORKERS   = int(os.environ.get('WORKERS', '8'))
SEED_BASE = int(os.environ.get('SEED_BASE', '310000000'))
ALPHA     = float(os.environ.get('ALPHA', '0.6'))

_BID = BidSet(parti=True)   # piros=False → silent_ulti component is exactly -2/+4


def _def_holds_7(deal):
    return any(c.suit == deal.trump and c.rank == '7'
               for c in list(deal.def1_hand) + list(deal.def2_hand))


CULL = int(os.environ.get('CULL', '1'))   # 0 = airtight check: disable EV_MULTI cull


def _play(*, deal, def_contract, def_weights):
    """God-parti soloist; defenders play `def_contract` (parti or multi)."""
    set_multi_cull(CULL)
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
    su = pv.components.get('silent_ulti', 0)
    return {
        'sol_parti_won': bool(soloist_won_simple(final_pos)),
        'def_silent_ulti': su < 0,    # a defender won trick 10 with the 7
        'def_bukott':      su > 0,    # a defender played the 7 in t10 and lost
        'seven_in_t10':    su != 0,   # the 7 reached the last trick at all
        'total_gp':        pv.total_per_def,
    }


def worker(args):
    seed, = args
    deal = deal_parti(seed=seed, alpha=ALPHA)
    if not _def_holds_7(deal):
        return None
    a = _summary(_play(deal=deal, def_contract='parti', def_weights=None))
    b = _summary(_play(deal=deal, def_contract='multi',
                       def_weights=dict(parti_pts=1.0, silent_ulti=2.0)))
    return (seed, a, b)


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def main():
    jobs = [(SEED_BASE + i,) for i in range(CAND)]
    t0 = time.perf_counter()
    rows = []
    done = 0
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(worker, jobs, chunksize=1):
            done += 1
            if r is not None:
                rows.append(r)
            if done % 40 == 0:
                print(f"  scanned {done}/{CAND}  kept {len(rows)}  "
                      f"wall={time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    N = len(rows)
    print()
    print(f"god-parti sol vs god defenders | kept {N}/{CAND} deals "
          f"(a defender holds the trump-7)  wall={wall:.0f}s")
    if N == 0:
        return

    for label, idx in [("A defenders god[parti]", 1),
                        ("B defenders god[parti+silent_ulti]", 2)]:
        s = [r[idx] for r in rows]
        wr  = sum(x['sol_parti_won'] for x in s) / N
        su  = sum(x['def_silent_ulti'] for x in s) / N
        bk  = sum(x['def_bukott'] for x in s) / N
        t10 = sum(x['seven_in_t10'] for x in s) / N
        gm, gs = _mean_std([x['total_gp'] for x in s])
        print(f"\n=== {label} ===")
        print(f"  sol parti WR {wr:.3f} | DEF silent-ulti {su:.3f} | "
              f"DEF bukott {bk:.3f} | 7 in trick10 {t10:.3f} | "
              f"sol GP/def {gm:+.3f} ± {gs:.3f}")

    a = [r[1]['total_gp'] for r in rows]
    b = [r[2]['total_gp'] for r in rows]
    d = [b[i] - a[i] for i in range(N)]
    dm, ds = _mean_std(d)
    t = dm / (ds / N ** 0.5) if ds > 0 else 0.0
    print("\n=== B − A (defender silent-ulti awareness, sol GP/def) ===")
    print(f"  {dm:+.3f}  t={t:+.2f}   (negative = defenders gained)")


if __name__ == "__main__":
    main()
