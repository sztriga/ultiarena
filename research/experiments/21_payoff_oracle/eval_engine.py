"""One-engine eval: contracts are weight configs on the SAME multi solver,
scored by the SAME (corrected) oracle.

Rows (PIMC32 soloist vs god-parti defenders, marriages declared):
  A0  contract='parti'           dedicated parti solver (reference)
  A1  multi[parti_pts=1]         must reproduce A0 exactly  (invariant)
  B   multi[parti + silent_ulti]
  C   multi[parti + silent_ulti + silent_100]   (corrected 40/20 pricing)

Reports: A1−A0 per-deal GP (the configurability invariant, should be ~0) and
C−B (the silent-100 effect under the corrected oracle).

Env: CAND, PIMC_N, WORKERS, SEED_BASE, ALPHA.
"""
from __future__ import annotations

import os, random, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from eval.dojo import deal_ulti_biased
from eval.pimc_matchup import god_pick
from solvers import determinize as _det
from solvers import pimc as _pimc
from solvers import pis as pis_bridge
from trickster._solver_core import set_multi_weights
from trickster.games.ulti.game import soloist_won_simple, soloist_points

from scoring.oracle import score as score_oracle
from recipe import sol_marriages
from engine import solver_weights, oracle_bid

CAND      = int(os.environ.get('CAND', '120'))
PIMC_N    = int(os.environ.get('PIMC_N', '32'))
WORKERS   = int(os.environ.get('WORKERS', '6'))
SEED_BASE = int(os.environ.get('SEED_BASE', '210000000'))
ALPHA     = float(os.environ.get('ALPHA', '0.6'))

_BID = oracle_bid()


def _pimc_pick(*, pos, contract, n_samples, seed, voids_dict):
    chosen, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, voids=voids_dict,
    )
    viewer = pis_bridge.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


def _play(*, deal, contract, weights, pimc_n, seed):
    if weights is not None:
        set_multi_weights(**weights)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract='parti',
        trump=deal.trump, talon=list(deal.talon), declare_marriages=True,
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)
    move_i = 0
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        if p == 0:
            chosen = _pimc_pick(pos=pos, contract=contract, n_samples=pimc_n,
                                seed=seed * 31337 + move_i, voids_dict=voids_dict)
        else:
            chosen = god_pick(pos=pos, contract='parti')
        if chosen is None:
            chosen = rng.choice(pis_bridge.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis_bridge.apply_move(pos, chosen)
        move_i += 1
    return pos


def _summary(final_pos):
    pv = score_oracle(final_pos=final_pos, bid=_BID)
    c = pv.components
    return {
        'parti_won':   bool(soloist_won_simple(final_pos)),
        'sol_pts':     soloist_points(final_pos),
        'reached_100': soloist_points(final_pos) >= 100,
        'got_40_100':  c.get('silent_40_100', 0) != 0,
        'got_20_100':  c.get('silent_20_100', 0) != 0,
        'total_gp':    pv.total_per_def,
    }


def worker(args):
    seed, = args
    deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
    has_40, has_20 = sol_marriages(deal.sol_hand, deal.trump)
    if not (has_40 or has_20):
        return None

    cfgs = {
        'A0': ('parti', None),
        'A1': ('multi', solver_weights(parti=True)),
        'B':  ('multi', solver_weights(parti=True, silent_ulti=True)),
        'C':  ('multi', solver_weights(parti=True, silent_ulti=True,
                                       silent_100=True, has_40=has_40, has_20=has_20)),
    }
    out = {}
    for k, (contract, w) in cfgs.items():
        out[k] = _summary(_play(deal=deal, contract=contract, weights=w,
                                pimc_n=PIMC_N, seed=seed))
    return (seed, has_40, has_20, out)


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, v ** 0.5


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
            if done % 20 == 0:
                print(f"  scanned {done}/{CAND}  kept {len(rows)}  "
                      f"wall={time.perf_counter()-t0:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    N = len(rows)
    n40 = sum(1 for r in rows if r[1]); n20 = sum(1 for r in rows if r[2])
    print()
    print(f"kept {N}/{CAND} marriage deals (40:{n40}, 20:{n20})  "
          f"wall={wall:.0f}s  pimc_n={PIMC_N}")
    if N == 0:
        return

    for label in ['A0', 'A1', 'B', 'C']:
        s = [r[3][label] for r in rows]
        wr = sum(x['parti_won'] for x in s) / N
        pm, _ = _mean_std([x['sol_pts'] for x in s])
        r100 = sum(x['reached_100'] for x in s) / N
        c40 = sum(x['got_40_100'] for x in s) / N
        c20 = sum(x['got_20_100'] for x in s) / N
        gm, gs = _mean_std([x['total_gp'] for x in s])
        names = {'A0': 'A0 parti (dedicated)', 'A1': 'A1 multi[parti]',
                 'B': 'B multi+ulti', 'C': 'C multi+ulti+100'}
        print(f"\n=== {names[label]} ===")
        print(f"  parti WR {wr:.3f} | pts {pm:.1f} | reached100 {r100:.3f} | "
              f"40-100 {c40:.3f} | 20-100 {c20:.3f} | GP/def {gm:+.3f} ± {gs:.3f}")

    def gp(label):
        return [r[3][label]['total_gp'] for r in rows]
    a0, a1, b, c = gp('A0'), gp('A1'), gp('B'), gp('C')

    def delta(x, y):
        d = [x[i] - y[i] for i in range(N)]
        m, sd = _mean_std(d)
        nmis = sum(1 for v in d if v != 0)
        t = m / (sd / N ** 0.5) if sd > 0 else 0.0
        return m, t, nmis
    dm, dt, nmis = delta(a1, a0)
    print("\n=== INVARIANT: A1 multi[parti] vs A0 dedicated parti ===")
    print(f"  per-deal GP diff: mean {dm:+.4f}  deals differing: {nmis}/{N}"
          + ("   ✓ IDENTICAL" if nmis == 0 else "   ⚠ DIVERGES"))
    cb_m, cb_t, _ = delta(c, b)
    ba_m, ba_t, _ = delta(b, a1)
    print("\n=== effects (corrected oracle) ===")
    print(f"  C − B (silent-100):  {cb_m:+.3f}  t={cb_t:+.2f}")
    print(f"  B − A1 (silent-ulti): {ba_m:+.3f}  t={ba_t:+.2f}")


if __name__ == "__main__":
    main()
