"""Flipped execution: GOD soloist vs PIMC32 defenders. Same silent-ulti /
silent-100 A/B as eval_engine, but now the soloist plays perfectly (no PIMC
noise on its side) and the imperfect players are the defenders. This measures
the PERFECT-PLAY CEILING of the silent-capture lever against realistic defence.

Rows (god soloist on the SAME multi engine; defenders PIMC32-parti; marriages
declared; scored by the corrected oracle):
  A  god multi[parti_pts=1]
  B  god multi[parti + silent_ulti]
  C  god multi[parti + silent_ulti + silent_100]

B − A = silent-ulti capture, C − B = silent-100 capture, at perfect soloist play.

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
# Defender play: 'god' (god-parti, deterministic, ~10x faster — default) or
# 'pimc' (PIMC32, the imperfect realistic defender).
DEF_MODE  = os.environ.get('DEF', 'god')

_BID = oracle_bid()


def _play_godsol(*, deal, weights, pimc_n, seed):
    """God soloist (multi objective = weights); PIMC32-parti defenders."""
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
            chosen = god_pick(pos=pos, contract='multi')   # god soloist
        elif DEF_MODE == 'god':
            chosen = god_pick(pos=pos, contract='parti')   # god defender (fast)
        else:
            chosen, averaged = _pimc.pimc_decision(
                true_pos=pos, contract='parti', n_samples=pimc_n,
                seed=seed * 31337 + move_i, voids=voids_dict,
            )
            if averaged:
                chosen = min(averaged, key=lambda c: averaged[c])   # defender minimises
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
    side = c.get('silent_ulti', 0)
    return {
        'parti_won':   bool(soloist_won_simple(final_pos)),
        'sol_pts':     soloist_points(final_pos),
        'reached_100': soloist_points(final_pos) >= 100,
        'got_silent_ulti': side > 0,     # sol won trick 10 with trump-7
        'got_40_100':  c.get('silent_40_100', 0) != 0,
        'got_20_100':  c.get('silent_20_100', 0) != 0,
        'got_durchmars': c.get('silent_durchmars', 0) != 0,   # sol swept all 10
        'total_gp':    pv.total_per_def,
    }


def worker(args):
    seed, = args
    deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
    has_40, has_20 = sol_marriages(deal.sol_hand, deal.trump)
    if not (has_40 or has_20):
        return None
    cfgs = {
        'A': solver_weights(parti=True),
        'B': solver_weights(parti=True, silent_ulti=True),
        'C': solver_weights(parti=True, silent_ulti=True, silent_100=True,
                            has_40=has_40, has_20=has_20),
        'D': solver_weights(parti=True, silent_ulti=True, silent_100=True,
                            silent_dm=True, has_40=has_40, has_20=has_20),
    }
    out = {k: _summary(_play_godsol(deal=deal, weights=w, pimc_n=PIMC_N, seed=seed))
           for k, w in cfgs.items()}
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
    deftag = "god-parti" if DEF_MODE == 'god' else f"PIMC{PIMC_N}"
    print(f"GOD soloist vs {deftag} defenders | kept {N}/{CAND} marriage deals "
          f"(40:{n40}, 20:{n20})  wall={wall:.0f}s")
    if N == 0:
        return

    for label, name in [('A', 'A god[parti]'), ('B', 'B god+ulti'),
                        ('C', 'C god+ulti+100'), ('D', 'D god+ulti+100+duri')]:
        s = [r[3][label] for r in rows]
        wr = sum(x['parti_won'] for x in s) / N
        su = sum(x['got_silent_ulti'] for x in s) / N
        r100 = sum(x['reached_100'] for x in s) / N
        c40 = sum(x['got_40_100'] for x in s) / N
        c20 = sum(x['got_20_100'] for x in s) / N
        dm = sum(x['got_durchmars'] for x in s) / N
        gm, gs = _mean_std([x['total_gp'] for x in s])
        print(f"\n=== {name} ===")
        print(f"  parti WR {wr:.3f} | silent-ulti {su:.3f} | reached100 {r100:.3f} "
              f"| 40-100 {c40:.3f} | 20-100 {c20:.3f} | DURI {dm:.3f} "
              f"| GP/def {gm:+.3f} ± {gs:.3f}")

    def gp(label):
        return [r[3][label]['total_gp'] for r in rows]
    a, b, c, d = gp('A'), gp('B'), gp('C'), gp('D')

    def delta(x, y):
        dd = [x[i] - y[i] for i in range(N)]
        m, sd = _mean_std(dd)
        t = m / (sd / N ** 0.5) if sd > 0 else 0.0
        return m, t
    bm, bt = delta(b, a); cm, ct = delta(c, b); dm_, dt = delta(d, c)
    print("\n=== effects (god soloist, corrected oracle) ===")
    print(f"  B − A (silent-ulti): {bm:+.3f}  t={bt:+.2f}")
    print(f"  C − B (silent-100):  {cm:+.3f}  t={ct:+.2f}")
    print(f"  D − C (silent-duri): {dm_:+.3f}  t={dt:+.2f}")


if __name__ == "__main__":
    main()
