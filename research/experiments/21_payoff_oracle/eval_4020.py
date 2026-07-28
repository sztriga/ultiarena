"""40/20 ablation: does pricing the silent 40-100 / 20-100 recover GP?

Marriage-holding, parti-feasible deals (ulti-biased dealer → fat trump suit, so
the soloist often holds the trump king+upper = a 40, and parti is winnable).
Defenders play god-parti; the soloist plays one of three engines; every
played-out deal is scored by the SAME contract oracle (which already credits
silent 40-100 / 20-100). Only the soloist's *policy* differs, so the GP delta is
the value of chasing 100 when holding a marriage.

  A  vanilla parti PIMC                                   (deployed baseline)
  B  multi[parti_pts=1, silent_ulti=2]                    (current multi)
  C  multi[B + score_geq_100 gated by held marriage]      (the new 40/20 pricing)

C - B isolates the 40/20 contribution; B - A is the silent-ulti sanity check.

Env: CAND (candidate seeds), PIMC_N, WORKERS, SEED_BASE, ALPHA.
"""
from __future__ import annotations

import os, random, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from ulti.eval.dojo import deal_ulti_biased
from ulti.eval.pimc_matchup import god_pick
from ulti.solvers import determinize as _det
from ulti.solvers import pimc as _pimc
from ulti.solvers import pis as pis_bridge
from trickster._solver_core import set_multi_weights
from trickster.games.ulti.game import soloist_won_simple, soloist_points

from ulti.scoring.oracle import BidSet, score as score_oracle
from recipe import sol_marriages, parti_multi_weights

CAND      = int(os.environ.get('CAND', '120'))
PIMC_N    = int(os.environ.get('PIMC_N', '32'))
WORKERS   = int(os.environ.get('WORKERS', '6'))
SEED_BASE = int(os.environ.get('SEED_BASE', '210000000'))
ALPHA     = float(os.environ.get('ALPHA', '0.6'))

_BID = BidSet(parti=True)


def _pimc_pick(*, pos, contract, n_samples, seed, voids_dict):
    chosen, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, voids=voids_dict,
    )
    viewer = pis_bridge.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


def _play_one(*, deal, sol_contract, pimc_n, seed):
    """Sol plays ``sol_contract`` via PIMC; defenders play god-parti.
    Marriages declared so the parti channel carries the 40/20 points."""
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
            chosen = _pimc_pick(
                pos=pos, contract=sol_contract, n_samples=pimc_n,
                seed=seed * 31337 + move_i, voids_dict=voids_dict,
            )
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
    comp = pv.components
    return {
        'parti_won':   bool(soloist_won_simple(final_pos)),
        'sol_pts':     soloist_points(final_pos),
        'reached_100': soloist_points(final_pos) >= 100,
        'got_40_100':  comp.get('silent_40_100', 0) != 0,
        'got_20_100':  comp.get('silent_20_100', 0) != 0,
        'got_40':      comp.get('silent_40', 0) != 0,
        'got_20':      comp.get('silent_20', 0) != 0,
        'total_gp':    pv.total_per_def,
    }


def worker(args):
    seed, = args
    deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
    has_40, has_20 = sol_marriages(deal.sol_hand, deal.trump)
    if not (has_40 or has_20):
        return None                       # off-target deal, skip

    # A — vanilla parti
    final_a = _play_one(deal=deal, sol_contract='parti', pimc_n=PIMC_N, seed=seed)
    # B — multi parti + silent ulti
    set_multi_weights(parti_pts=1.0, silent_ulti=2.0)
    final_b = _play_one(deal=deal, sol_contract='multi', pimc_n=PIMC_N, seed=seed)
    # C — B + gated score_geq_100
    set_multi_weights(**parti_multi_weights(has_40, has_20))
    final_c = _play_one(deal=deal, sol_contract='multi', pimc_n=PIMC_N, seed=seed)

    return (seed, has_40, has_20,
            _summary(final_a), _summary(final_b), _summary(final_c))


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
            if done % 10 == 0:
                wall = time.perf_counter() - t0
                print(f"  scanned {done}/{CAND}  kept {len(rows)}  "
                      f"wall={wall:.0f}s", flush=True)
    wall = time.perf_counter() - t0
    N = len(rows)
    n40 = sum(1 for r in rows if r[1])
    n20 = sum(1 for r in rows if r[2])
    print()
    print(f"kept {N}/{CAND} marriage deals (40: {n40}, 20: {n20})  "
          f"wall={wall:.0f}s  workers={WORKERS}  pimc_n={PIMC_N}")
    if N == 0:
        return

    for label, idx in [("A vanilla parti", 3), ("B multi+ulti", 4),
                        ("C multi+40/20", 5)]:
        s = [r[idx] for r in rows]
        wr   = sum(x['parti_won'] for x in s) / N
        pm, ps = _mean_std([x['sol_pts'] for x in s])
        r100 = sum(x['reached_100'] for x in s) / N
        c40  = sum(x['got_40_100'] for x in s) / N
        c20  = sum(x['got_20_100'] for x in s) / N
        gm, gs = _mean_std([x['total_gp'] for x in s])
        print()
        print(f"=== {label} ===")
        print(f"  parti win rate:        {wr:.3f}")
        print(f"  sol pts:               {pm:.2f} ± {ps:.2f}")
        print(f"  reached 100:           {r100:.3f}")
        print(f"  silent 40-100 hit:     {c40:.3f}")
        print(f"  silent 20-100 hit:     {c20:.3f}")
        print(f"  total GP / def:        {gm:+.3f} ± {gs:.3f}  (×2 sol = {2*gm:+.3f})")

    # paired deltas
    def gp(idx):
        return [r[idx]['total_gp'] for r in rows]
    a, b, c = gp(3), gp(4), gp(5)
    dCB, sCB = _mean_std([c[i] - b[i] for i in range(N)])
    dBA, sBA = _mean_std([b[i] - a[i] for i in range(N)])
    dCA, sCA = _mean_std([c[i] - a[i] for i in range(N)])
    def t(m, sd):
        return m / (sd / N ** 0.5) if sd > 0 else 0.0
    print()
    print("=== paired GP/def deltas (sol perspective) ===")
    print(f"  C - B (40/20 effect):  {dCB:+.3f}  t={t(dCB,sCB):+.2f}")
    print(f"  B - A (silent-ulti):   {dBA:+.3f}  t={t(dBA,sBA):+.2f}")
    print(f"  C - A (total):         {dCA:+.3f}  t={t(dCA,sCA):+.2f}")


if __name__ == "__main__":
    main()
