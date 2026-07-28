"""parti × multi[parti+ulti] full 2×2 matrix. No god solver.

Both sides PIMC. Four cells:
    sol \ def  | def=parti | def=multi
    -----------+-----------+----------
    sol=parti  | A         | B
    sol=multi  | C         | D

Same 50 ulti-biased deals (seeds, alpha=0.6, PIMC32) used across all
four cells. With ulti-biased dealer the trump-7 is always in sol's
hand, so only two of the four silent-ulti outcomes can occur:

    scenario     who played 7   who won trick    sol GP / def
    sol won      sol            sol              +2
    sol bukott   sol            def              -4
    (def won, def bukott: structurally impossible here — only sol can
    play the trump-7. To populate those, use parti-biased deals.)

Per cell, reports parti win rate, sol pts, both achievable ulti
outcomes, and total GP/def.
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

from ulti.eval.dojo import deal_ulti_biased
from ulti.solvers import determinize as _det
from ulti.solvers import pimc as _pimc
from ulti.solvers import pis as pis_bridge
from trickster._solver_core import set_multi_weights

sys.path.insert(0, str(Path(__file__).parent))
from ulti.scoring.oracle import BidSet, score as score_oracle

N            = 50
PIMC_N       = 32
ALPHA        = 0.6
SEED_BASE    = 90_000_000
N_WORKERS    = 4
WEIGHT_PARTI = 1.0
WEIGHT_ULTI  = 2.0

CELLS = [
    ("sol=parti, def=parti", "parti", "parti"),
    ("sol=parti, def=multi", "parti", "multi"),
    ("sol=multi, def=parti", "multi", "parti"),
    ("sol=multi, def=multi", "multi", "multi"),
]


def _init_worker():
    set_multi_weights(parti_pts=WEIGHT_PARTI, silent_ulti=WEIGHT_ULTI)


def _pimc_pick(*, pos, contract, n_samples, seed, voids_dict):
    chosen, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, voids=voids_dict,
    )
    viewer = pis_bridge.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


def _play_one(*, deal, sol_contract, def_contract, pimc_n, seed):
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract='parti',
        trump=deal.trump, talon=list(deal.talon),
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)
    move_i = 0
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        contract = sol_contract if p == 0 else def_contract
        chosen = _pimc_pick(
            pos=pos, contract=contract, n_samples=pimc_n,
            seed=seed * 31337 + move_i, voids_dict=voids_dict,
        )
        if chosen is None:
            chosen = rng.choice(pis_bridge.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis_bridge.apply_move(pos, chosen)
        move_i += 1
    return pos


_BID = BidSet(parti=True)


def _summary(final_pos):
    pv = score_oracle(final_pos=final_pos, bid=_BID)
    from trickster.games.ulti.game import (
        soloist_won_simple, soloist_points, last_trick_ulti_check,
    )
    side, won = last_trick_ulti_check(final_pos)
    return {
        "parti_won":   soloist_won_simple(final_pos),
        "sol_pts":     soloist_points(final_pos),
        "ulti_side":   side,
        "ulti_won":    won,
        "total_gp_per_def":  pv.total_per_def,
    }


def worker(args):
    seed, sol_c, def_c = args
    deal = deal_ulti_biased(seed=seed, alpha=ALPHA)
    final = _play_one(deal=deal, sol_contract=sol_c, def_contract=def_c,
                      pimc_n=PIMC_N, seed=seed)
    return (seed, sol_c, def_c, _summary(final))


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, v ** 0.5


def _print_cell(label, sums):
    n = len(sums)
    parti_wr = sum(s["parti_won"] for s in sums) / n
    pts_m, pts_s = _mean_std([s["sol_pts"] for s in sums])
    sol_ulti = sum(1 for s in sums if s["ulti_side"] == "soloist" and s["ulti_won"])  / n
    sol_buk  = sum(1 for s in sums if s["ulti_side"] == "soloist" and not s["ulti_won"]) / n
    def_ulti = sum(1 for s in sums if s["ulti_side"] == "defender" and s["ulti_won"])  / n
    def_buk  = sum(1 for s in sums if s["ulti_side"] == "defender" and not s["ulti_won"]) / n
    gp_m, gp_s = _mean_std([s["total_gp_per_def"] for s in sums])
    print()
    print(f"=== {label} ===")
    print(f"  parti win rate:           {parti_wr:.3f}")
    print(f"  sol parti pts:            {pts_m:.2f} ± {pts_s:.2f}")
    print(f"  silent ulti sol won:      {sol_ulti:.3f}")
    print(f"  silent ulti sol bukott:   {sol_buk:.3f}")
    print(f"  silent ulti def won:      {def_ulti:.3f}")
    print(f"  silent ulti def bukott:   {def_buk:.3f}")
    print(f"  total GP / def:           {gp_m:+.3f} ± {gp_s:.3f}")
    print(f"  total GP (×2 for sol):    {2*gp_m:+.3f}")


def main():
    # All jobs across all 4 cells, batched together for max parallelism.
    jobs = [(SEED_BASE + i, sol_c, def_c)
            for (_, sol_c, def_c) in CELLS
            for i in range(N)]
    print(f"Total jobs: {len(jobs)} ({len(CELLS)} cells × N={N})", flush=True)

    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(worker, jobs, chunksize=1):
            rows.append(r)
            if len(rows) % 10 == 0:
                wall = time.perf_counter() - t0
                rate = len(rows) / wall
                eta = (len(jobs) - len(rows)) / rate if rate else 0
                print(f"  {len(rows)}/{len(jobs)}  wall={wall:.0f}s  eta={eta:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0

    # Group by (sol_c, def_c)
    by_cell = {}
    for seed, sol_c, def_c, summary in rows:
        by_cell.setdefault((sol_c, def_c), []).append(summary)

    for label, sol_c, def_c in CELLS:
        _print_cell(label, by_cell[(sol_c, def_c)])

    # Compact matrix
    print()
    print("=== Compact matrix — silent ulti sol won % ===")
    print(f"{'':>12} {'def=parti':>12} {'def=multi':>12}")
    for sol_c in ("parti", "multi"):
        row = []
        for def_c in ("parti", "multi"):
            sums = by_cell[(sol_c, def_c)]
            v = sum(1 for s in sums if s["ulti_side"] == "soloist" and s["ulti_won"]) / len(sums)
            row.append(f"{v:>12.3f}")
        print(f"sol={sol_c:>7} " + " ".join(row))

    print()
    print("=== Compact matrix — total GP / def (sol's perspective) ===")
    print(f"{'':>12} {'def=parti':>12} {'def=multi':>12}")
    for sol_c in ("parti", "multi"):
        row = []
        for def_c in ("parti", "multi"):
            sums = by_cell[(sol_c, def_c)]
            v = sum(s["total_gp_per_def"] for s in sums) / len(sums)
            row.append(f"{v:>+12.3f}")
        print(f"sol={sol_c:>7} " + " ".join(row))

    print()
    print(f"Wall: {wall:.0f}s, {N_WORKERS} workers, N={N}, pimc_n={PIMC_N}, alpha={ALPHA}")


if __name__ == "__main__":
    main()
