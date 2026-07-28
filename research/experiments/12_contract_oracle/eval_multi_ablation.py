"""parti vs multi[parti+ulti] — silent ulti capture ablation.

Same ulti-biased deals. God plays defenders in both games so the only
variable is sol's strategy.

  Engine A (baseline): sol = vanilla parti PIMC
  Engine B (multi):    sol = multi PIMC with parti_pts=1.0, silent_ulti=2.0

Each played-out deal is scored with the contract oracle. Reports:
  - silent ulti hit rate (sol won 7 in last trick, scored)
  - bukott rate (sol played 7 in last trick and lost)
  - parti win rate (sol_pts ≥ 50)
  - mean & std of total sol GP per deal

Worker pool sets multi weights at startup via initializer (per-process
globals).
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

from ulti.eval.dojo import deal_ulti_biased
from ulti.eval.pimc_matchup import god_pick
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


def _init_worker():
    # Set the per-process multi weights once at worker startup.
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


def _play_one(*, deal, sol_contract, pimc_n, seed):
    """Play a full deal. Sol uses PIMC with the given contract; defenders
    use god (god picks for parti — defenders' true objective in a parti)."""
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


def worker(args):
    seed, = args
    deal = deal_ulti_biased(seed=seed, alpha=ALPHA)

    final_parti = _play_one(deal=deal, sol_contract='parti',
                            pimc_n=PIMC_N, seed=seed)
    final_multi = _play_one(deal=deal, sol_contract='multi',
                            pimc_n=PIMC_N, seed=seed)

    return (seed, _summary(final_parti), _summary(final_multi))


_BID = BidSet(parti=True)


def _summary(final_pos):
    """Per-deal facts the oracle would score."""
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
        "silent_ulti_signed": _signed_silent_ulti(side, won),
        "total_gp_per_def":  pv.total_per_def,
    }


def _signed_silent_ulti(side, won):
    if side == "soloist":
        return +1 if won else -2
    if side == "defender":
        return -1 if won else +2
    return 0


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / n
    return m, v ** 0.5


def main():
    jobs = [(SEED_BASE + i,) for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(worker, jobs, chunksize=1):
            rows.append(r)
            if len(rows) % 5 == 0:
                wall = time.perf_counter() - t0
                rate = len(rows) / wall
                eta = (N - len(rows)) / rate if rate else 0
                print(f"  {len(rows)}/{N}  wall={wall:.0f}s  eta={eta:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    # Aggregate per engine
    for label, idx in [("parti (baseline)", 1), ("multi[parti+ulti]", 2)]:
        sums = [r[idx] for r in rows]
        parti_wr  = sum(s["parti_won"] for s in sums) / N
        pts_m, pts_s = _mean_std([s["sol_pts"] for s in sums])
        sol_silent_ulti  = sum(1 for s in sums if s["ulti_side"] == "soloist" and s["ulti_won"])  / N
        sol_bukott       = sum(1 for s in sums if s["ulti_side"] == "soloist" and not s["ulti_won"]) / N
        def_silent_ulti  = sum(1 for s in sums if s["ulti_side"] == "defender" and s["ulti_won"])  / N
        def_bukott       = sum(1 for s in sums if s["ulti_side"] == "defender" and not s["ulti_won"]) / N
        gp_m, gp_s = _mean_std([s["total_gp_per_def"] for s in sums])
        print()
        print(f"=== {label} ===")
        print(f"  parti win rate:           {parti_wr:.3f}")
        print(f"  sol parti pts:            {pts_m:.2f} ± {pts_s:.2f}")
        print(f"  silent ulti sol won:      {sol_silent_ulti:.3f}")
        print(f"  silent ulti sol bukott:   {sol_bukott:.3f}")
        print(f"  silent ulti def won:      {def_silent_ulti:.3f}")
        print(f"  silent ulti def bukott:   {def_bukott:.3f}")
        print(f"  total GP / def:           {gp_m:+.3f} ± {gp_s:.3f}")
        print(f"  total GP (×2 for sol):    {2*gp_m:+.3f}")
    print()
    print(f"Wall: {wall:.0f}s, {N_WORKERS} workers, N={N}, pimc_n={PIMC_N}, alpha={ALPHA}")


if __name__ == "__main__":
    main()
