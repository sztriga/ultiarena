"""Compare new (binary parti + marriages baked in) vs old (margin parti,
no marriages) on parti-biased deals.

Two cells:
  A: legacy   — parti_mode=margin, declare_marriages=False
  B: new      — parti_mode=binary, declare_marriages=True

Both sides PIMC. Same seeds in both cells. After play, the final
position has marriages added to scores (for fair scoring via the
oracle) regardless of which cell played, so the GP comparison reflects
the *real* GP earned, not just the agent's internal view.

Caveat: PIMC determinization does not currently constrain sampled
worlds to match declared marriages (no must_hold for K+Q), so the
sampled worlds in cell B may have different marriage placements than
reality. The score totals carried into samples are the *true* marriage
points, so the value-of-position signal is correct on average; only
the trick-by-trick play within determinizations is slightly off.
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

from eval.dojo import deal_parti
from solvers import determinize as _det
from solvers import pimc as _pimc
from solvers import pis as pis_bridge
from trickster._solver_core import set_parti_mode
from trickster.games.ulti.game import declare_all_marriages

sys.path.insert(0, str(Path(__file__).parent))
from scoring.oracle import BidSet, score as score_oracle

N         = 50
PIMC_N    = 32
ALPHA     = 0.6
SEED_BASE = 80_000_000
N_WORKERS = 4

CELLS = [
    # (label, parti_mode int, declare_marriages bool)
    ("A_legacy_margin", 1, False),
    ("B_new_binary_marriages", 0, True),
]


def _pick_pimc(*, pos, contract, n_samples, seed, voids_dict):
    chosen, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, voids=voids_dict,
    )
    viewer = pis_bridge.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


def _play_one(*, deal, parti_mode, declare_marr, seed, pimc_n):
    set_parti_mode(parti_mode)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract='parti',
        trump=deal.trump, talon=list(deal.talon),
        declare_marriages=declare_marr,
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)
    move_i = 0
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        chosen = _pick_pimc(
            pos=pos, contract='parti', n_samples=pimc_n,
            seed=seed * 31337 + move_i, voids_dict=voids_dict,
        )
        if chosen is None:
            chosen = rng.choice(pis_bridge.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis_bridge.apply_move(pos, chosen)
        move_i += 1

    # For fair scoring: if the cell didn't bake in marriages, add them now
    # so the oracle sees correct sol_total vs def_total. Easiest way:
    # build a temp position with declare_marriages=True and read off the
    # marriage point totals from its starting scores.
    if not declare_marr:
        tmp = pis_bridge.build_position(
            hands=hands, soloist=0, leader=0, contract='parti',
            trump=deal.trump, talon=list(deal.talon),
            declare_marriages=True,
        )
        for i in range(3):
            pos.scores[i] += tmp.scores[i]
        # Also copy the marriages list so the oracle credits silent_forty
        # / silent_twenty GP components correctly.
        pos.marriages = list(tmp.marriages)
        pos.marriages_declared = True
    return pos


_BID = BidSet(parti=True)


def _summary(final_pos):
    pv = score_oracle(final_pos=final_pos, bid=_BID)
    from trickster.games.ulti.game import last_trick_ulti_check
    side, won = last_trick_ulti_check(final_pos)
    sol = final_pos.scores[final_pos.soloist]
    defs = sum(final_pos.scores[p] for p in range(3) if p != final_pos.soloist)
    return {
        "parti_won":   sol > defs,
        "sol_total":   sol,
        "def_total":   defs,
        "ulti_side":   side,
        "ulti_won":    won,
        "total_gp_per_def":  pv.total_per_def,
    }


def worker(args):
    seed, label, parti_mode, declare_marr = args
    deal = deal_parti(seed=seed, alpha=ALPHA)
    final = _play_one(deal=deal, parti_mode=parti_mode,
                      declare_marr=declare_marr, seed=seed, pimc_n=PIMC_N)
    return (seed, label, _summary(final))


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
    sol_m, sol_s = _mean_std([s["sol_total"] for s in sums])
    def_m, def_s = _mean_std([s["def_total"] for s in sums])
    sol_ulti = sum(1 for s in sums if s["ulti_side"] == "soloist" and s["ulti_won"])  / n
    sol_buk  = sum(1 for s in sums if s["ulti_side"] == "soloist" and not s["ulti_won"]) / n
    def_ulti = sum(1 for s in sums if s["ulti_side"] == "defender" and s["ulti_won"])  / n
    def_buk  = sum(1 for s in sums if s["ulti_side"] == "defender" and not s["ulti_won"]) / n
    gp_m, gp_s = _mean_std([s["total_gp_per_def"] for s in sums])
    print()
    print(f"=== {label} ===")
    print(f"  parti win rate:           {parti_wr:.3f}")
    print(f"  sol total pts (incl marr):{sol_m:.2f} ± {sol_s:.2f}")
    print(f"  def total pts (incl marr):{def_m:.2f} ± {def_s:.2f}")
    print(f"  silent ulti sol won:      {sol_ulti:.3f}")
    print(f"  silent ulti sol bukott:   {sol_buk:.3f}")
    print(f"  silent ulti def won:      {def_ulti:.3f}")
    print(f"  silent ulti def bukott:   {def_buk:.3f}")
    print(f"  total GP / def:           {gp_m:+.3f} ± {gp_s:.3f}")
    print(f"  total GP (×2 for sol):    {2*gp_m:+.3f}")


def main():
    jobs = [(SEED_BASE + i, label, mode, marr)
            for (label, mode, marr) in CELLS
            for i in range(N)]
    print(f"Total jobs: {len(jobs)} ({len(CELLS)} cells × N={N})", flush=True)

    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(worker, jobs, chunksize=1):
            rows.append(r)
            if len(rows) % 10 == 0:
                wall = time.perf_counter() - t0
                rate = len(rows) / wall
                eta = (len(jobs) - len(rows)) / rate if rate else 0
                print(f"  {len(rows)}/{len(jobs)}  wall={wall:.0f}s  eta={eta:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0

    by_cell = {}
    for _, label, summary in rows:
        by_cell.setdefault(label, []).append(summary)

    for label, _, _ in CELLS:
        _print_cell(label, by_cell[label])

    print()
    print("=== Head-to-head (per-seed match) ===")
    # Pair by seed
    by_seed = {}
    for seed, label, summary in rows:
        by_seed.setdefault(seed, {})[label] = summary
    a_label = CELLS[0][0]; b_label = CELLS[1][0]
    a_wins = b_wins = ties = 0
    a_gp = b_gp = 0.0
    for seed, d in by_seed.items():
        a = d[a_label]["total_gp_per_def"]
        b = d[b_label]["total_gp_per_def"]
        a_gp += a; b_gp += b
        if b > a: b_wins += 1
        elif a > b: a_wins += 1
        else: ties += 1
    n_pairs = len(by_seed)
    print(f"  per-seed GP wins:  A={a_wins}  B={b_wins}  ties={ties}  (of {n_pairs})")
    print(f"  mean GP per def:   A={a_gp/n_pairs:+.3f}  B={b_gp/n_pairs:+.3f}  Δ(B-A)={((b_gp-a_gp)/n_pairs):+.3f}")

    print()
    print(f"Wall: {wall:.0f}s, {N_WORKERS} workers, N={N}, pimc_n={PIMC_N}, alpha={ALPHA}, dealer=parti-biased")


if __name__ == "__main__":
    main()
