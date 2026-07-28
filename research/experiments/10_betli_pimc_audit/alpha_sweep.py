"""Betli vanilla-PIMC defender α sweep — corrected baseline.

Re-runs the same α grid as `vnet/betli`
(0.30, 0.50, 0.70, 0.80, 1.00) with god soloist vs vanilla PIMC defender,
using the centralised ``eval.pimc_matchup.pimc_pick`` (defender-aware).

Originally the comparison numbers in 04's CONCLUSIONS.md were generated
by a script that fed ``pimc_decision``'s ``chosen`` straight back as the
defender move — that's the soloist-perspective argmax, anti-optimal for
defenders. With the flip restored, vanilla PIMC matches or beats the
best published NN configuration at every α.

Usage:
    PYTHONPATH=. python3 experiments/10_betli_pimc_audit/alpha_sweep.py
"""
from __future__ import annotations
import json, time
from multiprocessing import Pool
from pathlib import Path

from ulti.eval.dojo import deal_betli
from ulti.eval.pimc_matchup import play_one, god_says_soloist_wins, defenders_won
from ulti.solvers import pis as pis_bridge

_CONTRACT = "betli"
ALPHAS    = [0.30, 0.50, 0.70, 0.80, 1.00]
N         = 200
PIMC_N    = 16
N_WORKERS = 8
SEED_BASE = 91_000_000
OUT_DIR   = Path(__file__).parent


def worker(args):
    alpha, seed = args
    deal = deal_betli(seed=seed, alpha=alpha)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos0 = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract=_CONTRACT,
        talon=list(deal.talon),
    )
    label_def_fav = not god_says_soloist_wins(pos0, _CONTRACT)
    final = play_one(
        deal=deal, contract=_CONTRACT,
        sol_strategy="god", def_strategy="pimc",
        pimc_n=PIMC_N, seed=seed,
    )
    def_won = defenders_won(final, _CONTRACT)
    return (alpha, label_def_fav, def_won)


def main() -> None:
    jobs = [(a, SEED_BASE + int(a * 1e6) + i)
            for a in ALPHAS for i in range(N)]
    print(f"Submitting {len(jobs)} games across {N_WORKERS} workers "
          f"(pimc_n={PIMC_N})", flush=True)

    results = {a: {"labels": [], "wins": []} for a in ALPHAS}
    grand_t0 = time.perf_counter()
    done = 0
    with Pool(N_WORKERS) as pool:
        for alpha, label, won in pool.imap_unordered(worker, jobs, chunksize=4):
            results[alpha]["labels"].append(label)
            results[alpha]["wins"].append(won)
            done += 1
            if done % 100 == 0:
                wall = time.perf_counter() - grand_t0
                rate = done / wall if wall else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"  {done}/{len(jobs)}  wall={wall:.1f}s  "
                      f"rate={rate:.2f}/s  eta={eta:.0f}s", flush=True)
    wall_total = time.perf_counter() - grand_t0

    rows = []
    mctsv_ref = {0.30: 0.755, 0.50: 0.503, 0.70: 0.444,
                 0.80: 0.320, 1.00: 0.295}
    print(f"\n# DONE in {wall_total:.1f}s\n")
    print(f"{'α':>5}  {'n_def':>6}  {'def_stop (PIMC)':>20}  {'MCTS+V ref':>12}")
    for a in ALPHAS:
        labels = results[a]["labels"]; wins = results[a]["wins"]
        n_def_fav = sum(labels)
        def_wins = sum(1 for l, w in zip(labels, wins) if l and w)
        def_stop = def_wins / max(n_def_fav, 1)
        se = (def_stop * (1 - def_stop) / max(n_def_fav, 1)) ** 0.5
        rows.append({
            "alpha": a, "n": N, "pimc_n": PIMC_N,
            "n_def_fav": n_def_fav,
            "def_stop_count": def_wins,
            "def_stop": def_stop, "def_stop_se": se,
            "mctsv_ref": mctsv_ref[a],
        })
        print(f"  {a:>3}  {n_def_fav:>6}  "
              f"{def_stop:>6.3f} ± {se:.3f}  ({def_wins:>3}/{n_def_fav})  "
              f"{mctsv_ref[a]:>10.3f}")

    (OUT_DIR / "sweep.json").write_text(json.dumps({
        "alphas": ALPHAS, "n": N, "pimc_n": PIMC_N,
        "n_workers": N_WORKERS, "wall_s": wall_total, "rows": rows,
    }, indent=2))
    print(f"\nwrote {OUT_DIR / 'sweep.json'}")


if __name__ == "__main__":
    main()
