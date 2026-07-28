"""Durchmars PIMC-vs-god α sweep, parallelised across worker processes.

For each (mode, α) we precompute a god label per deal (does sol have a
winning line?), then play two matchups on the same seeds:
  * PIMC-as-sol  vs  god-as-def    → ``sol_hold``  (capture of winnable)
  * god-as-sol   vs  PIMC-as-def   → ``def_stop``  (deny of unwinnable)

PIMC uses ``eval.pimc_matchup.pimc_pick`` which already MIN-flips the
soloist-perspective value for defender seats.

Usage:
    PYTHONPATH=. python3 experiments/09_duri_pipeline/pimc_alpha_sweep.py
"""
from __future__ import annotations
import json, time
from multiprocessing import Pool
from pathlib import Path

from ulti.eval.dojo import deal_durchmars_colored, deal_durchmars_colorless
from ulti.eval.pimc_matchup import play_one, god_says_soloist_wins, defenders_won
from ulti.solvers import pis as pis_bridge

_CONTRACT = "durchmars"
N_DEALS   = 50
PIMC_N    = 16
N_WORKERS = 8
SEED_BASE = 51_000_000

ALPHAS_COLORED   = [0.0, 0.3, 0.6, 1.0, 1.5]
ALPHAS_COLORLESS = [0.5, 1.0, 1.5, 2.0, 2.5]
OUT_DIR          = Path(__file__).parent


_DEAL_FNS = {
    "colored":   deal_durchmars_colored,
    "colorless": deal_durchmars_colorless,
}


def _label_deal(args):
    mode, alpha, seed = args
    d = _DEAL_FNS[mode](seed=seed, alpha=alpha)
    hands = [list(d.sol_hand), list(d.def1_hand), list(d.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract=_CONTRACT,
        trump=d.trump, talon=list(d.talon),
    )
    return (mode, alpha, seed, god_says_soloist_wins(pos, _CONTRACT))


def _play_match(args):
    mode, alpha, seed, sol_strat, def_strat = args
    d = _DEAL_FNS[mode](seed=seed, alpha=alpha)
    final = play_one(
        deal=d, contract=_CONTRACT,
        sol_strategy=sol_strat, def_strategy=def_strat,
        pimc_n=PIMC_N, seed=seed,
    )
    return (mode, alpha, seed, sol_strat, def_strat,
            defenders_won(final, _CONTRACT))


def main() -> None:
    grand_t0 = time.perf_counter()

    # Phase 1 — god labels (parallel)
    label_jobs = []
    for mode, alphas in (("colored", ALPHAS_COLORED),
                         ("colorless", ALPHAS_COLORLESS)):
        for a in alphas:
            for i in range(N_DEALS):
                label_jobs.append((mode, a, SEED_BASE + int(a * 1e6) + i))
    print(f"Labelling {len(label_jobs)} deals across {N_WORKERS} workers...",
          flush=True)
    labels: dict = {}
    with Pool(N_WORKERS) as pool:
        for mode, a, seed, lab in pool.imap_unordered(_label_deal, label_jobs, chunksize=8):
            labels[(mode, a, seed)] = lab
    print(f"  labelled in {time.perf_counter() - grand_t0:.1f}s\n", flush=True)

    # Phase 2 — play matches (parallel)
    match_jobs = []
    for mode, alphas in (("colored", ALPHAS_COLORED),
                         ("colorless", ALPHAS_COLORLESS)):
        for a in alphas:
            for i in range(N_DEALS):
                seed = SEED_BASE + int(a * 1e6) + i
                for sol_s, def_s in [("pimc", "god"), ("god", "pimc")]:
                    match_jobs.append((mode, a, seed, sol_s, def_s))
    print(f"Playing {len(match_jobs)} matches across {N_WORKERS} workers...",
          flush=True)
    t_match0 = time.perf_counter()
    results: dict = {}
    done = 0
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_play_match, match_jobs, chunksize=4):
            mode, a, seed, sol_s, def_s, def_won = r
            results.setdefault((mode, a, sol_s, def_s), []).append(
                (labels[(mode, a, seed)], def_won)
            )
            done += 1
            if done % 200 == 0:
                wall = time.perf_counter() - t_match0
                rate = done / wall if wall else 0
                eta = (len(match_jobs) - done) / rate if rate else 0
                print(f"  {done}/{len(match_jobs)}  wall={wall:.1f}s  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)
    print(f"  matches done in {time.perf_counter() - t_match0:.1f}s\n", flush=True)

    # Phase 3 — aggregate
    rows = []
    for (mode, a, sol_s, def_s), pairs in sorted(results.items()):
        n = len(pairs)
        n_sol_fav = sum(1 for lab, _ in pairs if lab)
        n_def_fav = n - n_sol_fav
        sol_hold = sum(1 for lab, d in pairs if lab and not d) / max(n_sol_fav, 1)
        def_stop = sum(1 for lab, d in pairs if (not lab) and d) / max(n_def_fav, 1)
        rows.append({
            "mode": mode, "alpha": a, "matchup": f"{sol_s} vs {def_s}",
            "n": n, "n_sol_fav": n_sol_fav, "n_def_fav": n_def_fav,
            "sol_hold": sol_hold, "def_stop": def_stop,
        })

    print(f"\n# SUMMARY (N={N_DEALS}/α, pimc_n={PIMC_N})\n")
    print(f"{'mode':<10} {'α':>5} {'matchup':<14} "
          f"{'sol_fav':>8} {'def_fav':>8} {'sol_hold':>10} {'def_stop':>10}")
    for r in rows:
        print(f"{r['mode']:<10} {r['alpha']:>5} {r['matchup']:<14} "
              f"{r['n_sol_fav']:>8} {r['n_def_fav']:>8} "
              f"{r['sol_hold']:>10.3f} {r['def_stop']:>10.3f}")

    out = {
        "n_per_alpha": N_DEALS, "pimc_n": PIMC_N, "n_workers": N_WORKERS,
        "rows": rows, "wall_s": time.perf_counter() - grand_t0,
    }
    (OUT_DIR / "pimc_alpha_sweep.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'pimc_alpha_sweep.json'}")


if __name__ == "__main__":
    main()
