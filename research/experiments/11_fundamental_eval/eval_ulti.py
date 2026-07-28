"""Fundamental-contract eval — ulti only.

Same template as eval_betli.py. Per (α, seed):
  1. Build the deal (deal_ulti_biased: 10-card sol + 2-card talon,
     trump-7 mandatory in sol's hand).
  2. God-label the opening position — does god predict sol can win
     trick 10 with trump-7?
  3. Compute PIMC32 value at t=0 from sol's seat — the value head.
  4. Play two head-to-head games on the same seed:
       a) PIMC sol vs god def → sol_hold
       b) god sol vs PIMC def → def_stop

PIMC defender now correctly plants the trump-7 in sol's hand per the
ulti auction must-hold rule (see solvers/determinize._ulti_must_hold).

Per α, aggregate:
  sol_hold              — PIMC takes / god-winnable
  def_stop              — PIMC denies / god-unwinnable
  value_auc             — AUC of pred (PIMC32 t=0) vs god label
  value_mean_solfav     — mean pred on sol-fav hands
  value_mean_deffav     — mean pred on def-fav hands
  brier                 — mean squared error of (pred/10 − label)

Parallel via multiprocessing.Pool. Per-result checkpoint to disk so a
mid-run kill loses nothing.
"""
from __future__ import annotations

import json, time
from multiprocessing import Pool
from pathlib import Path

from eval.dojo import deal_ulti_biased
from eval.pimc_matchup import (
    play_one, god_says_soloist_wins, defenders_won,
)
from solvers import pimc as _pimc
from solvers import pis as pis_bridge

_CONTRACT     = "ulti"
_VALUE_SCALE  = 10.0       # WIN=10, LOSE=0

ALPHAS    = [0.00, 0.60, 1.50]
N         = 200
PIMC_N    = 32
N_WORKERS = 8
SEED_BASE = 81_000_000
OUT_DIR   = Path(__file__).parent
CHECKPOINT_PATH = OUT_DIR / "checkpoint_ulti.jsonl"


def _value_at_open(pos, seed):
    _, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=_CONTRACT, n_samples=PIMC_N, seed=seed,
    )
    return float("nan") if not averaged else max(averaged.values())


def worker(args):
    alpha, seed = args
    deal = deal_ulti_biased(seed=seed, alpha=alpha)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos0 = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract=_CONTRACT,
        trump=deal.trump, talon=list(deal.talon),
    )
    label_sol = god_says_soloist_wins(pos0, _CONTRACT)
    pred      = _value_at_open(pos0, seed)

    final_a = play_one(
        deal=deal, contract=_CONTRACT,
        sol_strategy="pimc", def_strategy="god",
        pimc_n=PIMC_N, seed=seed,
    )
    pimcsol_sol_won = not defenders_won(final_a, _CONTRACT)

    final_b = play_one(
        deal=deal, contract=_CONTRACT,
        sol_strategy="god", def_strategy="pimc",
        pimc_n=PIMC_N, seed=seed,
    )
    godsol_sol_won = not defenders_won(final_b, _CONTRACT)

    return (alpha, seed, label_sol, pred, pimcsol_sol_won, godsol_sol_won)


def _auc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l]
    neg = [s for l, s in zip(labels, scores) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _se(p, n):
    return (p * (1 - p) / max(n, 1)) ** 0.5


def _load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return []
    out = []
    with open(CHECKPOINT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(tuple(json.loads(line)))
    return out


def main() -> None:
    all_jobs = [(a, SEED_BASE + int(a * 1e6) + i)
                for a in ALPHAS for i in range(N)]

    cached = _load_checkpoint()
    done_seeds = {(r[0], r[1]) for r in cached}
    jobs = [j for j in all_jobs if (j[0], j[1]) not in done_seeds]
    print(f"Resuming with {len(cached)}/{len(all_jobs)} in checkpoint; "
          f"{len(jobs)} remaining; {N_WORKERS} workers (N={N}/α, pimc_n={PIMC_N})",
          flush=True)

    rows = [(r[0], r[2], r[3], r[4], r[5]) for r in cached]
    t0 = time.perf_counter()
    done = 0
    with open(CHECKPOINT_PATH, "a") as ckpt:
        with Pool(N_WORKERS) as pool:
            for r in pool.imap_unordered(worker, jobs, chunksize=2):
                ckpt.write(json.dumps(list(r)) + "\n")
                ckpt.flush()
                rows.append((r[0], r[2], r[3], r[4], r[5]))
                done += 1
                if done % 25 == 0:
                    wall = time.perf_counter() - t0
                    rate = done / wall if wall else 0
                    eta  = (len(jobs) - done) / rate if rate else 0
                    print(f"  {done}/{len(jobs)}  wall={wall:.0f}s  "
                          f"rate={rate:.2f}/s  eta={eta:.0f}s", flush=True)
    wall_total = time.perf_counter() - t0
    print(f"\nDone in {wall_total:.0f}s (checkpoint has {len(rows)} rows)\n",
          flush=True)

    by_alpha: dict = {}
    for r in rows:
        by_alpha.setdefault(r[0], []).append(r)

    summary = []
    for a in ALPHAS:
        rs = by_alpha[a]
        labels = [r[1] for r in rs]
        preds  = [r[2] for r in rs]
        pwins  = [r[3] for r in rs]
        gwins  = [r[4] for r in rs]
        n      = len(rs)
        n_sf   = sum(labels); n_df = n - n_sf
        sol_hold = sum(1 for l, w in zip(labels, pwins) if l and w) / max(n_sf, 1)
        def_stop = sum(1 for l, w in zip(labels, gwins) if (not l) and (not w)) / max(n_df, 1)
        val_auc  = _auc(labels, preds)
        m_sf = sum(p for l, p in zip(labels, preds) if l)     / max(n_sf, 1)
        m_df = sum(p for l, p in zip(labels, preds) if not l) / max(n_df, 1)
        brier = sum(((p / _VALUE_SCALE) - (1.0 if l else 0.0)) ** 2
                    for l, p in zip(labels, preds)) / n
        summary.append({
            "alpha": a, "n": n,
            "n_sol_fav": n_sf, "n_def_fav": n_df,
            "sol_hold": sol_hold, "sol_hold_se": _se(sol_hold, n_sf),
            "def_stop": def_stop, "def_stop_se": _se(def_stop, n_df),
            "value_auc": val_auc,
            "value_mean_solfav": m_sf,
            "value_mean_deffav": m_df,
            "brier": brier,
        })

    print(f"{'α':>5} {'n_sf':>5} {'n_df':>5} "
          f"{'sol_hold':>15} {'def_stop':>15} {'val_AUC':>8} "
          f"{'mean(SF)':>9} {'mean(DF)':>9} {'Brier':>7}")
    for r in summary:
        print(f"{r['alpha']:>5} {r['n_sol_fav']:>5} {r['n_def_fav']:>5} "
              f"{r['sol_hold']:>7.3f}±{r['sol_hold_se']:.3f} "
              f"{r['def_stop']:>7.3f}±{r['def_stop_se']:.3f} "
              f"{r['value_auc']:>8.3f} "
              f"{r['value_mean_solfav']:>9.2f} {r['value_mean_deffav']:>9.2f} "
              f"{r['brier']:>7.3f}")

    out = {
        "contract": _CONTRACT, "n_per_alpha": N, "pimc_n": PIMC_N,
        "n_workers": N_WORKERS, "seed_base": SEED_BASE,
        "wall_s": wall_total, "rows": summary,
    }
    (OUT_DIR / "results_ulti.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_DIR / 'results_ulti.json'}")


if __name__ == "__main__":
    main()
