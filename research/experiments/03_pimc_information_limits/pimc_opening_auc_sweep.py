"""Opening PIMC: AUC vs N sweep (per-viewpoint).

For each deal, sample N_MAX worlds per viewer and record the binary
soloist-win outcome of each. For each N in N_GRID use the first N samples
to form a probability, then compute ROC-AUC against the god label.

Soloist viewpoint and defender viewpoint (averaged across def1, def2) are
scored separately.

Usage:
    PYTHONPATH=. python3 scripts/pimc_opening_auc_sweep.py --seeds 50 --alpha 0.7 --nmax 16
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ulti.eval.dojo import deal_betli
from ulti.solvers import determinize as _det
from ulti.solvers import pis as pis_bridge

_BETLI_WIN_VAL = 10.0
_CONTRACT = "betli"


def _samples_for_viewer(true_pos, viewer: int, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    iset = _det.build_info_set(true_pos, viewer, _CONTRACT)
    out = []
    for _ in range(n):
        hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            sample_pos = pis_bridge.clone_with_hands_and_talon(true_pos, hands, talon)
        else:
            sample_pos = pis_bridge.clone_with_hands(true_pos, hands)
        vals = pis_bridge.solve_all(sample_pos, contract=_CONTRACT)
        out.append(1 if max(vals.values()) >= _BETLI_WIN_VAL - 1e-6 else 0)
    return out


def _auc(labels: list[int], scores: list[float]) -> float:
    """ROC-AUC by Mann-Whitney U over score ranks (handles ties)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    paired = sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg)
    return paired / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--nmax",  type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    n_grid = [1, 2, 4, 8, 16, 32, 64, 128]
    n_grid = [n for n in n_grid if n <= args.nmax]

    print(f"AUC sweep: α={args.alpha}, deals=[{args.start}, {args.start + args.seeds}), "
          f"N_max={args.nmax}, grid={n_grid}\n")

    god_labels: list[int] = []      # 1 = soloist win, 0 = defender win
    sol_samples: list[list[int]] = []  # per-deal binary samples from sol viewer
    d1_samples:  list[list[int]] = []
    d2_samples:  list[list[int]] = []

    t0 = time.perf_counter()
    for s in range(args.start, args.start + args.seeds):
        deal = deal_betli(seed=s, alpha=args.alpha)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        pos = pis_bridge.build_position(
            hands=hands, soloist=0, leader=0, contract=_CONTRACT,
            talon=list(deal.talon),
        )
        gv = pis_bridge.solve_all(pos, contract=_CONTRACT)
        god_labels.append(1 if max(gv.values()) >= _BETLI_WIN_VAL - 1e-6 else 0)

        sol_samples.append(_samples_for_viewer(pos, 0, args.nmax, s * 7919 + 1))
        d1_samples .append(_samples_for_viewer(pos, 1, args.nmax, s * 7919 + 2))
        d2_samples .append(_samples_for_viewer(pos, 2, args.nmax, s * 7919 + 3))
    t_sample = time.perf_counter() - t0

    rows = []
    print(f"{'N':>3}  {'AUC_sol':>8}  {'AUC_def':>8}  {'sol_p_mean':>11}  {'def_p_mean':>11}")
    for n in n_grid:
        sol_p = [sum(samps[:n]) / n for samps in sol_samples]
        d1_p  = [sum(samps[:n]) / n for samps in d1_samples]
        d2_p  = [sum(samps[:n]) / n for samps in d2_samples]
        def_p = [(a + b) / 2 for a, b in zip(d1_p, d2_p)]

        auc_sol = _auc(god_labels, sol_p)
        auc_def = _auc(god_labels, def_p)

        rows.append({
            "n": n, "auc_sol": auc_sol, "auc_def": auc_def,
            "sol_p_mean": sum(sol_p) / len(sol_p),
            "def_p_mean": sum(def_p) / len(def_p),
        })
        print(f"{n:>3}  {auc_sol:>8.3f}  {auc_def:>8.3f}  "
              f"{sum(sol_p)/len(sol_p):>11.3f}  {sum(def_p)/len(def_p):>11.3f}",
              flush=True)

    out = {
        "alpha": args.alpha, "n_deals": args.seeds,
        "god_pos_rate": sum(god_labels) / len(god_labels),
        "n_grid": n_grid, "rows": rows,
        "wall_s": time.perf_counter() - t0,
    }
    out_path = Path("/tmp/auc_sweep.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nGod positive (soloist) rate: {out['god_pos_rate']:.2%}")
    print(f"Sampling wall: {t_sample:.1f}s   total: {out['wall_s']:.1f}s")
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
