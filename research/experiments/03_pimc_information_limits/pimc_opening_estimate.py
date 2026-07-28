"""Opening PIMC estimate per viewpoint (no game played).

For each deal:
  * god verdict from the true hands;
  * sample N worlds from each player's info set, perfect-solve each,
    record fraction that come out as a soloist win.
  * "soloist opinion" = sol's win rate
  * "defender opinion" = average of def1's and def2's win rates

Usage:
    PYTHONPATH=. python3 scripts/pimc_opening_estimate.py --seeds 50 --alpha 0.7 --n 1
"""
from __future__ import annotations

import argparse
import random
import time

from ulti.eval.dojo import deal_betli
from ulti.solvers import determinize as _det
from ulti.solvers import pis as pis_bridge

_BETLI_WIN_VAL = 10.0
_CONTRACT = "betli"


def _viewpoint_winrate(true_pos, viewer: int, n: int, seed: int) -> float:
    rng = random.Random(seed)
    iset = _det.build_info_set(true_pos, viewer, _CONTRACT)
    wins = 0
    for _ in range(n):
        hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            sample_pos = pis_bridge.clone_with_hands_and_talon(true_pos, hands, talon)
        else:
            sample_pos = pis_bridge.clone_with_hands(true_pos, hands)
        vals = pis_bridge.solve_all(sample_pos, contract=_CONTRACT)
        if max(vals.values()) >= _BETLI_WIN_VAL - 1e-6:
            wins += 1
    return wins / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--n",     type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    print(f"Opening PIMC estimate: N={args.n}, α={args.alpha}, "
          f"{args.seeds} deals from seed {args.start}")
    print(f"{'seed':>4}  {'god':>10}  {'sol_op':>6}  {'def_op':>6}")
    print("-" * 40)

    rows = []
    t0 = time.perf_counter()
    for s in range(args.start, args.start + args.seeds):
        deal = deal_betli(seed=s, alpha=args.alpha)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        pos = pis_bridge.build_position(
            hands=hands, soloist=0, leader=0, contract=_CONTRACT,
            talon=list(deal.talon),
        )
        god_vals = pis_bridge.solve_all(pos, contract=_CONTRACT)
        god = "soloist" if max(god_vals.values()) >= _BETLI_WIN_VAL - 1e-6 else "defenders"

        sol_op = _viewpoint_winrate(pos, viewer=0, n=args.n, seed=s * 7919 + 1)
        def1   = _viewpoint_winrate(pos, viewer=1, n=args.n, seed=s * 7919 + 2)
        def2   = _viewpoint_winrate(pos, viewer=2, n=args.n, seed=s * 7919 + 3)
        def_op = (def1 + def2) / 2

        rows.append((s, god, sol_op, def_op))
        print(f"{s:>4}  {god:>10}  {sol_op:>6.2f}  {def_op:>6.2f}", flush=True)

    elapsed = time.perf_counter() - t0

    # Summary tables: (god verdict) × (opinion bucket)
    print()
    for label, get in [("Soloist opinion (sol viewpoint)", lambda r: r[2]),
                       ("Defender opinion (avg def1, def2)", lambda r: r[3])]:
        gs_sol = sum(1 for r in rows if r[1] == "soloist"  and get(r) >= 0.5)
        gs_def = sum(1 for r in rows if r[1] == "soloist"  and get(r) <  0.5)
        gd_sol = sum(1 for r in rows if r[1] == "defenders" and get(r) >= 0.5)
        gd_def = sum(1 for r in rows if r[1] == "defenders" and get(r) <  0.5)
        print(f"{label}  (threshold ≥0.5 = predicts soloist):")
        print(f"                     pred=sol  pred=def")
        print(f"  god=soloist          {gs_sol:>3}      {gs_def:>3}")
        print(f"  god=defenders        {gd_sol:>3}      {gd_def:>3}")
        print()
    print(f"Total wall: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
