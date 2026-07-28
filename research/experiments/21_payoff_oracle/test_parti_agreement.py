"""Do EV_PARTI and EV_MULTI[parti_pts=1] ever DISAGREE about a parti win?

Double-dummy (perfect-info) test: solve the SAME complete deals both ways and
compare the minimax WIN/LOSS. Both terminals are `scores[sol] > def_score`
(scaled 0/1 vs -1/+1), so they must agree unless a cull is unsound. Also report
how often they pick a DIFFERENT best move (tie-breaking among equally-optimal
parti lines) — that's the only thing PIMC can turn into a GP difference.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.eval.dojo import deal_ulti_biased
from ulti.solvers import pis as pis_bridge
from trickster._solver_core import solve_best, set_parti_mode, set_multi_weights

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    set_parti_mode(0)   # binary EV_PARTI: terminal 1.0 win / 0.0 loss
    win_disagree = 0
    move_disagree = 0
    t0 = time.perf_counter()
    for i in range(N):
        deal = deal_ulti_biased(seed=400_000 + i, alpha=0.6)
        hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        gs = pis_bridge.build_position(
            hands=hands, soloist=0, leader=0, contract='parti',
            trump=deal.trump, talon=list(deal.talon), declare_marriages=True,
        )
        mv_p, val_p = solve_best(gs, contract='parti')
        set_multi_weights(parti_pts=1.0)
        mv_m, val_m = solve_best(gs, contract='multi')

        win_p = val_p > 0.5     # EV_PARTI binary
        win_m = val_m > 0.0     # EV_MULTI parti_pts (+1/-1)
        if win_p != win_m:
            win_disagree += 1
            print(f"  [WIN DISAGREE] seed {400000+i}: parti={val_p} multi={val_m}")
        if mv_p != mv_m:
            move_disagree += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}  wall={time.perf_counter()-t0:.0f}s", flush=True)

    print()
    print(f"N={N} complete deals")
    print(f"  WIN/LOSS disagreements:  {win_disagree}/{N}"
          + ("   ✓ they NEVER disagree about a parti win" if win_disagree == 0
             else "   ⚠ UNSOUND"))
    print(f"  best-move differs (ties): {move_disagree}/{N}"
          f"   ({100*move_disagree/N:.0f}% — equally-optimal lines, the PIMC noise source)")


if __name__ == "__main__":
    main()
