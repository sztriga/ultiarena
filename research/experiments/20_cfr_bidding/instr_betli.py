"""Instrument the canon auction's COMMITTED betli/duri: decision_p, who bid it
(P0 open vs overtaker), god outcome. Resolves the contradiction between the
well-calibrated P0 decision_p (72% at 0.3) and the 8.6% committed win-rate."""
import os, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import simulate
from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.vnet.pickup.composite import CompositePickup

EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent.parent / "19_colorless_split"
N = int(os.environ.get("N_EVAL", 30000))
_P = None

def _pk():
    global _P
    if _P is None:
        _P = CompositePickup.load(trump_weights=EXP18/"multihead_v18a.pt",
            betli_weights=EXP19/"colorless_betli.pt",
            durchmars_weights=EXP19/"colorless_durchmars.pt")
    return _P

def _one(seed):
    p = _pk()
    r = simulate(seed, [p,p,p], play_out=False)
    if r['winner_pid'] is None or r['contract'] not in ('betli','durchmars'):
        return None
    pos = pis.build_position(hands=[r['sol_hand'],r['def1'],r['def2']], soloist=0,
        leader=0, contract=r['contract'], trump=None, talon=r['talon'])
    god = god_says_soloist_wins(pos, contract=r['contract'])
    return (r['contract'], float(r['win_p']), r['winner_pid'], r['n_pickups'],
            1 if god else 0)

def main():
    print(f"=== committed betli/duri instrumentation  N={N} ===", flush=True)
    seeds=[100000+i for i in range(N)]
    rows=[]
    t0=time.perf_counter()
    with Pool(8) as pool:
        for x in pool.imap_unordered(_one, seeds, chunksize=32):
            if x: rows.append(x)
    print(f"  {time.perf_counter()-t0:.0f}s   committed betli/duri: {len(rows)}\n")
    for c in ('betli','durchmars'):
        v=[r for r in rows if r[0]==c]
        if not v:
            print(f"{c}: none"); continue
        wp=np.array([r[1] for r in v]); god=np.array([r[4] for r in v])
        opener=sum(1 for r in v if r[3]==1)      # n_pickups==1 ⇒ P0 uncontested
        overtk=len(v)-opener
        print(f"=== {c}: n={len(v)} ({len(v)/N*100:.2f}%)  god-win {god.mean()*100:.1f}% ===")
        print(f"  by route: P0-opened(uncontested) {opener}  overtaken/contested {overtk}")
        for tag,mask in [("P0-opened",[r[3]==1 for r in v]),
                         ("overtaken",[r[3]>1 for r in v])]:
            mask=np.array(mask)
            if mask.sum():
                print(f"    {tag}: n={mask.sum()}  mean win_p {wp[mask].mean():.3f}"
                      f"  god-win {god[mask].mean()*100:.1f}%")
        print(f"  decision_p (win_p) bins:")
        for lo,hi in [(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,1.01)]:
            m=(wp>=lo)&(wp<hi)
            if m.sum(): print(f"    [{lo:.1f},{hi:.1f}): n={m.sum():>4}  "
                              f"god-win {god[m].mean()*100:5.1f}%")

if __name__=="__main__":
    main()
