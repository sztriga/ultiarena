"""Evaluate CompositePickup (v18a trump + structural colorless) vs v18a.

MODE=god  : god-win% of the hands the auction commits to (decision-region
            overconfidence test). Compare betli/duri to v18a's ~15%.
MODE=play : full PIMC32-vs-god playout — per-contract GP/won% (bleeders)
            and seat totals. Compare betli's −3.79 / P0's −0.302.

Usage: MODE=god N_DEALS=3000 python eval_composite.py
       MODE=play N_DEALS=3000 python eval_composite.py
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import simulate
from ulti.solvers import pis
from ulti.eval.pimc_matchup import god_says_soloist_wins
from ulti.vnet.pickup.composite import CompositePickup

EXP17 = Path(__file__).parent.parent / "17_clean_pickup_net"
EXP19 = Path(__file__).parent

N         = int(os.environ.get("N_DEALS", 3000))
MODE      = os.environ.get("MODE", "god")
SEED_BASE = 100_000
N_WORKERS = 8

# v18a reference (single-model) for side-by-side.
V18A_GODWIN = {'ulti/hearts': 77.5, 'parti/hearts': 21.5, 'ulti/bells': 80.0,
               'ulti/leaves': 73.3, 'ulti/acorns': 76.8,
               'betli/colorless': 14.9, 'durchmars/colorless': 40.0}
V18A_PLAY = {  # (freq%, won%, GP/def)
    'betli/colorless': (4.7, 12.1, -3.79),
    'durchmars/colorless': (0.5, 40.0, -1.20),
}
V18A_SEAT = {'P0': -0.302, 'P1': 0.205, 'P2': 0.098}

_PICKER = None


def _get():
    global _PICKER
    if _PICKER is None:
        _PICKER = CompositePickup.load(
            trump_weights=Path(__file__).parent.parent /
                "18_canonical_pickup" / "multihead_v18a.pt",
            betli_weights=EXP19 / "colorless_betli.pt",
            durchmars_weights=EXP19 / "colorless_durchmars.pt",
        )
    return _PICKER


def _run_god(seed):
    p = _get()
    r = simulate(seed, [p, p, p], play_out=False)
    if r['winner_pid'] is None:
        return ('PASS_PENALTY', None)
    pos = pis.build_position(
        hands=[r['sol_hand'], r['def1'], r['def2']], soloist=0, leader=0,
        contract=r['contract'], trump=r['trump'], talon=r['talon'])
    win = god_says_soloist_wins(pos, contract=r['contract'])
    return (f"{r['contract']}/{r['trump'] or 'colorless'}", bool(win))


def _run_play(seed):
    p = _get()
    return simulate(seed, [p, p, p], play_out=True)


def main():
    print(f"=== Composite eval  MODE={MODE}  N={N} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()

    if MODE == "god":
        by = defaultdict(list)
        done = 0
        with Pool(N_WORKERS) as pool:
            for bid, win in pool.imap_unordered(_run_god, seeds, chunksize=8):
                if win is not None:
                    by[bid].append(win)
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{N}  {time.perf_counter()-t0:.0f}s", flush=True)
        print(f"  wall {time.perf_counter()-t0:.0f}s\n")
        print("god-win% of committed hands (composite vs v18a):")
        print(f"  {'contract':>22}  {'n':>5}  {'god-win%':>9}  {'v18a':>7}")
        for bid in sorted(by, key=lambda b: -len(by[b])):
            w = by[bid]
            ref = V18A_GODWIN.get(bid)
            print(f"  {bid:>22}  {len(w):>5}  {100*sum(w)/len(w):>8.1f}  "
                  f"{(f'{ref:.1f}' if ref is not None else '—'):>7}")
        return

    # MODE == play
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_run_play, seeds, chunksize=4):
            rows.append(r)
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{N}  {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"  wall {time.perf_counter()-t0:.0f}s\n")

    print("Per-contract (composite):")
    print(f"  {'contract':>22}  {'n':>5}  {'freq%':>6}  {'won%':>6}  "
          f"{'GP/def':>8}   v18a(freq/won/GP)")
    bycon = defaultdict(list)
    for r in rows:
        bycon[r['winning_bid']].append(r['gp_per_def'])
    for k in sorted(bycon, key=lambda x: -len(bycon[x])):
        v = bycon[k]
        won = sum(1 for x in v if x > 0)
        ref = V18A_PLAY.get(k)
        rs = (f"{ref[0]}/{ref[1]}/{ref[2]:+.2f}" if ref else "")
        print(f"  {k:>22}  {len(v):>5}  {len(v)/N*100:>5.1f}  "
              f"{won/len(v)*100:>5.1f}  {sum(v)/len(v):>+8.2f}   {rs}")

    print("\nSeat totals (composite vs v18a GP/deal):")
    for pid in (0, 1, 2):
        net = sum(r['gps'][pid] for r in rows) / N
        print(f"  P{pid}: {net:+.3f}   (v18a {V18A_SEAT[f'P{pid}']:+.3f})")


if __name__ == "__main__":
    main()
