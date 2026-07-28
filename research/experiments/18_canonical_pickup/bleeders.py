"""Comprehensive contract/seat scorecard for the v18a symmetric auction.

Runs the N=3000 symmetric auction (all three seats = v18a, def=god,
sol=PIMC32, PASS_PENALTY=-2) and produces:

  Table 1 — contract scorecard (soloist view): freq, won%, GP/def, total
  Table 2 — contract × soloist seat: who solos what, how it scores
  Table 3 — seat totals: solo/defend counts + net GP/deal
  Table 4 — net GP per seat split by contract (incl. defender side)

Bleeders (avg GP/def < 0 with n≥20) are flagged in Table 1.

Caveat: numbers carry the soloist-PIMC32 vs god-defender handicap.

Usage: N_DEALS=3000 python bleeders.py
"""
from __future__ import annotations

import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))

from auction_h2h import simulate
from ulti.vnet.pickup.v18 import Exp18Pickup

EXP_DIR   = Path(__file__).parent
WEIGHTS   = EXP_DIR / "multihead_v18a.pt"
N         = int(os.environ.get("N_DEALS", 3000))
SEED_BASE = 100_000
N_WORKERS = 8

_PICKER = None


def _get():
    global _PICKER
    if _PICKER is None:
        _PICKER = Exp18Pickup.load(WEIGHTS)
    return _PICKER


def _run_one(seed):
    p = _get()
    return simulate(seed, [p, p, p])


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    print(f"=== v18a bleeder scorecard: N={N}, def=god, sol=PIMC32 ===",
          flush=True)
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(_run_one, seeds, chunksize=4):
            rows.append(r)
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{N}  wall={time.perf_counter()-t0:.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0
    print(f"  wall: {wall:.0f}s ({wall/60:.1f} min)\n")

    with open(EXP_DIR / "bleeders_rows.json", "w") as f:
        json.dump(rows, f)

    contracts = sorted({r['winning_bid'] for r in rows},
                       key=lambda b: -sum(1 for r in rows
                                          if r['winning_bid'] == b))

    # ── Table 1 — contract scorecard (soloist view) ──────────────────
    print("TABLE 1 — Contract scorecard (soloist/winner view)")
    print(f"  {'contract':>22}  {'n':>5}  {'freq%':>6}  {'won%':>6}  "
          f"{'GP/def':>8}  {'sol total':>9}  flag")
    for b in contracts:
        vals = [r['gp_per_def'] for r in rows if r['winning_bid'] == b]
        n = len(vals)
        won = sum(1 for v in vals if v > 0)
        avg = _mean(vals)
        # soloist total GP from this contract = sum of 2*gp_per_def
        sol_total = sum(2 * v for v in vals) if b != 'PASS_PENALTY' \
            else sum(2 * (-2.0) for _ in vals)
        flag = ""
        if b != 'PASS_PENALTY' and avg < 0 and n >= 20:
            flag = "◀ BLEEDER"
        elif b == 'PASS_PENALTY':
            flag = "(forced pass)"
        print(f"  {b:>22}  {n:>5}  {n/N*100:>5.1f}  "
              f"{won/n*100 if n else 0:>5.1f}  {avg:>+8.2f}  "
              f"{sol_total:>+9.0f}  {flag}")

    # ── Table 2 — contract × soloist seat ────────────────────────────
    print("\nTABLE 2 — Contract × soloist seat (how each seat scores when it solos)")
    print(f"  {'contract':>22}  {'seat':>4}  {'n':>5}  {'won%':>6}  {'GP/def':>8}")
    for b in contracts:
        if b == 'PASS_PENALTY':
            continue
        for pid in (0, 1, 2):
            vals = [r['gp_per_def'] for r in rows
                    if r['winning_bid'] == b and r['winner_pid'] == pid]
            if not vals:
                continue
            won = sum(1 for v in vals if v > 0)
            print(f"  {b:>22}  P{pid:>3}  {len(vals):>5}  "
                  f"{won/len(vals)*100:>5.1f}  {_mean(vals):>+8.2f}")

    # ── Table 3 — seat totals ────────────────────────────────────────
    print("\nTABLE 3 — Seat totals")
    print(f"  {'seat':>4}  {'soloed':>6}  {'defended':>8}  {'passes-as-opener':>16}  "
          f"{'net GP':>8}  {'GP/deal':>8}")
    n_pen = sum(1 for r in rows if r['winner_pid'] is None)
    for pid in (0, 1, 2):
        soloed = sum(1 for r in rows if r['winner_pid'] == pid)
        defended = sum(1 for r in rows
                       if r['winner_pid'] is not None and r['winner_pid'] != pid)
        net = sum(r['gps'][pid] for r in rows)
        # P0 is the only forced opener; penalty deals hit P0 as "opener pass"
        opener_pass = n_pen if pid == 0 else 0
        print(f"  P{pid:>3}  {soloed:>6}  {defended:>8}  {opener_pass:>16}  "
              f"{net:>+8.0f}  {net/N:>+8.3f}")
    print(f"  (forced-pass deals: {n_pen})")

    # ── Table 4 — net GP per seat split by contract (incl. defending) ─
    print("\nTABLE 4 — Net GP/deal per seat, by contract (soloist + defender side)")
    print(f"  {'contract':>22}  {'P0':>8}  {'P1':>8}  {'P2':>8}")
    for b in contracts:
        sub = [r for r in rows if r['winning_bid'] == b]
        cells = []
        for pid in (0, 1, 2):
            cells.append(sum(r['gps'][pid] for r in sub) / N)
        print(f"  {b:>22}  {cells[0]:>+8.3f}  {cells[1]:>+8.3f}  {cells[2]:>+8.3f}")


if __name__ == "__main__":
    main()
