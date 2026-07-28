"""Minigame bid eval — full pipeline:

  1. Deal 12 cards to sol, 10 to each defender.
  2. Sol evaluates all 66 discards × {4 trumps × (parti, ulti),
     betli, durchmars} via PIMC32 → expected GP per defender per
     combination.
  3. Sol picks max(0, best) — pass if no positive-EV bid exists.
  4. PLAY OUT the chosen bid with both sides using PIMC32, score
     the terminal via the minigame GP table.
  5. Report rich diagnostics.
"""
from __future__ import annotations

import random, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from _lib import deal_12_10_10, eval_one_deal, best_record
from ulti.solvers import pis, pimc as _pimc, determinize as _det
from ulti.scoring.oracle import BidSet, score as score_oracle

N         = 300
PIMC_N    = 32
SEED_BASE = 100_000
N_WORKERS = 8


def _pimc_pick(*, pos, contract, n_samples, seed, voids_dict):
    chosen, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, voids=voids_dict,
    )
    viewer = pis.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


def _play_out(*, remaining_10, def1, def2, talon, contract, trump,
              pimc_n, seed):
    """Play the chosen bid all the way through with PIMC on both sides.
    Returns the final GameState."""
    pos = pis.build_position(
        hands=[remaining_10, def1, def2], soloist=0, leader=0,
        contract=contract, trump=trump, talon=talon,
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)
    move_i = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        chosen = _pimc_pick(
            pos=pos, contract=contract, n_samples=pimc_n,
            seed=seed * 31337 + move_i, voids_dict=voids_dict,
        )
        if chosen is None:
            chosen = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis.apply_move(pos, chosen)
        move_i += 1
    return pos


def _score_terminal(pos, *, contract, piros):
    """Score the played-out terminal via the central scoring oracle.
    Returns GP per defender (signed, sol's perspective).

    silents=False: minigame deliberately ignores silent bonuses.
    score_parti=True only when the bid IS parti — for ulti bids the
    minigame focuses on ulti payoff only (no auto-parti bundling yet;
    that comes when we add combined contracts).
    """
    bid = BidSet(
        parti     = (contract == 'parti'),
        ulti      = (contract == 'ulti'),
        durchmars = (contract == 'durchmars'),
        betli     = (contract == 'betli'),
        piros     = piros,
    )
    return score_oracle(
        final_pos=pos, bid=bid,
        score_parti=(contract == 'parti'),
        silents=False,
    ).total_per_def


def worker(seed):
    sol12, d1, d2 = deal_12_10_10(seed)
    t0 = time.perf_counter()
    recs = eval_one_deal(sol12, d1, d2, pimc_n=PIMC_N, seed=seed * 17)
    eval_wall = time.perf_counter() - t0
    best = best_record(recs)

    if best is None:
        return {
            'seed': seed, 'pass': True, 'predicted_ev': 0.0,
            'actual_gp': 0.0, 'contract': None, 'trump': None,
            'discard': None, 'eval_wall': eval_wall, 'play_wall': 0.0,
        }

    discard_pair, contract, trump, p_make, predicted_ev = best
    piros = (trump == 'hearts')
    remaining = [c for c in sol12 if c not in discard_pair]
    talon = list(discard_pair)

    t1 = time.perf_counter()
    final_pos = _play_out(
        remaining_10=remaining, def1=d1, def2=d2, talon=talon,
        contract=contract, trump=trump, pimc_n=PIMC_N, seed=seed * 919,
    )
    play_wall = time.perf_counter() - t1
    actual_gp = _score_terminal(final_pos, contract=contract, piros=piros)

    return {
        'seed': seed, 'pass': False, 'predicted_ev': predicted_ev,
        'actual_gp': actual_gp, 'contract': contract, 'trump': trump,
        'discard': tuple(str(c) for c in discard_pair),
        'p_make': p_make, 'eval_wall': eval_wall, 'play_wall': play_wall,
    }


def main():
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(worker, seeds):
            rows.append(r)
            if len(rows) % 5 == 0:
                wall = time.perf_counter() - t0
                print(f"  {len(rows)}/{N}  wall={wall:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    n = len(rows)
    n_pass = sum(1 for r in rows if r['pass'])
    n_bid = n - n_pass

    print()
    print(f"=== Summary (N={n}, PIMC{PIMC_N}, {N_WORKERS} workers) ===")
    print(f"  pass rate:                  {n_pass}/{n} ({n_pass/n*100:.1f}%)")
    print(f"  bid rate:                   {n_bid}/{n} ({n_bid/n*100:.1f}%)")
    if n_bid > 0:
        bid_rows = [r for r in rows if not r['pass']]
        pred_mean = sum(r['predicted_ev'] for r in bid_rows) / n_bid
        actual_mean = sum(r['actual_gp'] for r in bid_rows) / n_bid
        print(f"  mean predicted EV (bids):   {pred_mean:+.3f}")
        print(f"  mean actual GP (bids):      {actual_mean:+.3f}")
        print(f"  calibration delta:          {actual_mean - pred_mean:+.3f}")
    overall_gp = sum(r['actual_gp'] for r in rows) / n
    print(f"  mean GP per deal (incl pass): {overall_gp:+.3f}")
    print(f"  total sol GP (×2):          {overall_gp*2*n:+.1f}")

    # Bid distribution
    print()
    print("=== Bid distribution ===")
    from collections import Counter
    bid_counter = Counter()
    for r in rows:
        if r['pass']:
            bid_counter['pass'] += 1
        else:
            key = f"{r['contract']}/{r['trump'] or 'colorless'}"
            bid_counter[key] += 1
    for key, count in sorted(bid_counter.items(), key=lambda x: -x[1]):
        print(f"  {key:>20}: {count:>3}  ({count/n*100:>5.1f}%)")

    # Per-contract calibration & success rate
    print()
    print("=== Per-contract performance ===")
    print(f"  {'contract':>20}  {'n':>4}  {'pred EV':>8}  {'actual':>7}  {'won':>4}  {'won %':>7}")
    by_c = {}
    for r in rows:
        if r['pass']:
            continue
        k = f"{r['contract']}/{r['trump'] or 'colorless'}"
        by_c.setdefault(k, []).append(r)
    for k in sorted(by_c, key=lambda x: -len(by_c[x])):
        sub = by_c[k]
        pe = sum(r['predicted_ev'] for r in sub) / len(sub)
        ag = sum(r['actual_gp'] for r in sub) / len(sub)
        won = sum(1 for r in sub if r['actual_gp'] > 0)
        print(f"  {k:>20}  {len(sub):>4}  {pe:>+7.2f}  {ag:>+6.2f}  "
              f"{won:>4}  {won/len(sub)*100:>6.1f}%")

    # Calibration scatter (predicted vs actual)
    print()
    print("=== Per-deal calibration (sorted by predicted EV) ===")
    print(f"  {'seed':>10}  {'contract':>20}  {'pred':>7}  {'p_make':>7}  {'actual':>7}")
    bid_rows = [r for r in rows if not r['pass']]
    for r in sorted(bid_rows, key=lambda x: -x['predicted_ev']):
        c = f"{r['contract']}/{r['trump'] or 'colorless'}"
        print(f"  {r['seed']:>10}  {c:>20}  {r['predicted_ev']:>+6.2f}  "
              f"{r.get('p_make',0):>6.2f}  {r['actual_gp']:>+6.2f}")

    # Timing
    eval_mean = sum(r['eval_wall'] for r in rows) / n
    play_mean = sum(r['play_wall'] for r in rows) / n
    print()
    print(f"=== Timing ===")
    print(f"  wall total:                 {wall:.0f}s")
    print(f"  mean eval / deal:           {eval_mean:.1f}s")
    print(f"  mean play / deal:           {play_mean:.1f}s")


if __name__ == "__main__":
    main()
