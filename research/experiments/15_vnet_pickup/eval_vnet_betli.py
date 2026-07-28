"""Compare v-net vs PIMC32 for the betli pickup decision.

Replicates the minigame eval (`experiments/14_minigame_bid_eval/run_minigame.py`)
on the same N=20 seeds, but the betli contract's pickup is decided by
the trained v-net instead of PIMC32. Other contracts (parti, ulti,
durchmars) still use PIMC32.

Reports: wall time for the betli leg only, bid distribution shift,
actual GP, and any deal where the chosen bid changed.
"""
from __future__ import annotations

import itertools, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "12_contract_oracle"))
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))
sys.path.insert(0, str(Path(__file__).parent))

from _lib import deal_12_10_10, _ev_per_def
from run_minigame import _play_out, _score_terminal
from solvers import pis, pimc as _pimc
from ulti.card import SUITS
from train_betli import BetliNet

N         = 20
PIMC_N    = 32
SEED_BASE = 100_000
N_WORKERS = 4
WEIGHTS   = Path(__file__).parent / "betli_vnet.pt"


def _hand_vec(hand) -> np.ndarray:
    v = np.zeros(32, dtype=np.float32)
    for c in hand:
        v[c.id] = 1.0
    return v


def _load_vnet():
    m = BetliNet()
    m.load_state_dict(torch.load(WEIGHTS, weights_only=True))
    m.eval()
    return m


def eval_one_deal_vnet(sol12, d1, d2, *, pimc_n, seed, vnet):
    """Same eval as the PIMC minigame, except betli pickups use the
    v-net instead of PIMC32."""
    records = []
    discards = list(itertools.combinations(sol12, 2))
    rng_seed = seed

    # Batch all betli hand featurizations for a single torch call.
    betli_hands = []
    for discard_pair in discards:
        remaining = [c for c in sol12 if c not in discard_pair]
        betli_hands.append(_hand_vec(remaining))
    X_betli = torch.from_numpy(np.stack(betli_hands))
    with torch.no_grad():
        p_betli_all = vnet(X_betli).numpy()

    for i, discard_pair in enumerate(discards):
        remaining = [c for c in sol12 if c not in discard_pair]
        talon = list(discard_pair)

        for trump in SUITS:
            piros = (trump == 'hearts')
            for contract in ('parti', 'ulti'):
                if contract == 'ulti':
                    has_t7 = any(c.suit == trump and c.rank == '7'
                                 for c in remaining)
                    if not has_t7:
                        records.append((discard_pair, contract, trump,
                                        0.0, _ev_per_def(contract, piros, 0.0)))
                        continue
                pos = pis.build_position(
                    hands=[remaining, d1, d2], soloist=0, leader=0,
                    contract=contract, trump=trump, talon=talon,
                )
                rng_seed += 1
                _, avg = _pimc.pimc_decision(
                    true_pos=pos, contract=contract, n_samples=pimc_n,
                    seed=rng_seed,
                )
                p = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
                records.append((discard_pair, contract, trump, p,
                                _ev_per_def(contract, piros, p)))

        # durchmars: still PIMC
        pos = pis.build_position(
            hands=[remaining, d1, d2], soloist=0, leader=0,
            contract='durchmars', trump=None, talon=talon,
        )
        rng_seed += 1
        _, avg = _pimc.pimc_decision(
            true_pos=pos, contract='durchmars', n_samples=pimc_n, seed=rng_seed,
        )
        p_d = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
        records.append((discard_pair, 'durchmars', None, p_d,
                        _ev_per_def('durchmars', False, p_d)))

        # betli: v-net (single network call already batched above)
        p_b = float(p_betli_all[i])
        records.append((discard_pair, 'betli', None, p_b,
                        _ev_per_def('betli', False, p_b)))

    return records


def worker(seed):
    vnet = _load_vnet()
    sol12, d1, d2 = deal_12_10_10(seed)
    t0 = time.perf_counter()
    recs = eval_one_deal_vnet(sol12, d1, d2, pimc_n=PIMC_N, seed=seed * 17,
                               vnet=vnet)
    eval_wall = time.perf_counter() - t0

    best = max(recs, key=lambda r: r[4])
    if best[4] < 0:
        return {'seed': seed, 'pass': True, 'predicted_ev': 0.0,
                'actual_gp': 0.0, 'contract': None, 'trump': None,
                'eval_wall': eval_wall, 'play_wall': 0.0}
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
    return {'seed': seed, 'pass': False, 'predicted_ev': predicted_ev,
            'actual_gp': actual_gp, 'contract': contract, 'trump': trump,
            'p_make': p_make, 'eval_wall': eval_wall, 'play_wall': play_wall}


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

    n = len(rows); n_pass = sum(1 for r in rows if r['pass'])
    print()
    print(f"=== Summary (N={n}, v-net betli + PIMC{PIMC_N} others) ===")
    print(f"  pass rate:  {n_pass}/{n}")
    bid_rows = [r for r in rows if not r['pass']]
    if bid_rows:
        pe = sum(r['predicted_ev'] for r in bid_rows)/len(bid_rows)
        ag = sum(r['actual_gp'] for r in bid_rows)/len(bid_rows)
        print(f"  mean pred EV (bids): {pe:+.3f}")
        print(f"  mean actual GP (bids): {ag:+.3f}")
        print(f"  calibration delta: {ag-pe:+.3f}")
    overall = sum(r['actual_gp'] for r in rows)/n
    print(f"  mean GP/deal (incl pass): {overall:+.3f}")

    # Bid dist
    from collections import Counter
    bd = Counter()
    for r in rows:
        if r['pass']:
            bd['pass'] += 1
        else:
            bd[f"{r['contract']}/{r['trump'] or 'colorless'}"] += 1
    print()
    print("=== Bid distribution ===")
    for k, c in sorted(bd.items(), key=lambda x: -x[1]):
        print(f"  {k:>22}: {c:>3}  ({c/n*100:>5.1f}%)")

    em = sum(r['eval_wall'] for r in rows)/n
    pm = sum(r['play_wall'] for r in rows)/n
    print()
    print(f"=== Timing (s/deal) ===")
    print(f"  wall total: {wall:.0f}s")
    print(f"  mean eval / deal: {em:.2f}s")
    print(f"  mean play / deal: {pm:.2f}s")


if __name__ == "__main__":
    main()
