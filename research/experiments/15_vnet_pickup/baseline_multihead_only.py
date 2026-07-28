"""Multi-head pickup baseline (N=300, sol=PIMC32, def=god).

Mirrors baseline_v2_only.py but loads MultiHeadPickupNet weights and
routes each contract through its own head.
"""
from __future__ import annotations

import itertools, random, sys, time
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))
sys.path.insert(0, str(Path(__file__).parent))

from _lib import deal_12_10_10, _ev_per_def
from solvers import pis, pimc as _pimc, determinize as _det
from eval.pimc_matchup import god_pick
from ulti.card import SUITS
from _vlib import CONTRACT_CONFIGS, featurize, EXP_DIR
from vnet.pickup import MultiHeadPickupNet, pad_to_unified, UNIFIED_DIM
from scoring.oracle import BidSet, score as score_oracle

N         = 300
PIMC_N    = 32
SEED_BASE = 100_000
N_WORKERS = 8


def _weights_path() -> Path:
    return EXP_DIR / "multihead_v1.pt"


_NET = None
def _load_net():
    global _NET
    if _NET is not None:
        return _NET
    m = MultiHeadPickupNet(contracts=list(CONTRACT_CONFIGS))
    m.load_state_dict(torch.load(_weights_path(), weights_only=True))
    m.eval()
    _NET = m
    return m


def _pad(X, has_trump):
    return pad_to_unified(X, has_trump)


def _pick(sol12, *, seed):
    net = _load_net()
    discards = list(itertools.combinations(sol12, 2))
    best = None

    for cname, cfg in CONTRACT_CONFIGS.items():
        if cfg.has_trump:
            for trump in SUITS:
                X = np.stack([
                    featurize([c for c in sol12 if c not in dp], trump, True)
                    for dp in discards
                ])
                with torch.no_grad():
                    ps = net(torch.from_numpy(_pad(X, True)), cname).numpy()
                piros = (trump == 'hearts')
                for i, dp in enumerate(discards):
                    if cname == 'ulti':
                        rem = [c for c in sol12 if c not in dp]
                        if not any(c.suit == trump and c.rank == '7' for c in rem):
                            continue
                    p = float(ps[i])
                    ev = _ev_per_def(cname, piros, p)
                    if best is None or ev > best[0]:
                        best = (ev, dp, cname, trump, p)
        else:
            X = np.stack([
                featurize([c for c in sol12 if c not in dp], None, False)
                for dp in discards
            ])
            with torch.no_grad():
                ps = net(torch.from_numpy(_pad(X, False)), cname).numpy()
            for i, dp in enumerate(discards):
                p = float(ps[i])
                ev = _ev_per_def(cname, False, p)
                if best is None or ev > best[0]:
                    best = (ev, dp, cname, None, p)

    if best is None or best[0] < 0:
        return None
    return best


def _play_pimc_vs_god(*, sol10, d1, d2, talon, contract, trump, pimc_n, seed):
    pos = pis.build_position(
        hands=[sol10, d1, d2], soloist=0, leader=0,
        contract=contract, trump=trump, talon=talon,
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)
    move_i = 0
    while not pis.is_terminal(pos):
        p = pis.current_player(pos)
        if p == 0:
            chosen, _ = _pimc.pimc_decision(
                true_pos=pos, contract=contract, n_samples=pimc_n,
                seed=seed * 31337 + move_i, voids=voids_dict,
            )
        else:
            chosen = god_pick(pos=pos, contract=contract)
        if chosen is None:
            chosen = rng.choice(pis.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis.apply_move(pos, chosen)
        move_i += 1
    return pos


def _score(pos, *, contract, piros):
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
    pick = _pick(sol12, seed=seed)
    out = {'seed': seed}
    if pick is None:
        out.update({'pass': True, 'contract': None, 'trump': None,
                    'pred_ev': 0.0, 'actual_gp': 0.0, 'p': 0.0})
        return out
    ev, discard, contract, trump, p = pick
    piros = (trump == 'hearts')
    remaining = [c for c in sol12 if c not in discard]
    talon = list(discard)
    final = _play_pimc_vs_god(
        sol10=remaining, d1=d1, d2=d2, talon=talon,
        contract=contract, trump=trump, pimc_n=PIMC_N, seed=seed * 919,
    )
    actual = _score(final, contract=contract, piros=piros)
    out.update({'pass': False, 'contract': contract, 'trump': trump,
                'pred_ev': ev, 'actual_gp': actual, 'p': p})
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    seeds = [SEED_BASE + i for i in range(N)]
    t0 = time.perf_counter()
    rows = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(worker, seeds):
            rows.append(r)
            if len(rows) % 25 == 0:
                wall = time.perf_counter() - t0
                print(f"  {len(rows)}/{N}  wall={wall:.0f}s", flush=True)
    wall = time.perf_counter() - t0

    print()
    print(f"=== Multi-head v1 pickup (N={N}, sol=PIMC32, def=god) ===")
    n_pass = sum(1 for r in rows if r['pass'])
    bid_rows = [r for r in rows if not r['pass']]
    pe = _mean([r['pred_ev'] for r in bid_rows])
    ag = _mean([r['actual_gp'] for r in bid_rows])
    overall = sum(r['actual_gp'] for r in rows) / N
    print(f"  pass={n_pass}/{N} ({n_pass/N*100:.1f}%)  "
          f"pred_ev_bids={pe:+.3f}  actual_bids={ag:+.3f}  "
          f"GP/deal={overall:+.3f}  sol_total={overall*2*N:+.1f}")

    print()
    print("=== Multi-head per-contract performance ===")
    bd = Counter()
    for r in rows:
        if r['pass']:
            bd['pass'] += 1
        else:
            bd[f"{r['contract']}/{r['trump'] or 'colorless'}"] += 1
    print(f"  {'bid':>22}  {'n':>4}  {'pred EV':>8}  {'actual':>8}  "
          f"{'won':>4}  {'won %':>7}")
    for k in sorted(bd, key=lambda x: -bd[x]):
        if k == 'pass':
            print(f"  {'pass':>22}  {bd[k]:>4}")
            continue
        sub = [r for r in rows if not r['pass']
               and f"{r['contract']}/{r['trump'] or 'colorless'}" == k]
        pem = _mean([r['pred_ev'] for r in sub])
        agm = _mean([r['actual_gp'] for r in sub])
        won = sum(1 for r in sub if r['actual_gp'] > 0)
        print(f"  {k:>22}  {len(sub):>4}  {pem:>+7.2f}  {agm:>+7.2f}  "
              f"{won:>4}  {won/len(sub)*100:>6.1f}%")

    out_path = EXP_DIR / "baseline_multihead_picks.csv"
    with open(out_path, 'w') as f:
        f.write("seed,pass,contract,trump,pred_ev,actual_gp,p\n")
        for r in rows:
            f.write(f"{r['seed']},{int(r['pass'])},"
                    f"{r['contract'] or ''},{r['trump'] or ''},"
                    f"{r['pred_ev']:.4f},{r['actual_gp']:.4f},{r['p']:.4f}\n")
    print(f"\nPer-deal picks → {out_path}")
    print(f"Total wall: {wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
