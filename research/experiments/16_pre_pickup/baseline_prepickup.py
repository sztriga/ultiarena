"""Pickup-vs-pass measurement.

Per deal:
  - Post-pickup baseline (exp 15 v2 + aggressive thresholds on 12 cards)
      → GP_post
  - Pre-pickup variant:
      sol10 = sol12[:10]; talon = sol12[10:]
      oracle on sol10 → pickup yes/no
      if yes: GP_pre = GP_post (same post-pickup logic applied)
      if no:  GP_pre = 0

Aggregate at N=3000.
"""
from __future__ import annotations

import itertools, random, sys, time, os
from multiprocessing import Pool
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "15_vnet_pickup"))
sys.path.insert(0, str(Path(__file__).parent))

from _lib import deal_12_10_10, _ev_per_def
from solvers import pis, pimc as _pimc, determinize as _det
from eval.pimc_matchup import god_pick
from ulti.card import SUITS
from vnet.pickup import CONTRACT_CONFIGS, PickupNetV2, featurize, input_dim
from scoring.oracle import BidSet, score as score_oracle
import pre_pickup_oracle as oracle

EXP15_DIR = Path(__file__).parent.parent / "15_vnet_pickup"

N         = int(os.environ.get("N_DEALS", 300))
PIMC_N    = 32
SEED_BASE = 100_000
N_WORKERS = 8

# Default: same aggressive thresholds we found in exp 15 (N=3000).
# Override via env: THRESH="betli:0,parti:0,ulti:0,durchmars:0" for raw.
def _parse_thresh(s: str) -> dict:
    default = {'betli': 4.0, 'parti': 0.5, 'ulti': 1.5, 'durchmars': 8.0}
    if not s:
        return default
    out = dict(default)
    for kv in s.split(','):
        k, v = kv.split(':')
        out[k.strip()] = float(v)
    return out

THRESH = _parse_thresh(os.environ.get("THRESH", ""))


def _v2_weights(name: str) -> Path:
    return EXP15_DIR / f"{name}_vnet_v2.pt"


_NETS = None
def _load_nets():
    global _NETS
    if _NETS is not None:
        return _NETS
    nets = {}
    for name, cfg in CONTRACT_CONFIGS.items():
        m = PickupNetV2(in_dim=input_dim(cfg))
        m.load_state_dict(torch.load(_v2_weights(name), weights_only=True))
        m.eval()
        nets[name] = m
    _NETS = nets
    return nets


def _post_pickup_pick(sol12):
    """exp 15 v2 + aggressive thresholds. Returns (ev,dp,cname,trump,p) or None."""
    nets = _load_nets()
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
                    ps = nets[cname](torch.from_numpy(X)).numpy()
                piros = (trump == 'hearts')
                for i, dp in enumerate(discards):
                    if cname == 'ulti':
                        rem = [c for c in sol12 if c not in dp]
                        if not any(c.suit == trump and c.rank == '7' for c in rem):
                            continue
                    p = float(ps[i])
                    ev = _ev_per_def(cname, piros, p)
                    if ev >= THRESH[cname] and (best is None or ev > best[0]):
                        best = (ev, dp, cname, trump, p)
        else:
            X = np.stack([
                featurize([c for c in sol12 if c not in dp], None, False)
                for dp in discards
            ])
            with torch.no_grad():
                ps = nets[cname](torch.from_numpy(X)).numpy()
            for i, dp in enumerate(discards):
                p = float(ps[i])
                ev = _ev_per_def(cname, False, p)
                if ev >= THRESH[cname] and (best is None or ev > best[0]):
                    best = (ev, dp, cname, None, p)
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
    sol10 = sol12[:10]
    # Post-pickup picker (always picks up)
    pick = _post_pickup_pick(sol12)
    if pick is None:
        gp_post = 0.0
        post_contract = post_trump = None
    else:
        ev, discard, cname, trump, p = pick
        piros = (trump == 'hearts')
        rem = [c for c in sol12 if c not in discard]
        talon = list(discard)
        final = _play_pimc_vs_god(
            sol10=rem, d1=d1, d2=d2, talon=talon,
            contract=cname, trump=trump, pimc_n=PIMC_N, seed=seed * 919,
        )
        gp_post = _score(final, contract=cname, piros=piros)
        post_contract, post_trump = cname, trump

    # Pre-pickup oracle on sol10 only
    oracle_pick = oracle.pick(sol10, thresh=THRESH)
    pickup = oracle_pick is not None
    gp_pre = gp_post if pickup else 0.0

    return {
        'seed': seed,
        'gp_post': gp_post,
        'gp_pre':  gp_pre,
        'post_contract': post_contract,
        'post_trump':    post_trump,
        'pickup':        pickup,
        'oracle_contract': oracle_pick['contract'] if pickup else None,
        'oracle_trump':    oracle_pick['trump']    if pickup else None,
        'oracle_ev':       oracle_pick['mean_ev']  if pickup else 0.0,
    }


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

    n_post_bid = sum(1 for r in rows if r['post_contract'] is not None)
    n_pre_bid  = sum(1 for r in rows if r['pickup'])
    gp_post = sum(r['gp_post'] for r in rows) / N
    gp_pre  = sum(r['gp_pre']  for r in rows) / N

    # 2x2 confusion: oracle pickup × post-pickup picks up
    cm = {(True,True):0,(True,False):0,(False,True):0,(False,False):0}
    for r in rows:
        cm[(r['pickup'], r['post_contract'] is not None)] += 1

    print()
    tdesc = ",".join(f"{k}={v}" for k, v in THRESH.items())
    print(f"=== Pre-pickup measurement (N={N}, sol=PIMC32, def=god) ===")
    print(f"  thresholds: {tdesc}")
    print(f"  post-pickup baseline: {n_post_bid}/{N} bids ({n_post_bid/N*100:.1f}%)  "
          f"GP/deal={gp_post:+.3f}  sol_total={gp_post*2*N:+.1f}")
    print(f"  pre-pickup oracle  : {n_pre_bid}/{N} pickups ({n_pre_bid/N*100:.1f}%)  "
          f"GP/deal={gp_pre:+.3f}  sol_total={gp_pre*2*N:+.1f}")
    print(f"  Δ(pre - post) GP/deal: {gp_pre - gp_post:+.4f}")
    print()
    print("=== Pickup confusion (oracle row, post-pickup column) ===")
    print(f"                  post=bid    post=pass")
    print(f"  oracle=pickup   {cm[(True,True)]:>8}    {cm[(True,False)]:>8}")
    print(f"  oracle=pass     {cm[(False,True)]:>8}    {cm[(False,False)]:>8}")

    # Disagreement deals analysis
    fp = [r for r in rows if r['pickup'] and r['post_contract'] is None]
    fn = [r for r in rows if not r['pickup'] and r['post_contract'] is not None]
    print()
    print(f"  oracle pickup, post=pass ({len(fp)}): GP_pre would be 0 (oracle correct in hindsight? avg post EV: {_mean([r['gp_post'] for r in fp]):+.2f})")
    print(f"  oracle pass, post=bid  ({len(fn)}): missed real bids. avg GP_post={_mean([r['gp_post'] for r in fn]):+.2f}  total missed={sum(r['gp_post'] for r in fn):+.1f}")

    print()
    print(f"Total wall: {wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
