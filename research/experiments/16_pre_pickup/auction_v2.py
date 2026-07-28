"""Calibrated, threshold-free auction.

Clean decision rules — no per-contract calibration thresholds:
  - P0 opener: bid the best contract above rank 0 (excl. non-piros parti)
    IF its EV > -2.  Else pay -2/def penalty.
  - Overtaker: oracle pickup-EV vs PIMC pass-EV.  Pickup if oracle > pass.
  - Overtaker commit: MUST bid the highest-EV (contract, trump, discard)
    above current rank.  No fallback to pass.

Uses CalibratedPickupNet (v2 weights + isotonic correction per contract).
"""
from __future__ import annotations

import itertools, random, sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "15_vnet_pickup"))
sys.path.insert(0, str(Path(__file__).parent))

from _lib import deal_12_10_10, _ev_per_def
from solvers import pis, pimc as _pimc, determinize as _det
from eval.pimc_matchup import god_pick
from ulti.card import SUITS, DECK, Card
from vnet.pickup import CONTRACT_CONFIGS, featurize, input_dim
from vnet.pickup.calibration import CalibratedPickupNet
from scoring.oracle import BidSet, score as score_oracle
import itertools as _it

import os
EXP15_DIR = Path(__file__).parent.parent / "15_vnet_pickup"
PIMC_N = 32
# Game rule: per-defender GP P0 forfeits if they decline to bid.
PASS_PENALTY = float(os.environ.get("PASS_PENALTY", "-2.0"))
# Decision rule: P0 bids only if best contract EV > BID_FLOOR.
# Default = PASS_PENALTY (no buffer); raise above PASS_PENALTY to add a
# calibration-safety buffer (avoid bidding shaky predictions).
BID_FLOOR = float(os.environ.get("BID_FLOOR", str(PASS_PENALTY)))


def contract_rank(contract: str, trump: Optional[str]) -> int:
    piros = (trump == 'hearts')
    if contract == 'parti':     return 2 if piros else 1
    if contract == 'ulti':      return 6 if piros else 3
    if contract == 'betli':     return 4
    if contract == 'durchmars': return 5
    raise ValueError(contract)


def _is_legal_bid(contract: str, trump: Optional[str]) -> bool:
    if contract == 'parti' and trump != 'hearts':
        return False
    return True


_CNET = None
def _load_cnet():
    global _CNET
    if _CNET is None:
        _CNET = CalibratedPickupNet.load(EXP15_DIR)
    return _CNET


# ─── Pickup commit (post-talon-reveal): find best (discard, contract, trump) ──
def _best_bid_above_rank(hand12, *, min_rank: int,
                         ev_floor: Optional[float] = None):
    """Return (ev, discard, cname, trump, p) for the best bid above rank.
    If ev_floor is set, return None if best EV < ev_floor.
    """
    cnet = _load_cnet()
    discards = list(itertools.combinations(hand12, 2))
    best = None
    for cname, cfg in CONTRACT_CONFIGS.items():
        trumps = SUITS if cfg.has_trump else [None]
        for trump in trumps:
            if not _is_legal_bid(cname, trump):
                continue
            rank = contract_rank(cname, trump)
            if rank <= min_rank:
                continue
            X = np.stack([
                featurize([c for c in hand12 if c not in dp],
                          trump, cfg.has_trump) for dp in discards
            ])
            ps = cnet.predict(X, cname)
            piros = (trump == 'hearts')
            for i, dp in enumerate(discards):
                if cname == 'ulti':
                    rem = [c for c in hand12 if c not in dp]
                    if not any(c.suit == trump and c.rank == '7' for c in rem):
                        continue
                p = float(ps[i])
                ev = _ev_per_def(cname, piros, p)
                if best is None or ev > best[0]:
                    best = (ev, dp, cname, trump, p)
    if best is None:
        return None
    if ev_floor is not None and best[0] < ev_floor:
        return None
    return best


# ─── Pre-pickup oracle (calibrated, enumerated over 231 talons) ──
def _oracle_evaluate(sol_hand_10, *, min_rank: int = 0):
    """Return best (mean_ev, contract, trump, mean_p) above min_rank.
    Excludes non-piros parti."""
    cnet = _load_cnet()
    sol_set = set(sol_hand_10)
    unseen = [c for c in DECK if c not in sol_set]
    assert len(unseen) == 22
    talons = list(_it.combinations(unseen, 2))
    n_talons = len(talons)
    # Build all (231 × 66) final hands once; reuse across contracts.
    flat_finals = []
    for talon in talons:
        hand12 = sol_hand_10 + list(talon)
        for discard in _it.combinations(range(12), 2):
            final10 = [hand12[i] for i in range(12) if i not in discard]
            flat_finals.append(final10)

    best = None
    for cname, cfg in CONTRACT_CONFIGS.items():
        trumps = SUITS if cfg.has_trump else [None]
        for trump in trumps:
            if not _is_legal_bid(cname, trump):
                continue
            if contract_rank(cname, trump) <= min_rank:
                continue
            X = np.stack([
                featurize(h, trump, cfg.has_trump) for h in flat_finals
            ])
            ps = cnet.predict(X, cname)
            ps_mat = ps.reshape(n_talons, 66)
            best_per_talon = ps_mat.max(axis=1)
            mean_p = float(best_per_talon.mean())
            piros = (trump == 'hearts')
            mean_ev = _ev_per_def(cname, piros, mean_p)
            if best is None or mean_ev > best['mean_ev']:
                best = {
                    'contract': cname, 'trump': trump,
                    'mean_ev': mean_ev, 'mean_p': mean_p,
                    'rank': contract_rank(cname, trump),
                }
    return best


# ─── Pass-EV via PIMC32 ───────────────────────────────────────────
def _sol_p_make(sol_hand_play, def1, def2, *, contract, trump, talon, seed):
    pos = pis.build_position(
        hands=[sol_hand_play, def1, def2], soloist=0, leader=0,
        contract=contract, trump=trump, talon=talon,
    )
    _, avg = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=PIMC_N, seed=seed,
    )
    return max(0.0, min(1.0, max(avg.values()))) if avg else 0.0


# ─── Play layer ───────────────────────────────────────────────────
def _play_pimc_vs_god(*, sol10, d1, d2, talon, contract, trump, seed):
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
                true_pos=pos, contract=contract, n_samples=PIMC_N,
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
        parti=(contract == 'parti'), ulti=(contract == 'ulti'),
        durchmars=(contract == 'durchmars'), betli=(contract == 'betli'),
        piros=piros,
    )
    return score_oracle(
        final_pos=pos, bid=bid,
        score_parti=(contract == 'parti'), silents=False,
    ).total_per_def


# ─── Main auction ─────────────────────────────────────────────────
def simulate(seed: int) -> dict:
    sol12, d1_10, d2_10 = deal_12_10_10(seed)
    sol10_orig = sol12[:10]
    talon_orig = sol12[10:]
    hands = [sol10_orig.copy(), list(d1_10), list(d2_10)]
    talon = list(talon_orig)
    log = []

    # ── Round 0: P0 picks up. Bid if EV > -2, else penalty.
    p0_hand12 = hands[0] + talon
    pick = _best_bid_above_rank(p0_hand12, min_rank=0, ev_floor=BID_FLOOR)
    if pick is None:
        gps = [0, 0, 0]
        gps[0] = 2 * PASS_PENALTY
        gps[1] = -PASS_PENALTY
        gps[2] = -PASS_PENALTY
        return {
            'seed': seed, 'winner_pid': None,
            'winning_bid': 'PASS_PENALTY',
            'n_pickups': 1, 'log': [{
                'pid': 0, 'action': 'open_pass_penalty',
                'penalty_per_def': PASS_PENALTY,
            }],
            'gp_per_def': PASS_PENALTY, 'gps': gps,
        }
    ev, discard, cname, trump, p_v = pick
    hands[0] = [c for c in p0_hand12 if c not in discard]
    talon = list(discard)
    current = {
        'pid': 0, 'contract': cname, 'trump': trump,
        'rank': contract_rank(cname, trump), 'p_vnet': p_v, 'ev': ev,
    }
    log.append({
        'pid': 0, 'action': 'open_bid',
        'bid': f"{cname}/{trump or 'colorless'}",
        'rank': current['rank'], 'p': p_v, 'ev': ev,
    })

    passes = 0
    next_pid = 1
    n_pickups = 1
    while passes < 3:
        if next_pid == current['pid']:
            log.append({'pid': next_pid, 'action': 'auto_pass'})
            passes += 1
        else:
            cand = _oracle_evaluate(hands[next_pid], min_rank=current['rank'])
            if cand is None:
                log.append({'pid': next_pid, 'action': 'pass',
                            'reason': 'no candidate beats rank'})
                passes += 1
            else:
                p_make = _sol_p_make(
                    hands[current['pid']], hands[(current['pid']+1)%3],
                    hands[(current['pid']+2)%3],
                    contract=current['contract'], trump=current['trump'],
                    talon=talon,
                    seed=seed * 131 + next_pid * 17,
                )
                pass_ev = -_ev_per_def(current['contract'],
                                       current['trump'] == 'hearts',
                                       p_make)
                if cand['mean_ev'] <= pass_ev:
                    log.append({
                        'pid': next_pid, 'action': 'pass',
                        'reason': 'pickup ev < pass ev',
                        'pickup_ev': cand['mean_ev'], 'pass_ev': pass_ev,
                    })
                    passes += 1
                else:
                    new_hand12 = hands[next_pid] + talon
                    other = [p for p in range(3) if p != next_pid]
                    # Commit: MUST bid above rank (no ev_floor)
                    pick = _best_bid_above_rank(
                        new_hand12, min_rank=current['rank'])
                    if pick is None:
                        log.append({'pid': next_pid, 'action': 'pass',
                                    'reason': 'no legal bid above rank'})
                        passes += 1
                    else:
                        ev, discard, cname2, trump2, p_v2 = pick
                        hands[next_pid] = [c for c in new_hand12 if c not in discard]
                        talon = list(discard)
                        current = {
                            'pid': next_pid, 'contract': cname2, 'trump': trump2,
                            'rank': contract_rank(cname2, trump2),
                            'p_vnet': p_v2, 'ev': ev,
                        }
                        log.append({
                            'pid': next_pid, 'action': 'overtake',
                            'pickup_ev_oracle': cand['mean_ev'],
                            'pass_ev': pass_ev,
                            'bid': f"{cname2}/{trump2 or 'colorless'}",
                            'rank': current['rank'], 'p': p_v2, 'ev': ev,
                        })
                        passes = 0
                        n_pickups += 1
        next_pid = (next_pid + 1) % 3

    winner_pid = current['pid']
    sol_hand = hands[winner_pid]
    def1 = hands[(winner_pid + 1) % 3]
    def2 = hands[(winner_pid + 2) % 3]
    final = _play_pimc_vs_god(
        sol10=sol_hand, d1=def1, d2=def2, talon=talon,
        contract=current['contract'], trump=current['trump'],
        seed=seed * 919,
    )
    gp_per_def = _score(final, contract=current['contract'],
                        piros=(current['trump'] == 'hearts'))
    gps = [0, 0, 0]
    gps[winner_pid] = 2 * gp_per_def
    for pi in range(3):
        if pi != winner_pid:
            gps[pi] = -gp_per_def
    return {
        'seed': seed, 'winner_pid': winner_pid,
        'winning_bid': f"{current['contract']}/{current['trump'] or 'colorless'}",
        'n_pickups': n_pickups, 'log': log,
        'gp_per_def': gp_per_def, 'gps': gps,
    }


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    res = simulate(100000)
    print(f"=== seed 100000 (wall={time.perf_counter()-t0:.2f}s) ===")
    print(f"  winner: P{res['winner_pid']}  bid: {res['winning_bid']}")
    print(f"  gp_per_def: {res['gp_per_def']:+.1f}  gps: {res['gps']}")
    for e in res['log']:
        print(f"  {e}")
