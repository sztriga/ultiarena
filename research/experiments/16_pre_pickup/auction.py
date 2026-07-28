"""Bidding auction — multi-round pickup loop.

Rules (per milan's variant):
  - P0 starts with talon already picked up (12 cards). Discards 2 and bids.
    Forced to bid something (no pass on the opener).
  - In turn order, other players see current bid. They decide:
      pass     → no action
      pickup   → take the current talon (becomes 12), discard 2 back to talon,
                 lock in a new bid that STRICTLY OUTRANKS current.
  - Auction ends after 3 consecutive passes. The last bidder plays.

Contract ranking (low → high):
    1  parti  (non-piros)
    2  piros parti  (parti/hearts)
    3  ulti   (non-piros)
    4  betli
    5  durchmars
    6  piros ulti  (ulti/hearts)

Pass-EV (defender perspective vs current bid):
    -PIMC32(sol's post-discard pos, contract).p_make → -EV_per_def

Overtake-EV (would-be soloist on overtaker's 10 cards):
    oracle(player_hand_10, threshold) → best bid that ranks > current
    (oracle averages over 231 talons — pessimistic vs actual P0-discards,
    but consistent v1 simplification).
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
from ulti.solvers import pis, pimc as _pimc, determinize as _det
from ulti.eval.pimc_matchup import god_pick
from ulti.card import SUITS, Card
from ulti.vnet.pickup import CONTRACT_CONFIGS, PickupNetV2, featurize, input_dim
from ulti.scoring.oracle import BidSet, score as score_oracle
import pre_pickup_oracle as oracle

EXP15_DIR = Path(__file__).parent.parent / "15_vnet_pickup"

import os
PIMC_N = 32

THRESH = {'betli': 4.0, 'parti': 0.5, 'ulti': 1.5, 'durchmars': 8.0}

# When True, betli and durchmars use PIMC32 (not v-net) for the
# post-pickup discard+contract evaluation. Pre-pickup oracle still uses
# v-net everywhere (PIMC over 231 talons is too slow).
HYBRID_BD = bool(int(os.environ.get("HYBRID_BD", "0")))

PIMC_HYBRID_CONTRACTS = {'betli', 'durchmars'} if HYBRID_BD else set()


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


def contract_rank(contract: str, trump: Optional[str]) -> int:
    piros = (trump == 'hearts')
    if contract == 'parti':     return 2 if piros else 1
    if contract == 'ulti':      return 6 if piros else 3
    if contract == 'betli':     return 4
    if contract == 'durchmars': return 5
    raise ValueError(contract)


# Non-piros parti (any trump other than hearts) is illegal in the auction.
# Players never bid these — if no other contract is viable, they pay the
# pass penalty instead.
def _is_legal_bid(contract: str, trump: Optional[str]) -> bool:
    if contract == 'parti' and trump != 'hearts':
        return False
    return True


# P0 forced-open escape hatch: if every legal bid is worse than this,
# pay PASS_PENALTY per defender (no play).
PASS_PENALTY = -2.0


# ─── Picker: full v-net + threshold over a 12-card hand ────────────
def _post_pickup_pick(hand12, *, d1=None, d2=None, seed=0,
                      force_bid: bool = False, min_rank: int = 0):
    """exp 15 v2 + aggressive thresholds. Returns (ev,dp,cname,trump,p) or None.

    If HYBRID_BD: betli & durchmars Ps come from PIMC32 (not v-net).
        Requires d1, d2 hands (used for PIMC determinization).
    """
    nets = _load_nets()
    discards = list(itertools.combinations(hand12, 2))
    best_with_thresh = None
    best_any = None
    seed_counter = [seed]

    def _ps_for_contract(cname, cfg, trump, has_trump):
        """Return per-discard P_make array (len=66 or fewer if filtered)."""
        if cname in PIMC_HYBRID_CONTRACTS:
            assert d1 is not None and d2 is not None, \
                "hybrid PIMC needs defender hands"
            ps = np.zeros(len(discards), dtype=np.float32)
            for i, dp in enumerate(discards):
                rem = [c for c in hand12 if c not in dp]
                talon = list(dp)
                pos = pis.build_position(
                    hands=[rem, d1, d2], soloist=0, leader=0,
                    contract=cname, trump=trump, talon=talon,
                )
                seed_counter[0] += 1
                _, avg = _pimc.pimc_decision(
                    true_pos=pos, contract=cname,
                    n_samples=PIMC_N, seed=seed_counter[0],
                )
                ps[i] = max(0.0, min(1.0, max(avg.values()))) if avg else 0.0
            return ps
        X = np.stack([
            featurize([c for c in hand12 if c not in dp], trump, has_trump)
            for dp in discards
        ])
        with torch.no_grad():
            return nets[cname](torch.from_numpy(X)).numpy()

    for cname, cfg in CONTRACT_CONFIGS.items():
        if cfg.has_trump:
            for trump in SUITS:
                if not _is_legal_bid(cname, trump):
                    continue
                rank = contract_rank(cname, trump)
                if rank <= min_rank:
                    continue
                ps = _ps_for_contract(cname, cfg, trump, True)
                piros = (trump == 'hearts')
                for i, dp in enumerate(discards):
                    if cname == 'ulti':
                        rem = [c for c in hand12 if c not in dp]
                        if not any(c.suit == trump and c.rank == '7' for c in rem):
                            continue
                    p = float(ps[i])
                    ev = _ev_per_def(cname, piros, p)
                    if best_any is None or ev > best_any[0]:
                        best_any = (ev, dp, cname, trump, p)
                    if ev >= THRESH[cname] and (best_with_thresh is None or
                                                 ev > best_with_thresh[0]):
                        best_with_thresh = (ev, dp, cname, trump, p)
        else:
            rank = contract_rank(cname, None)
            if rank <= min_rank:
                continue
            ps = _ps_for_contract(cname, cfg, None, False)
            for i, dp in enumerate(discards):
                p = float(ps[i])
                ev = _ev_per_def(cname, False, p)
                if best_any is None or ev > best_any[0]:
                    best_any = (ev, dp, cname, None, p)
                if ev >= THRESH[cname] and (best_with_thresh is None or
                                             ev > best_with_thresh[0]):
                    best_with_thresh = (ev, dp, cname, None, p)
    if best_with_thresh is not None:
        return best_with_thresh
    if force_bid:
        return best_any
    return None


# ─── Overtake decision ────────────────────────────────────────────
def _overtake_candidate(hand10, *, min_rank: int) -> Optional[dict]:
    """Oracle pick whose rank > min_rank, with mean_ev ≥ contract threshold.
    Returns the chosen contract dict or None.

    Non-piros parti is excluded.
    """
    stats = oracle.evaluate(hand10)
    best = None
    for (cname, trump), s in stats.items():
        if not _is_legal_bid(cname, trump):
            continue
        if contract_rank(cname, trump) <= min_rank:
            continue
        if s['mean_ev'] < THRESH[cname]:
            continue
        if best is None or s['mean_ev'] > best['mean_ev']:
            best = {
                'contract': cname,
                'trump':    trump,
                'mean_ev':  s['mean_ev'],
                'mean_p':   s['mean_best_p'],
                'rank':     contract_rank(cname, trump),
            }
    return best


# ─── Pass EV via PIMC32 on sol's actual playing position ───────────
def _sol_p_make(sol_hand_play, def1, def2, *, contract, trump, talon, seed):
    pos = pis.build_position(
        hands=[sol_hand_play, def1, def2], soloist=0, leader=0,
        contract=contract, trump=trump, talon=talon,
    )
    _, avg = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=PIMC_N, seed=seed,
    )
    return max(0.0, min(1.0, max(avg.values()))) if avg else 0.0


# ─── Play out the winning bid ───────────────────────────────────────
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


# ─── Main auction loop ──────────────────────────────────────────────
def simulate(seed: int) -> dict:
    sol12, d1_10, d2_10 = deal_12_10_10(seed)
    # Conceptual split: sol's original 10 vs original talon
    sol10_orig = sol12[:10]
    talon_orig = sol12[10:]  # 2 cards
    hands = [sol10_orig.copy(), list(d1_10), list(d2_10)]
    talon = list(talon_orig)

    log = []  # list of (turn, pid, action, details)

    # --- Round 0: P0 picks up ----
    p0_hand12 = hands[0] + talon
    # No force_bid: if no contract clears its threshold, pick is None,
    # P0 pays the pass penalty (instead of being forced into the
    # least-bad bid, which is usually parti/hearts on a weak hand).
    pick = _post_pickup_pick(p0_hand12, d1=hands[1], d2=hands[2],
                             seed=seed * 41, force_bid=False)
    if pick is None or pick[0] < PASS_PENALTY:
        gps = [0, 0, 0]
        gps[0] = 2 * PASS_PENALTY      # sol pays 2x (one per defender)
        gps[1] = -PASS_PENALTY
        gps[2] = -PASS_PENALTY
        return {
            'seed': seed, 'winner_pid': None,
            'winning_bid': 'PASS_PENALTY',
            'n_pickups': 1, 'log': [{
                'pid': 0, 'action': 'open_pass_penalty',
                'best_force_ev': (pick[0] if pick else None),
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

    # --- Auction loop ----
    passes = 0
    next_pid = 1
    n_pickups = 1   # P0's initial counts as 1
    while passes < 3:
        if next_pid == current['pid']:
            # current bidder cannot overtake themselves
            log.append({'pid': next_pid, 'action': 'auto_pass'})
            passes += 1
        else:
            cand = _overtake_candidate(hands[next_pid],
                                       min_rank=current['rank'])
            if cand is None:
                # Estimate pass EV as defender (sanity diagnostic)
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
                log.append({
                    'pid': next_pid, 'action': 'pass',
                    'reason': 'no candidate beats rank',
                    'pass_ev_diag': pass_ev,
                })
                passes += 1
            else:
                # decide pickup vs pass
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
                    # Pickup: take talon (becomes 12), discard 2, lock bid
                    new_hand12 = hands[next_pid] + talon
                    other_pids = [p for p in range(3) if p != next_pid]
                    pick = _post_pickup_pick(
                        new_hand12,
                        d1=hands[other_pids[0]], d2=hands[other_pids[1]],
                        seed=seed * 41 + next_pid * 7,
                        force_bid=True, min_rank=current['rank'],
                    )
                    if pick is None:
                        log.append({
                            'pid': next_pid, 'action': 'pass',
                            'reason': 'no pickup outranks current after talon',
                        })
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
                            'pass_ev_diag': pass_ev,
                            'bid': f"{cname2}/{trump2 or 'colorless'}",
                            'rank': current['rank'], 'p': p_v2, 'ev': ev,
                        })
                        passes = 0
                        n_pickups += 1
        next_pid = (next_pid + 1) % 3

    # --- Play out the winning bid ----
    winner_pid = current['pid']
    sol_hand = hands[winner_pid]
    # Reorder hands for play: soloist at index 0
    def1 = hands[(winner_pid + 1) % 3]
    def2 = hands[(winner_pid + 2) % 3]
    final = _play_pimc_vs_god(
        sol10=sol_hand, d1=def1, d2=def2, talon=talon,
        contract=current['contract'], trump=current['trump'],
        seed=seed * 919,
    )
    gp_per_def = _score(final, contract=current['contract'],
                        piros=(current['trump'] == 'hearts'))
    # GP per player
    gps = [0, 0, 0]
    gps[winner_pid] = 2 * gp_per_def       # sol gets 2x
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
    wall = time.perf_counter() - t0
    print(f"=== Auction seed 100000 (wall={wall:.2f}s) ===")
    print(f"  winner: P{res['winner_pid']}   bid: {res['winning_bid']}")
    print(f"  n_pickups: {res['n_pickups']}")
    print(f"  gp_per_def: {res['gp_per_def']:+.1f}   gps: {res['gps']}")
    print(f"\nLog:")
    for i, e in enumerate(res['log']):
        print(f"  [{i}] P{e['pid']}: {e}")
