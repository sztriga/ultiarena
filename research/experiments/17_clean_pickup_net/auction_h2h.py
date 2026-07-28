"""Head-to-head auction: each seat uses its own pickup net.

Same rules as auction_v2 / tier3_auction, but the pickup net is per-seat.
pickers: [picker_p0, picker_p1, picker_p2]  — each has .predict(X, contract).
"""
from __future__ import annotations

import itertools, os, random, sys
from pathlib import Path
from typing import Optional, List

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "15_vnet_pickup"))

from _lib import deal_12_10_10, _ev_per_def
from ulti.solvers import pis, pimc as _pimc, determinize as _det
from ulti.eval.pimc_matchup import god_pick
from ulti.card import SUITS, DECK
from ulti.vnet.pickup import CONTRACT_CONFIGS, featurize
from ulti.scoring.oracle import BidSet, score as score_oracle
import itertools as _it

PIMC_N = 32
PASS_PENALTY = -2.0
BID_FLOOR = -2.0

# CANONICAL BIDDER (exp 20). The bid DECISION scores each contract by the
# DEBIAS_PCTL quantile of the 66 discard ps instead of the max — the max
# over-selects lucky noise (optimizer's curse) and bids inflated marginal
# contracts (betli/ulti). The discard actually PLAYED is still the argmax (that
# is not the inflation source). Quantile (not raw-10) so the talon's cards still
# count for talon-holders (real-harness P0). Real-harness head-to-head:
# +0.38 GP/seat-deal (t=5.2) over the old max behaviour.
#   ON BY DEFAULT.  Set DEBIAS_BID=0 to reproduce the old (inflated, the numbers
#   reported in exp 15-19) bidder.
DEBIAS_BID = os.environ.get("DEBIAS_BID", "1").lower() not in ("0", "false", "no")
DEBIAS_PCTL = float(os.environ.get("DEBIAS_PCTL", "0.80"))


def _debias_pctl(picker):
    """Debias quantile for de-inflating the 66 discard scores, or None for the
    original max (deployed) behaviour. A per-picker ``debias_pctl`` attribute
    overrides the global flag, so patched and deployed seats can be mixed at
    one table (head-to-head)."""
    p = getattr(picker, 'debias_pctl', None)
    if p is not None:
        return p
    return DEBIAS_PCTL if DEBIAS_BID else None


def contract_rank(contract, trump):
    piros = (trump == 'hearts')
    if contract == 'parti':     return 2 if piros else 1
    if contract == 'ulti':      return 6 if piros else 3
    if contract == 'betli':     return 4
    if contract == 'durchmars': return 5
    raise ValueError(contract)


def _is_legal_bid(contract, trump):
    if contract == 'parti' and trump != 'hearts':
        return False
    return True


def _best_bid_above_rank(hand12, *, picker, min_rank, ev_floor=None):
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
            ps = picker.predict(X, cname)
            piros = (trump == 'hearts')
            valid = []
            for i, dp in enumerate(discards):
                if cname == 'ulti':
                    rem = [c for c in hand12 if c not in dp]
                    if not any(c.suit == trump and c.rank == '7' for c in rem):
                        continue
                valid.append(i)
            if not valid:
                continue
            vps = np.array([float(ps[i]) for i in valid])
            play_dp = discards[valid[int(vps.argmax())]]   # discard to PLAY
            # decision p: argmax (default) or de-inflated quantile (debias)
            pctl = _debias_pctl(picker)
            dec_p = (float(np.quantile(vps, pctl)) if pctl is not None
                     else float(vps.max()))
            ev = _ev_per_def(cname, piros, dec_p)
            if best is None or ev > best[0]:
                best = (ev, play_dp, cname, trump, dec_p)
    if best is None:
        return None
    if ev_floor is not None and best[0] < ev_floor:
        return None
    return best


def _oracle_evaluate(sol_hand_10, *, picker, min_rank=0):
    sol_set = set(sol_hand_10)
    unseen = [c for c in DECK if c not in sol_set]
    talons = list(_it.combinations(unseen, 2))
    n_talons = len(talons)
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
            ps = picker.predict(X, cname)
            ps_mat = ps.reshape(n_talons, 66)
            pctl = _debias_pctl(picker)
            best_per_talon = (np.quantile(ps_mat, pctl, axis=1)
                              if pctl is not None else ps_mat.max(axis=1))
            mean_p = float(best_per_talon.mean())
            piros = (trump == 'hearts')
            mean_ev = _ev_per_def(cname, piros, mean_p)
            if best is None or mean_ev > best['mean_ev']:
                best = {'contract': cname, 'trump': trump,
                        'mean_ev': mean_ev, 'mean_p': mean_p,
                        'rank': contract_rank(cname, trump)}
    return best


def _sol_p_make(sol_hand_play, def1, def2, *, contract, trump, talon, seed):
    pos = pis.build_position(
        hands=[sol_hand_play, def1, def2], soloist=0, leader=0,
        contract=contract, trump=trump, talon=talon,
    )
    _, avg = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=PIMC_N, seed=seed,
    )
    return max(0.0, min(1.0, max(avg.values()))) if avg else 0.0


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


def simulate(seed: int, pickers: List, play_out: bool = True) -> dict:
    """pickers[pid] is the picker for seat pid (0/1/2).

    ``play_out=False`` resolves the auction only (n_pickups, winner,
    bid_seq) and skips the PIMC32-vs-god play-out — much faster when you
    only need auction-level stats. ``n_pickups`` counts every commit
    (P0's open + each overtake); ``bid_seq`` is the list of
    (pid, contract, trump) commits in order.
    """
    sol12, d1_10, d2_10 = deal_12_10_10(seed)
    sol10_orig = sol12[:10]
    talon_orig = sol12[10:]
    hands = [sol10_orig.copy(), list(d1_10), list(d2_10)]
    talon = list(talon_orig)

    # P0 opener
    p0_hand12 = hands[0] + talon
    pick = _best_bid_above_rank(p0_hand12, picker=pickers[0],
                                min_rank=0, ev_floor=BID_FLOOR)
    if pick is None:
        gps = [2 * PASS_PENALTY, -PASS_PENALTY, -PASS_PENALTY]
        return {'seed': seed, 'winner_pid': None,
                'winning_bid': 'PASS_PENALTY', 'gp_per_def': PASS_PENALTY,
                'gps': gps, 'n_pickups': 0, 'bid_seq': []}
    ev, discard, cname, trump, p_v = pick
    hands[0] = [c for c in p0_hand12 if c not in discard]
    talon = list(discard)
    current = {'pid': 0, 'contract': cname, 'trump': trump,
               'rank': contract_rank(cname, trump)}
    n_pickups = 1
    bid_seq = [(0, cname, trump)]

    passes = 0
    next_pid = 1
    while passes < 3:
        if next_pid == current['pid']:
            passes += 1
        else:
            cand = _oracle_evaluate(hands[next_pid],
                                    picker=pickers[next_pid],
                                    min_rank=current['rank'])
            if cand is None:
                passes += 1
            else:
                p_make = _sol_p_make(
                    hands[current['pid']], hands[(current['pid']+1)%3],
                    hands[(current['pid']+2)%3],
                    contract=current['contract'], trump=current['trump'],
                    talon=talon, seed=seed * 131 + next_pid * 17,
                )
                pass_ev = -_ev_per_def(current['contract'],
                                       current['trump'] == 'hearts', p_make)
                if cand['mean_ev'] <= pass_ev:
                    passes += 1
                else:
                    new_hand12 = hands[next_pid] + talon
                    pick = _best_bid_above_rank(new_hand12,
                                                picker=pickers[next_pid],
                                                min_rank=current['rank'])
                    if pick is None:
                        passes += 1
                    else:
                        ev, discard, cname2, trump2, p_v2 = pick
                        hands[next_pid] = [c for c in new_hand12 if c not in discard]
                        talon = list(discard)
                        current = {'pid': next_pid, 'contract': cname2,
                                   'trump': trump2,
                                   'rank': contract_rank(cname2, trump2)}
                        n_pickups += 1
                        bid_seq.append((next_pid, cname2, trump2))
                        passes = 0
        next_pid = (next_pid + 1) % 3

    winner_pid = current['pid']
    if not play_out:
        return {'seed': seed, 'winner_pid': winner_pid,
                'winning_bid': f"{current['contract']}/{current['trump'] or 'colorless'}",
                'gp_per_def': None, 'gps': None,
                'n_pickups': n_pickups, 'bid_seq': bid_seq,
                'contract': current['contract'], 'trump': current['trump'],
                'sol_hand': list(hands[winner_pid]),
                'def1': list(hands[(winner_pid + 1) % 3]),
                'def2': list(hands[(winner_pid + 2) % 3]),
                'talon': list(talon)}
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
        'gp_per_def': gp_per_def, 'gps': gps,
        'n_pickups': n_pickups, 'bid_seq': bid_seq,
    }
