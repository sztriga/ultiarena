"""Pre-pickup oracle: E[P(make X) | 10-card hand], enumerated over talons.

Per hand, per contract, per trump:
  1. Enumerate all 231 possible talons (2 of the 22 unseen cards).
  2. For each talon, form the 12-card hand and consider all 66 discards.
  3. Use exp 15 v2 net to predict P(make X) for each (talon, discard).
  4. Per talon: best = max P over discards.
  5. Aggregate: mean over talons = E[best P | hand].

Also reports max / min / median over talons for inspection.

Public entry: evaluate(sol_hand_10, all_unseen_cards) → dict.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import sys
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "15_vnet_pickup"))
sys.path.insert(0, str(Path(__file__).parent.parent / "14_minigame_bid_eval"))

from ulti.vnet.pickup import CONTRACT_CONFIGS, PickupNetV2, featurize, input_dim
from ulti.card import SUITS, DECK, Card
from _lib import _ev_per_def

EXP15_DIR = Path(__file__).parent.parent / "15_vnet_pickup"


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


def _all_unseen(sol_hand: list[Card]) -> list[Card]:
    sol_set = set(sol_hand)
    return [c for c in DECK if c not in sol_set]


def evaluate(sol_hand_10: list[Card],
             unseen: Optional[list[Card]] = None) -> dict:
    """Return per-(contract, trump) statistics.

    Output keys:
      (contract_name, trump_or_None) → {
        'mean_best_p': scalar, mean over 231 talons of max-over-discards P,
        'mean_ev':     scalar, EV per defender from mean_best_p,
        'best_per_talon': np.ndarray (231,),
        'max_best_p':  scalar, max over talons,
        'min_best_p':  scalar, min over talons,
      }
    """
    nets = _load_nets()
    if unseen is None:
        unseen = _all_unseen(sol_hand_10)
    assert len(sol_hand_10) == 10
    assert len(unseen) == 22

    talons = list(itertools.combinations(unseen, 2))   # 231
    n_talons = len(talons)
    # Pre-build all 12-card hands and the 66 final-10-card hands per talon.
    # We do this once per hand and reuse across contracts; trump only enters
    # via the featurizer.
    all_finals = []  # shape: (n_talons, 66, 10)
    for talon in talons:
        hand12 = sol_hand_10 + list(talon)
        per_talon = []
        for discard in itertools.combinations(range(12), 2):
            final10 = [hand12[i] for i in range(12) if i not in discard]
            per_talon.append(final10)
        all_finals.append(per_talon)
    # shape (231 * 66, 10) flat
    flat_finals = [h for talon_rows in all_finals for h in talon_rows]
    assert len(flat_finals) == n_talons * 66

    out = {}
    for cname, cfg in CONTRACT_CONFIGS.items():
        trumps = SUITS if cfg.has_trump else [None]
        for trump in trumps:
            X = np.stack([
                featurize(h, trump, cfg.has_trump) for h in flat_finals
            ])
            with torch.no_grad():
                ps = nets[cname](torch.from_numpy(X)).numpy()
            ps_mat = ps.reshape(n_talons, 66)
            best_per_talon = ps_mat.max(axis=1)   # (231,)
            mean_best = float(best_per_talon.mean())
            piros = (trump == 'hearts')
            mean_ev = _ev_per_def(cname, piros, mean_best)
            out[(cname, trump)] = {
                'mean_best_p': mean_best,
                'mean_ev':     mean_ev,
                'max_best_p':  float(best_per_talon.max()),
                'min_best_p':  float(best_per_talon.min()),
                'best_per_talon': best_per_talon,
            }
    return out


def pick(sol_hand_10: list[Card], *,
         thresh: Optional[dict] = None) -> Optional[dict]:
    """Apply per-contract pass thresholds and return the best pick.

    thresh: {contract_name: min_ev}. Defaults to all 0.
    Returns dict with chosen contract/trump/ev/p, or None for pass.
    """
    if thresh is None:
        thresh = {c: 0.0 for c in CONTRACT_CONFIGS}
    stats = evaluate(sol_hand_10)
    best = None
    for (cname, trump), s in stats.items():
        if s['mean_ev'] < thresh.get(cname, 0.0):
            continue
        if best is None or s['mean_ev'] > best['mean_ev']:
            best = {
                'contract': cname,
                'trump': trump,
                'mean_ev': s['mean_ev'],
                'mean_p': s['mean_best_p'],
            }
    return best


if __name__ == "__main__":
    # Smoke: evaluate a single deal from exp 14
    from _lib import deal_12_10_10
    import time

    sol12, _, _ = deal_12_10_10(seed=100000)
    sol10 = sol12[:10]   # first 10 — fake split into "10 + 2 talon"
    print(f"Sol's 10-card hand: {sorted(c.id for c in sol10)}")
    t0 = time.perf_counter()
    stats = evaluate(sol10)
    wall = time.perf_counter() - t0
    print(f"\nEvaluation wall: {wall*1000:.1f} ms")
    print()
    print(f"  {'contract':>22}  {'mean P':>7}  {'mean EV':>8}  "
          f"{'min P':>6}  {'max P':>6}")
    rows = sorted(stats.items(), key=lambda x: -x[1]['mean_ev'])
    for (cname, trump), s in rows:
        label = f"{cname}/{trump or 'colorless'}"
        print(f"  {label:>22}  {s['mean_best_p']:>7.3f}  "
              f"{s['mean_ev']:>+7.3f}  "
              f"{s['min_best_p']:>6.3f}  {s['max_best_p']:>6.3f}")
    print()
    p = pick(sol10)
    if p is None:
        print("Pick: pass")
    else:
        print(f"Pick: {p['contract']}/{p['trump'] or 'colorless'}  "
              f"mean_ev={p['mean_ev']:+.3f}  mean_p={p['mean_p']:.3f}")
