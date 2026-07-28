"""Tournament: CFR bidder vs the composite bidder, card-play held fixed.

Both agents see only (own 10 cards, public bid history). The same standard
auction + PIMC32-soloist-vs-god-defender playout scores every table, so we
isolate bidding quality.

Controlled, paired comparison ("lone hero vs 2 composites", seat-rotated):
  for each eval deal d and hero seat s:
    base  = all-composite table          → composite-hero GP at seat s
    hero  = seat s replaced by the CFR agent, opponents still composite
  delta = CFR-hero GP − composite-hero GP, paired on the same deal+seat+
  opponents. Positive ⇒ CFR is the better bidder.

Usage: N_EVAL=2000 STRATEGY=strategy.pkl python tournament.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent.parent / "17_clean_pickup_net"))
sys.path.insert(0, str(Path(__file__).parent))   # local modules win over repo pkgs

from auction_h2h import _play_pimc_vs_god, _score      # noqa: E402

import game as G                                        # noqa: E402
from common import (ACTIONS, deal_10_10_10_2, raw_avail,  # noqa: E402
                    action_realization, PASS)
from bid_agents import CompositeAgent, OracleComposite, CFRAgent  # noqa: E402
from vnet.pickup.composite import CompositePickup       # noqa: E402

EXP18 = Path(__file__).parent.parent / "18_canonical_pickup"
EXP19 = Path(__file__).parent.parent / "19_colorless_split"

N_EVAL    = int(os.environ.get("N_EVAL", 2000))
SEED_BASE = int(os.environ.get("SEED_BASE", 100_000))
N_WORKERS = int(os.environ.get("N_WORKERS", 8))
STRATEGY  = Path(__file__).parent / os.environ.get("STRATEGY", "strategy.pkl")

_PICKER = _COMP = _HERO = None


COMP_Q = os.environ.get("COMP_Q")           # uniform make-prob prior override
BASELINE = os.environ.get("BASELINE", "clean")   # field/baseline: clean | oracle
HERO = os.environ.get("HERO", "cfr")             # hero seat: cfr | clean | oracle


def _make(kind, Q):
    if kind == "oracle":
        return OracleComposite(_PICKER, Q=Q)
    if kind == "clean":
        return CompositeAgent(_PICKER, Q=Q)
    if kind == "cfr":
        return CFRAgent.load(_PICKER, STRATEGY,
                             fallback=CompositeAgent(_PICKER, Q=Q))
    raise ValueError(kind)


def _init():
    global _PICKER, _COMP, _HERO
    if _PICKER is None:
        _PICKER = CompositePickup.load(
            trump_weights=EXP18 / "multihead_v18a.pt",
            betli_weights=EXP19 / "colorless_betli.pt",
            durchmars_weights=EXP19 / "colorless_durchmars.pt")
        Q = None
        if COMP_Q is not None:
            from common import ACTIONS as _A
            Q = {a: float(COMP_Q) for a in _A}
        _COMP = _make(BASELINE, Q)      # the field (2 seats) + pairing baseline
        _HERO = _make(HERO, Q)          # the lone hero seat
    return _PICKER, _COMP, _HERO


def run_table(hands, talon, agents, picker, base_seed):
    avail = np.array([[bool(raw_avail(hands[p])[a]) for a in ACTIONS]
                      for p in range(3)], dtype=bool)
    ctx = {'avail': avail}
    hist = ()
    import random
    rng = random.Random(base_seed ^ 0x9E3779B9)
    while not G.is_terminal(hist):
        la = G.legal_actions(hist, ctx)
        if len(la) == 1:
            hist = G.apply(hist, la[0])
            continue
        p = G.to_move(hist)
        a = agents[p].act(hand10=hands[p], history=hist, legal_actions=la, rng=rng)
        if a not in la:
            a = PASS
        hist = G.apply(hist, a)

    if G.is_pass_out(hist):
        return {'gps': [-4.0, 2.0, 2.0], 'winner': None,
                'action': 'passout', 'contract': None}
    holder, level = G._holder_level(hist)
    action = G.RANK_TO_ACTION[level]
    real = action_realization(picker, hands[holder], talon, action)
    d1 = hands[(holder + 1) % 3]
    d2 = hands[(holder + 2) % 3]
    pos = _play_pimc_vs_god(sol10=real['sol10'], d1=d1, d2=d2,
                            talon=real['discard'], contract=real['contract'],
                            trump=real['trump'], seed=base_seed * 919)
    gpd = _score(pos, contract=real['contract'], piros=(real['trump'] == 'hearts'))
    gps = [-gpd, -gpd, -gpd]
    gps[holder] = 2 * gpd
    return {'gps': gps, 'winner': holder, 'action': action,
            'contract': f"{real['contract']}/{real['trump'] or 'colorless'}"}


def _worker(seed):
    picker, comp, heroagent = _init()
    h = list(deal_10_10_10_2(seed))
    hands, talon = h[:3], h[3]
    base = run_table(hands, talon, [comp, comp, comp], picker, seed)
    rows = []
    for s in range(3):
        line = [comp, comp, comp]
        line[s] = heroagent
        hero = run_table(hands, talon, line, picker, seed)
        rows.append({
            'seat': s,
            'comp_gp': base['gps'][s],
            'cfr_gp': hero['gps'][s],
            'comp_win': base['winner'] == s,
            'cfr_win': hero['winner'] == s,
            'comp_action': base['action'] if base['winner'] == s else None,
            'cfr_action': hero['action'] if hero['winner'] == s else None,
            'base_action': base['action'],
            'hero_action': hero['action'],
        })
    return rows


def main():
    print(f"=== tournament  N={N_EVAL}  strategy={STRATEGY.name} ===", flush=True)
    seeds = [SEED_BASE + i for i in range(N_EVAL)]
    rows = []
    t0 = time.perf_counter()
    with Pool(N_WORKERS) as pool:
        for rs in pool.imap_unordered(_worker, seeds, chunksize=8):
            rows.extend(rs)
            if len(rows) % 1500 == 0:
                print(f"  {len(rows)//3}/{N_EVAL} deals  "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    print(f"  wall {time.perf_counter()-t0:.0f}s  ({len(rows)} hero-seat rows)\n")

    comp_gp = np.mean([r['comp_gp'] for r in rows])
    cfr_gp = np.mean([r['cfr_gp'] for r in rows])
    deltas = np.array([r['cfr_gp'] - r['comp_gp'] for r in rows])
    n = len(deltas)
    mean_d = deltas.mean()
    se = deltas.std(ddof=1) / np.sqrt(n)
    n_diff = int((deltas != 0).sum())
    print(f"=== headline: lone-hero GP per seat-deal "
          f"(hero={HERO}, field={BASELINE}, COMP_Q={COMP_Q}) ===")
    print(f"  {BASELINE+'-hero':<14}: {comp_gp:+.4f}  (≈0 by zero-sum self-play)")
    print(f"  {HERO+'-hero':<14}: {cfr_gp:+.4f}")
    print(f"  Δ (CFR − comp) : {mean_d:+.4f} ± {1.96*se:.4f} (95% CI)   "
          f"t={mean_d/se:.1f}")
    print(f"  bids differ on {n_diff}/{n} ({n_diff/n*100:.1f}%) seat-deals; "
          f"Δ on those = {deltas[deltas != 0].mean():+.3f}")

    print("\n=== per-seat (0 = forced opener) ===")
    for s in range(3):
        dd = np.array([r['cfr_gp'] - r['comp_gp'] for r in rows if r['seat'] == s])
        se_s = dd.std(ddof=1) / np.sqrt(len(dd))
        nd = int((dd != 0).sum())
        print(f"  seat {s}:  Δ {dd.mean():+.3f} ± {1.96*se_s:.3f}  "
              f"t={dd.mean()/se_s:.1f}  (differ {nd}/{len(dd)})")

    po_c = np.mean([r['base_action'] == 'passout' for r in rows])
    po_f = np.mean([r['hero_action'] == 'passout' for r in rows])
    print(f"\n  pass-out rate:  composite-table {po_c*100:.1f}%   "
          f"CFR-hero-table {po_f*100:.1f}%")

    # what the hero seat ends up playing (when it wins the auction)
    def breakdown(action_key, gp_key, win_key):
        by = defaultdict(list)
        for r in rows:
            tag = r[action_key] if r[win_key] else 'defend/passout'
            by[tag].append(r[gp_key])
        return by

    print("\n=== hero-seat outcome mix (freq%, mean GP) ===")
    print(f"  {'contract played by hero':>26}  {'composite':>20}  {'CFR':>20}")
    cb = breakdown('comp_action', 'comp_gp', 'comp_win')
    fb = breakdown('cfr_action', 'cfr_gp', 'cfr_win')
    keys = sorted(set(cb) | set(fb),
                  key=lambda k: -(len(cb.get(k, [])) + len(fb.get(k, []))))
    n = len(rows)
    for k in keys:
        cv, fv = cb.get(k, []), fb.get(k, [])
        cs = f"{len(cv)/n*100:5.1f}% {np.mean(cv):+6.2f}" if cv else "   —    "
        fs = f"{len(fv)/n*100:5.1f}% {np.mean(fv):+6.2f}" if fv else "   —    "
        print(f"  {k:>26}  {cs:>20}  {fs:>20}")

    # betli/duri commit rate (the bleeders)
    for bleeder in ('betli/colorless', 'durchmars/colorless'):
        cn = sum(1 for r in rows if r['comp_win'] and r['comp_action'] == bleeder)
        fn = sum(1 for r in rows if r['cfr_win'] and r['cfr_action'] == bleeder)
        print(f"\n  hero commits {bleeder:>20}: composite {cn} ({cn/n*100:.2f}%)"
              f"   CFR {fn} ({fn/n*100:.2f}%)")


if __name__ == "__main__":
    main()
