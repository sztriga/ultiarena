"""Reusable A/B validation harness for the contract suite (exp 22).

One `play()` runs a deterministic god-vs-god deal where each side gets its own
objective; `run_ab()` runs paired A/B configs over filtered deals and prints a
metric table + B−A deltas, with flushed tail-able progress.

Each side's config: {sol_contract, sol_weights, def_contract, def_weights}.
A side using contract='multi' has its weights set per-ply (soloist maximises,
defenders minimise), so ANY objective combo works despite the single global
weight vector. A side using a dedicated evaluator ('parti'/'ulti'/'betli'/
'durchmars') ignores the multi weights. Plain parti = multi {'parti_pts':1.0}.

Example — silent-ulti soloist A/B:
    run_ab(dealer='ulti', filt='marriage', oracle_bid=oracle_bid(),
           configs={'A parti': {'sol_weights': {'parti_pts':1.0}},
                    'B +ulti': {'sol_weights': solver_weights(parti=True, silent_ulti=True)}},
           show=['silent_ulti'], n_cand=2000, seed_base=210_000_000)
"""
from __future__ import annotations

import os, sys, time
import multiprocessing as mp
from pathlib import Path

# fork (not macOS-default spawn): workers inherit the already-loaded solver/
# modules instead of re-importing the caller — avoids re-running an unguarded
# run_ab() (fork-bomb) and is faster (no per-worker .so reload).
_MP = mp.get_context('fork')

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
sys.path.insert(0, str(Path(__file__).parent))

from ulti.eval.dojo import deal_ulti_biased, deal_parti, deal_durchmars_colored
from ulti.eval.pimc_matchup import god_pick
from ulti.solvers import pis as pis_bridge
from trickster._solver_core import set_multi_weights
from trickster.games.ulti.game import (soloist_won_simple, soloist_points,
                                       soloist_tricks, soloist_won_durchmars,
                                       marriage_points)
from ulti.scoring.oracle import score as score_oracle
from recipe import sol_marriages

# ── dealers + filters (named so they survive multiprocessing) ──────────────
DEALERS = {'ulti': deal_ulti_biased, 'parti': deal_parti,
           'duri': deal_durchmars_colored}


def _f_none(d):      return True
def _f_has40(d):     return sol_marriages(d.sol_hand, d.trump)[0]
def _f_has20(d):     return sol_marriages(d.sol_hand, d.trump)[1]
def _f_marriage(d):  h40, h20 = sol_marriages(d.sol_hand, d.trump); return h40 or h20
def _f_has40_only(d): h40, h20 = sol_marriages(d.sol_hand, d.trump); return h40 and not h20
def _f_has20_only(d): h40, h20 = sol_marriages(d.sol_hand, d.trump); return h20 and not h40
def _f_def7(d):
    return any(c.suit == d.trump and c.rank == '7'
               for c in list(d.def1_hand) + list(d.def2_hand))

FILTERS = {'none': _f_none, 'has_40': _f_has40, 'has_20': _f_has20,
           'marriage': _f_marriage, 'has_40_only': _f_has40_only,
           'has_20_only': _f_has20_only, 'def_holds_7': _f_def7}


# ── playout + metric bundle ────────────────────────────────────────────────
def play(deal, cfg):
    """Deterministic god-vs-god playout under cfg (a config dict)."""
    sc = cfg.get('sol_contract', 'multi')
    dc = cfg.get('def_contract', 'parti')
    sw = cfg.get('sol_weights') or {'parti_pts': 1.0}
    dw = cfg.get('def_weights') or {'parti_pts': 1.0}
    both_multi = (sc == 'multi' and dc == 'multi')  # only then must we re-set per ply
    if sc == 'multi' and not both_multi:
        set_multi_weights(**sw)
    if dc == 'multi' and not both_multi:
        set_multi_weights(**dw)

    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=0, leader=0, contract='parti',
        trump=deal.trump, talon=list(deal.talon), declare_marriages=True,
    )
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        if p == 0:
            if both_multi:
                set_multi_weights(**sw)
            chosen = god_pick(pos=pos, contract=sc)
        else:
            if both_multi:
                set_multi_weights(**dw)
            chosen = god_pick(pos=pos, contract=dc)
        pis_bridge.apply_move(pos, chosen)
    return pos


def _bundle(fp, pv):
    """Standard metric bundle from a played-out position + scored payoff."""
    c = pv.components
    sp = soloist_points(fp)
    return {
        'gp':           pv.total_per_def,
        'parti_won':    float(soloist_won_simple(fp)),
        'sol_pts':      float(sp),
        'card_pts':     float(sp - marriage_points(fp, fp.soloist)),
        'reached_100':  float(sp >= 100),
        'silent_ulti':     float(c.get('silent_ulti', 0) > 0),   # sol won 7 @t10
        'def_silent_ulti': float(c.get('silent_ulti', 0) < 0),   # def won 7
        'sol_swept':       float(soloist_won_durchmars(fp)),
        'got_silent_dm':   float(c.get('silent_durchmars', 0) != 0),
        'got_40_100':      float(c.get('silent_40_100', 0) != 0),
        'got_20_100':      float(c.get('silent_20_100', 0) != 0),
        'bid_40_100_made': float(c.get('40_100', 0) > 0),
        'bid_20_100_made': float(c.get('20_100', 0) > 0),
        'betli_won':       float(soloist_tricks(fp) == 0),
    }


# ── multiprocessing plumbing ───────────────────────────────────────────────
_G = {}


def _init(dealer, filt, configs, oracle_bid, alpha):
    _G.update(dealer=dealer, filt=filt, configs=configs,
              oracle_bid=oracle_bid, alpha=alpha)


def _worker(seed):
    deal = DEALERS[_G['dealer']](seed=seed, alpha=_G['alpha'])
    if not FILTERS[_G['filt']](deal):
        return None
    out = {}
    for name, cfg in _G['configs'].items():
        fp = play(deal, cfg)
        pv = score_oracle(final_pos=fp, bid=_G['oracle_bid'])
        out[name] = _bundle(fp, pv)
    return (seed, out)


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / n) ** 0.5


def run_ab(*, dealer, oracle_bid, configs, n_cand, show=(), filt='none',
           seed_base, workers=8, alpha=0.6, label=''):
    """Run paired A/B(/C/D...) configs over filtered deals; print a metric
    table and B−A deltas (vs the FIRST config) on gp + each `show` metric."""
    names = list(configs.keys())
    metrics = ['gp', *show]
    print(f"[{label or 'run_ab'}] dealer={dealer} filt={filt} "
          f"CAND={n_cand} configs={names}", flush=True)
    t0 = time.perf_counter()
    rows, scanned = [], 0
    with _MP.Pool(workers, initializer=_init,
                  initargs=(dealer, filt, configs, oracle_bid, alpha)) as pool:
        for r in pool.imap_unordered(_worker,
                                     (seed_base + i for i in range(n_cand)),
                                     chunksize=8):
            scanned += 1
            if r is not None:
                rows.append(r)
            if scanned % 500 == 0:
                print(f"  scanned {scanned}/{n_cand} kept {len(rows)} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    N = len(rows)
    print(f"\nkept {N}/{n_cand}  wall={time.perf_counter()-t0:.0f}s")
    if N == 0:
        return []

    def col(name, metric):
        return [r[1][name][metric] for r in rows]

    for name in names:
        cells = []
        for m in metrics:
            mu, sd = _mean_std(col(name, m))
            cells.append(f"{m} {mu:+.3f}" if m == 'gp' else f"{m} {mu:.3f}")
        print(f"  {name:<22} | " + " | ".join(cells))

    base = names[0]
    print(f"\n  deltas vs '{base}':")
    for name in names[1:]:
        line = []
        for m in metrics:
            d = [r[1][name][m] - r[1][base][m] for r in rows]
            mu, sd = _mean_std(d)
            if m == 'gp':
                t = mu / (sd / N ** 0.5) if sd > 0 else 0.0
                line.append(f"gp {mu:+.3f} (t={t:+.1f})")
            else:
                line.append(f"{m} {mu:+.3f}")
        print(f"  {name:<22} − {base:<14} | " + " | ".join(line))
    return rows
