"""Solver performance benchmark across all contract types.

For each contract, time a FULL double-dummy solve (solve_best from trick 1, cold
TT) on the same random deals. Single-process for clean per-solve latency. Reports
mean / median / p95 / max ms per contract, with a flushed running heartbeat.

Each row = (label, build_contract, solve_contract, trump?, multi-weights). The
position is built with build_contract (sets betli/has_ulti flags + rank) and
solved with solve_contract's evaluator. multi rows set the weights first.
"""
import os, sys, time
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from ulti.eval.dojo import deal_ulti_biased
from ulti.solvers import pis
from trickster._solver_core import solve_best, set_multi_weights

N    = int(os.environ.get('N', '300'))
SEED = int(os.environ.get('SEED', '900000000'))

CONFIGS = [
    # label,              build,       solve,       trump,   weights
    ('parti',             'parti',     'parti',     'trump', None),
    ('ulti',              'ulti',      'ulti',      'trump', None),
    ('betli',             'betli',     'betli',     None,    None),
    ('duri-colored',      'durchmars', 'durchmars', 'trump', None),
    ('duri-colorless',    'durchmars', 'durchmars', None,    None),
    ('multi[parti]',      'parti',     'multi',     'trump', {'parti_pts': 1.0}),
    ('multi[parti+ulti]', 'parti',     'multi',     'trump', {'parti_pts': 1.0, 'silent_ulti': 2.0}),
    ('multi[parti+dm]',   'parti',     'multi',     'trump', {'parti_pts': 1.0, 'silent_durchmars': 3.0}),
    ('multi[parti+100]',  'parti',     'multi',     'trump', {'parti_pts': 1.0, 'score_geq_100': 2.0}),
    ('multi[full]',       'parti',     'multi',     'trump',
        {'parti_pts': 1.0, 'silent_ulti': 2.0, 'score_geq_100': 2.0, 'silent_durchmars': 3.0}),
]


def pct(xs, p):
    s = sorted(xs)
    return s[int(p / 100.0 * (len(s) - 1))]


def main():
    print(f"[perf] N={N} deals × {len(CONFIGS)} contracts | full-deal double-dummy "
          f"solve (cold TT), single-process | dealer=ulti_biased", flush=True)
    times = {c[0]: [] for c in CONFIGS}
    t_start = time.perf_counter()
    for i in range(N):
        deal = deal_ulti_biased(seed=SEED + i, alpha=0.6)
        base = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
        for label, bc, sc, tspec, w in CONFIGS:
            trump = deal.trump if tspec == 'trump' else None
            gs = pis.build_position(hands=[list(h) for h in base], soloist=0,
                                    leader=0, contract=bc, trump=trump)
            if w is not None:
                set_multi_weights(**w)
            t0 = time.perf_counter()
            solve_best(gs, contract=sc)
            times[label].append((time.perf_counter() - t0) * 1000.0)
        if (i + 1) % 25 == 0:
            wall = time.perf_counter() - t_start
            rate = (i + 1) / wall
            eta = (N - i - 1) / rate
            means = " ".join(f"{lbl.split('[')[0][:5]}:{sum(times[lbl])/len(times[lbl]):.0f}"
                             for lbl, *_ in CONFIGS)
            print(f"  {i+1}/{N} wall={wall:.0f}s eta={eta:.0f}s | mean ms {means}", flush=True)

    wall = time.perf_counter() - t_start
    print(f"\n=== solve time per contract (ms) | N={N} | wall={wall:.0f}s ===")
    print(f"{'contract':<20}{'mean':>9}{'median':>9}{'p95':>9}{'p99':>9}{'max':>9}")
    for label, *_ in CONFIGS:
        xs = times[label]
        print(f"{label:<20}{sum(xs)/len(xs):>9.1f}{pct(xs,50):>9.1f}"
              f"{pct(xs,95):>9.1f}{pct(xs,99):>9.1f}{max(xs):>9.1f}", flush=True)
    # total throughput
    allt = [t for xs in times.values() for t in xs]
    print(f"\ntotal solves: {len(allt)} | overall mean {sum(allt)/len(allt):.1f} ms "
          f"| {len(allt)/wall:.0f} solves/s")


if __name__ == '__main__':
    main()
