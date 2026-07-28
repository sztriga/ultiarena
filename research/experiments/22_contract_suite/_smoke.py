"""Harness smoke: reproduce the silent-ulti soloist A/B (exp21 ≈ B−A +1.25)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_ab
from engine import solver_weights, oracle_bid

# NB: keep run_ab() under __main__ — contract scripts should too.
if __name__ == '__main__':
    run_ab(
        label='smoke silent-ulti',
        dealer='ulti', filt='marriage', oracle_bid=oracle_bid(),
        configs={
            'A parti':  {'sol_weights': {'parti_pts': 1.0}, 'def_contract': 'parti'},
            'B +ulti':  {'sol_weights': solver_weights(parti=True, silent_ulti=True),
                         'def_contract': 'parti'},
        },
        show=['silent_ulti', 'parti_won'],
        n_cand=1500, seed_base=210_000_000, workers=8,
    )
