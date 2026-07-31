"""Every move cull must be value-preserving.

The solver drops moves from the search in three places:

  * ``_cull_betli_soloist``    — betli soloist: keep only the highest card per suit
                                 (DOMINANCE — the dropped cards are worse, not equal)
  * ``_cull_betli_def_groups`` — betli defenders: merge runs with no live card between
  * ``_cull_parti_blocks``     — parti/ulti/duri/multi: same, plus a card-points split
                                 and the trump-7 kept isolated

All three are justified by an argument in a comment, and everything the engine plays
rests on those arguments being true. This test pins them: solve every legal move at a
random mid-game position with the culls ON (production) and OFF (``_cull_noop``
everywhere) and require identical values. If someone widens a cull past what it can
prove, the values move and this fails loudly.

``solve_root`` evaluates every legal move at the root — the cull only acts at interior
nodes — so a difference here is exactly "the cull pruned a line that mattered".
"""
from __future__ import annotations

import random

import pytest

from _positions import CASES, apply_weights, make_position
from ulti.solvers import pis
from ultisolver._solver_core import get_cull_enabled, set_cull_enabled

N_POSITIONS = 40
PLY_DEPTHS = (3, 5, 8, 11)   # deliberately not all trick boundaries: vary who is on move
TOL = 1e-5


@pytest.fixture(autouse=True)
def _restore_cull():
    yield
    set_cull_enabled(1)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_cull_preserves_every_move_value(case):
    label, solve_c = case[0], case[1]
    rng = random.Random(hash(label) & 0xFFFF)
    checked = 0
    for i in range(N_POSITIONS * 3):
        if checked >= N_POSITIONS:
            break
        pos = make_position(rng, case, PLY_DEPTHS[i % len(PLY_DEPTHS)])
        if pos is None:
            continue
        apply_weights(case)

        set_cull_enabled(1)
        with_cull = pis.solve_all(pos, contract=solve_c)
        set_cull_enabled(0)
        without_cull = pis.solve_all(pos, contract=solve_c)
        set_cull_enabled(1)

        assert set(with_cull) == set(without_cull), (
            f"{label}: cull changed the set of root moves")
        for card, value in with_cull.items():
            assert value == pytest.approx(without_cull[card], abs=TOL), (
                f"{label}: cull changed the value of {card} "
                f"({value} with cull vs {without_cull[card]} without) — "
                f"the cull pruned a line that mattered")
        checked += 1

    assert checked >= N_POSITIONS // 2, (
        f"{label}: only {checked} usable positions generated")


def test_toggle_defaults_to_on():
    """Production must never run with the culls disabled."""
    assert get_cull_enabled() == 1
