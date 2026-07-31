"""Cards inside an equivalence block really are interchangeable.

``ulti.solvers.blocks`` states the "these are the same move" rule in Python so play-time
code can randomise inside a block and stop leaking a tell (always leading the top of a
run tells the opponent you hold nothing above it). That is only safe if the block really
is equivalent.

This proves it directly rather than by inspection: ``solve_root`` returns the exact
minimax value of EVERY legal move, so every card in a block must come back with the same
number. Anything else means the rule merged two cards that play differently.

The rule is fed PUBLIC information only (``live_others`` treats every unseen card,
talon included, as still live) — the same information the engine has at the table, and
strictly less than the solver's determinized view. Less information can only split
blocks further, so passing here is the conservative direction.
"""
from __future__ import annotations

import random

import pytest

from _positions import CASES, apply_weights, make_position
from ulti.card import Card
from ulti.solvers import pis
from ulti.solvers.blocks import equivalence_blocks, live_others

N_POSITIONS = 60
PLY_DEPTHS = (3, 5, 8, 11)   # deliberately not all trick boundaries: vary who is on move
TOL = 1e-5


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_block_members_have_identical_value(case):
    label, solve_c, _build_c, trump, _has_ulti, colorless, _w = case
    rng = random.Random(hash(label) & 0xFFFF)
    isolate = frozenset({Card(suit=trump, rank="7").id}) if trump else frozenset()

    checked = merged = 0
    for i in range(N_POSITIONS * 3):
        if checked >= N_POSITIONS:
            break
        pos = make_position(rng, case, PLY_DEPTHS[i % len(PLY_DEPTHS)])
        if pos is None:
            continue
        apply_weights(case)

        viewer = pis.current_player(pos)
        values = pis.solve_all(pos, contract=solve_c)
        # Block over the solver's OWN move set, not legal_actions(): for a betli-family
        # soloist ``_legal`` has already applied the dominance cull (keep the highest card
        # per suit), so the dominated cards have no value to compare against. That is the
        # same set the play-time mixer is allowed to touch — see test_betli_soloist_*.
        blocks = equivalence_blocks(list(values), live_others(pos, viewer),
                                    colorless=colorless, isolate=isolate)

        for block in blocks:
            if len(block) < 2:
                continue
            merged += 1
            head = values[block[0]]
            for card in block[1:]:
                assert values[card] == pytest.approx(head, abs=TOL), (
                    f"{label}: {block[0]} and {card} were called interchangeable but "
                    f"solve to {head} vs {values[card]} — the block rule is wrong, "
                    f"randomising inside it would throw away value")
        checked += 1

    assert checked >= N_POSITIONS // 2, f"{label}: only {checked} usable positions"
    assert merged > 0, (
        f"{label}: no multi-card blocks were produced — the test proved nothing")


def test_betli_soloist_moves_are_pre_reduced_by_dominance():
    """Why the play-time mixer must skip the betli-family soloist.

    ``_legal`` itself keeps only the highest card per suit for the soloist whenever
    ``betli`` is set — which covers plain betli AND colourless durchmars, since
    ``build_position`` maps the latter onto the betli rank ordering. That is a DOMINANCE
    reduction: the dropped cards are worse, not equal, so randomising over them would
    give away real value. The Python ``legal_actions`` does not apply it, so the two move
    lists legitimately differ and the mixer must not use the wider one.
    """
    rng = random.Random(11)
    saw_reduction = False
    for case in CASES:
        if not case[5]:                      # colourless == betli-family
            continue
        for i in range(60):
            pos = make_position(rng, case, PLY_DEPTHS[i % len(PLY_DEPTHS)])
            if pos is None or pis.current_player(pos) != 0:   # 0 == soloist
                continue
            solver_moves = set(pis.solve_all(pos, contract=case[1]))
            python_moves = set(pis.legal_actions(pos))
            assert solver_moves <= python_moves
            if solver_moves < python_moves:
                saw_reduction = True
                suits = [c.suit for c in solver_moves]
                assert len(suits) == len(set(suits)), (
                    "dominance cull must leave at most one card per suit")
    assert saw_reduction, "expected at least one soloist position to be reduced"


def test_captured_cards_do_not_break_a_run():
    """A 'plugged hole' merges: 7-9 are one block once the 8 is out of circulation."""
    hand = [Card(suit="hearts", rank="7"), Card(suit="hearts", rank="9")]
    eight = Card(suit="hearts", rank="8")

    still_live = equivalence_blocks(hand, [eight], colorless=False)
    assert len(still_live) == 2, "a live 8 must split 7 and 9"

    plugged = equivalence_blocks(hand, [], colorless=False)
    assert len(plugged) == 1 and len(plugged[0]) == 2, "a dead 8 must merge 7 and 9"


def test_points_split_a_run():
    """King and Ten are rank-adjacent in colourless order but not point-equal."""
    hand = [Card(suit="hearts", rank="king"), Card(suit="hearts", rank="ace")]
    assert len(equivalence_blocks(hand, [], colorless=False)) == 2


def test_trump_seven_is_isolated():
    """The trump 7 carries the ulti payoff — it never shares a block."""
    hand = [Card(suit="hearts", rank="7"), Card(suit="hearts", rank="8")]
    isolate = frozenset({Card(suit="hearts", rank="7").id})
    assert len(equivalence_blocks(hand, [], colorless=False, isolate=isolate)) == 2
    assert len(equivalence_blocks(hand, [], colorless=False)) == 1
