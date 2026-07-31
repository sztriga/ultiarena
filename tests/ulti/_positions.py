"""Random mid-game position generator shared by the cull tests.

Deals a real 10/10/10 + 2-talon game, plays a random legal prefix, and hands back the
resulting position. Mid-game is the interesting case for the block rule: captured cards
create the "plugged holes" that let a run merge in the first place.
"""
from __future__ import annotations

import random
from typing import Any, List, Optional, Tuple

from ulti.card import DECK
from ulti.solvers import pis

# (label, solve_contract, build_contract, trump, has_ulti, colorless, multi_weights)
# Mirrors what apps/api/play.py actually solves with: betli / durchmars / multi("parti").
CASES: List[Tuple[str, str, str, Optional[str], bool, bool, Optional[dict]]] = [
    ("betli",            "betli",     "betli",     None,     False, True,  None),
    ("duri-colorless",   "durchmars", "durchmars", None,     False, True,  None),
    ("duri-colored",     "durchmars", "durchmars", "hearts", False, False, None),
    ("ulti",             "ulti",      "ulti",      "hearts", True,  False, None),
    ("parti",            "parti",     "parti",     "hearts", False, False, None),
    # The production play path: EV_MULTI with a non-zero silent_ulti weight, which is
    # what arms the trump-7 isolation inside _cull_parti_blocks for BOTH sides.
    ("multi-silent-ulti", "multi",    "parti",     "hearts", False, False,
     {"parti_pts": 1.0, "silent_ulti": 4.0, "score_geq_100": 2.0}),
]


def make_position(rng: random.Random, case, n_plies: int) -> Optional[Any]:
    """Deal, play ``n_plies`` random legal moves, return the position (None if it ended)."""
    _label, solve_c, build_c, trump, has_ulti, colorless, _w = case
    deck = list(DECK)
    rng.shuffle(deck)
    if solve_c == "durchmars":
        # A random hand loses durchmars on trick 1 (the soloist must win every trick), so
        # every position would be terminal before it got interesting. Hand the soloist the
        # 6 strongest cards — strong enough to keep winning tricks, still not a lock.
        from ulti.solvers.blocks import strength as _str
        rank = sorted(deck, reverse=True,
                      key=lambda c: (c.suit == trump, _str(c, colorless)))
        top = rank[:6]
        top_ids = {c.id for c in top}
        rest = [c for c in deck if c.id not in top_ids]
        rng.shuffle(rest)
        deck = top + rest[:4] + rest[4:]
    hands = [deck[0:10], deck[10:20], deck[20:30]]
    talon = deck[30:32]
    pos = pis.build_position(
        hands=hands, soloist=0, leader=0, contract=build_c, trump=trump,
        talon=talon, declare_marriages=(trump is not None), has_ulti=has_ulti,
    )
    for _ in range(n_plies):
        if pis.is_terminal(pos):
            return None
        acts = pis.legal_actions(pos)
        if not acts:
            return None
        # Prefer a move that keeps the contract alive. Betli ends the moment the soloist
        # takes a trick and durchmars the moment they lose one, so purely random play
        # collapses those games within a trick or two and we would never reach a
        # mid-game position worth testing.
        rng.shuffle(acts)
        for cand in acts:
            probe = pos.clone()
            pis.apply_move(probe, cand)
            if not pis.is_terminal(probe):
                pis.apply_move(pos, cand)
                break
        else:
            return None
    if pis.is_terminal(pos) or len(pis.legal_actions(pos)) < 2:
        return None      # nothing to compare when there is no choice
    return pos


def apply_weights(case) -> None:
    """Install the EV_MULTI weight vector for ``case`` (no-op for the other contracts)."""
    weights = case[6]
    if weights is None:
        return
    from ultisolver._solver_core import set_multi_weights
    set_multi_weights(**weights)
