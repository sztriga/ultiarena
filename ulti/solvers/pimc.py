"""
Perfect Information Monte Carlo (PIMC) player.

Algorithm
---------
1. Build the on-move player's information set from the public position
   plus optional play-history evidence (voids, must-hold from auction).
2. Sample N random determinizations consistent with that information set.
3. Solve each determinization with the god solver from ``solvers.pis``.
4. Average the per-move values across samples; return the argmax.

The information-set construction (talon visibility, void constraints,
auction must-holds, contract-specific rules) lives in
:mod:`solvers.determinize`. This file is a thin loop.
"""
from __future__ import annotations

import random
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ulti.solvers import determinize as _det
from ulti.solvers import pis as _pis
from ulti.card import Card


def pimc_decision(
    *,
    true_pos:  Any,
    contract:  str,
    n_samples: int,
    seed:      int,
    voids:     Optional[Dict[int, FrozenSet[str]]] = None,
    must_hold: Optional[Dict[int, List[Card]]] = None,
) -> Tuple[Optional[Card], Dict[Card, float]]:
    """Run PIMC for the player on move at ``true_pos``.

    ``voids`` is an optional per-player suit-void map from observed play
    (track with :class:`solvers.determinize.Voids`). ``must_hold`` is an
    optional override for the contract's auction-derived constraints
    (pass ``None`` to use the registered rule, ``{}`` to disable).

    Returns ``(chosen_card, score_avg)`` where ``score_avg`` is the mean
    god-solver value per candidate across the ``n_samples`` worlds.
    """
    rng    = random.Random(seed)
    viewer = _pis.current_player(true_pos)

    iset = _det.build_info_set(
        true_pos, viewer, contract,
        voids=voids, must_hold=must_hold,
    )

    score_sums:  Dict[Card, float] = {}
    score_count: Dict[Card, int]   = {}

    for _ in range(n_samples):
        hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            sample_pos = _pis.clone_with_hands_and_talon(true_pos, hands, talon)
        else:
            sample_pos = _pis.clone_with_hands(true_pos, hands)

        values = _pis.solve_all(sample_pos, contract=contract)
        for card, v in values.items():
            score_sums[card]  = score_sums.get(card, 0.0) + float(v)
            score_count[card] = score_count.get(card, 0) + 1

    if not score_sums:
        return None, {}

    averaged = {c: score_sums[c] / score_count[c] for c in score_sums}
    best     = max(averaged, key=lambda c: averaged[c])
    return best, averaged
