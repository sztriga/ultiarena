"""PIMC + V: 1-ply value-only defender.

Like ``pimc.py`` but the per-world evaluator is a learned value function
``value_fn(pos) -> float`` (defender-perspective: +1 = defenders win)
instead of the god solver. For each legal action ``a`` at the on-move
defender's turn:

  1. Sample ``n_samples`` worlds from the info-set.
  2. In each world, apply ``a`` to get next state ``s'``.
  3. Score ``a`` ← mean over K worlds of ``value_fn(s')``.
  4. Return argmax_a.

This is Step 1 of vnet/betli/PLAN.md.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from solvers import determinize as _det
from solvers import pis as _pis
from ulti.card import Card


def pimc_v_decision(
    *,
    true_pos:  Any,
    contract:  str,
    n_samples: int,
    value_fn:  Callable[[Any, int], float],
    seed:      int,
    voids:     Optional[Dict[int, FrozenSet[str]]] = None,
    must_hold: Optional[Dict[int, List[Card]]] = None,
    sol_prior_alpha: Optional[float] = None,
) -> Tuple[Optional[Card], Dict[Card, float]]:
    """1-ply value-only defender decision.

    ``value_fn(pos_after, viewer)`` is called for each (sampled world,
    candidate action). It must return the defender-perspective value of
    the post-action state from ``viewer``'s POV (a defender index in
    {1,2}). +1 means defenders win, −1 means soloist wins.

    Returns ``(chosen_card, score_avg)`` where ``score_avg`` is the mean
    value of each candidate across ``n_samples`` worlds.
    """
    rng    = random.Random(seed)
    viewer = _pis.current_player(true_pos)
    if viewer == 0:
        raise ValueError("pimc_v_decision is defender-only (viewer must be 1 or 2)")

    iset = _det.build_info_set(
        true_pos, viewer, contract,
        voids=voids, must_hold=must_hold,
    )

    score_sums:  Dict[Card, float] = {}
    score_count: Dict[Card, int]   = {}

    for _ in range(n_samples):
        if sol_prior_alpha is not None:
            hands, talon = _det.sample_world(iset, rng, sol_prior_alpha=sol_prior_alpha)
        else:
            hands, talon = _det.sample_world(iset, rng)
        if iset.talon_known is None:
            sample_pos = _pis.clone_with_hands_and_talon(true_pos, hands, talon)
        else:
            sample_pos = _pis.clone_with_hands(true_pos, hands)

        legal_here = _pis.legal_actions(sample_pos)
        for a in legal_here:
            child = sample_pos.clone()
            _pis.apply_move(child, a)
            v = float(value_fn(child, viewer))
            score_sums[a]  = score_sums.get(a, 0.0) + v
            score_count[a] = score_count.get(a, 0) + 1

    if not score_sums:
        return None, {}

    averaged = {c: score_sums[c] / score_count[c] for c in score_sums}
    best     = max(averaged, key=lambda c: averaged[c])
    return best, averaged
