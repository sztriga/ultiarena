"""God-provider — the perfect-INFO perception ceiling (exp 25, Phase 1).

Bids using TRUE double-dummy makeability (it sees every hand) instead of net
estimates, then runs the SAME composer + auction as the net agent. So the only
thing that differs is perception. The strength gap

    net-agent  →  god-agent

is the MAXIMUM strength buyable by better perception — an UPPER bound. It answers
"can retraining the nets help?" from above: small gap ⇒ perception isn't the
bottleneck; large gap ⇒ real headroom. (The *trainable* slice is the gap to a
perfect MARGINAL estimator = high-N PIMC; god − PIMC = the irreducible info gap.)

Cost: 7 god solves per (trump × discard); heavy but fine at moderate N.
"""
from __future__ import annotations

import os
import sys

_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for p in (_E23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from ladder import GPTable                 # noqa: E402
from bidder import BaseProbs               # noqa: E402
from recipe_local import sol_marriages     # noqa: E402
from auction import _best_pickup           # noqa: E402


def god_base_probs(hand10, trump, d1, d2, talon):
    """TRUE (0/1) double-dummy makeability of each base event, given all hands."""
    from ulti.solvers import pis
    from ulti.eval.pimc_matchup import god_says_soloist_wins
    from trickster._solver_core import set_multi_weights

    has40, has20 = sol_marriages(hand10, trump)

    def gsw(contract, t, restrict=None, multi=False):
        if multi:
            set_multi_weights(score_geq_100=1.0)
        pos = pis.build_position(
            hands=[list(hand10), list(d1), list(d2)], soloist=0, leader=0,
            contract=("parti" if multi else contract), trump=t, talon=list(talon),
            declare_marriages=(multi or t is not None), marriage_restrict=restrict)
        return 1.0 if god_says_soloist_wins(
            pos, contract=("multi" if multi else contract)) else 0.0

    return BaseProbs(
        p_parti=gsw("parti", trump),
        p_ulti=gsw("ulti", trump),
        p_reach100_40=(gsw(None, trump, restrict="40", multi=True) if has40 else 0.0),
        p_reach100_20=(gsw(None, trump, restrict="20", multi=True) if has20 else 0.0),
        p_duri_colored=gsw("durchmars", trump),
        p_betli=gsw("betli", None),
        p_duri_colorless=gsw("durchmars", None),
        has_40=has40, has_20=has20, trump_is_hearts=(trump == "hearts"),
    )


def make_god_bid_fn():
    """Factory (picklable for fork workers) → the perfect-info bid_fn."""
    gp = GPTable()

    def bid_fn(cards, current, others=None):
        d1, d2 = others                              # the other two seats' hands
        def pf(hand10, trump, talon):
            return god_base_probs(hand10, trump, d1, d2, talon)
        # god perception has no estimation noise → NO debias (plain max over discards)
        return _best_pickup(cards, pf, current, gp, pctl=None)

    return bid_fn
