"""PIMC-provider — the perfect-MARGINAL perception ceiling (exp 25, Phase 1b).

Estimates P(make | own hand) by sampling opponent hands consistent with the
soloist's own hand (own-hand info ONLY — it does NOT peek at `others`, so no
cheating) and god-solving each determinization. This is the ceiling a perfectly-
trained net could reach (a net just approximates this marginal). So:
    net → PIMC = TRAINABLE headroom (what better nets can buy)
    PIMC → god = IRREDUCIBLE info gap (can't be trained away)

Uses debias 0.80 like the net agent → the h2h(pimc, net) isolates purely the
perception SOURCE (pimc marginal vs net estimate). N_DET is the sample count
(higher = closer to the true marginal; 8-10 is a rough lower bound on the ceiling).
"""
from __future__ import annotations

import os
import random
import sys

_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
_E14 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/14_minigame_bid_eval"
for p in (_E23, _E14, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from ladder import GPTable                 # noqa: E402
from bidder import BaseProbs               # noqa: E402
from recipe_local import sol_marriages     # noqa: E402
from auction import _best_pickup           # noqa: E402
from _lib import fresh_deck                # noqa: E402

N_DET = int(os.environ.get("N_DET", "8"))


def pimc_base_probs(hand10, trump, talon, n_det, seed):
    from solvers import pis
    from eval.pimc_matchup import god_says_soloist_wins
    from trickster._solver_core import set_multi_weights

    has40, has20 = sol_marriages(hand10, trump)
    held = {c.id for c in hand10} | {c.id for c in talon}
    unknown = [c for c in fresh_deck() if c.id not in held]      # 20 cards
    rng = random.Random(seed)

    def gsw(contract, d1, d2, t, restrict=None, multi=False):
        if multi:
            set_multi_weights(score_geq_100=1.0)
        pos = pis.build_position(
            hands=[list(hand10), d1, d2], soloist=0, leader=0,
            contract=("parti" if multi else contract), trump=t, talon=list(talon),
            declare_marriages=(multi or t is not None), marriage_restrict=restrict)
        return 1.0 if god_says_soloist_wins(
            pos, contract=("multi" if multi else contract)) else 0.0

    acc = dict(parti=0.0, ulti=0.0, r40=0.0, r20=0.0, dc=0.0, betli=0.0, d0=0.0)
    for _ in range(n_det):
        cards = unknown[:]
        rng.shuffle(cards)
        d1, d2 = cards[:10], cards[10:20]
        acc["parti"] += gsw("parti", d1, d2, trump)
        acc["ulti"] += gsw("ulti", d1, d2, trump)
        if has40:
            acc["r40"] += gsw(None, d1, d2, trump, restrict="40", multi=True)
        if has20:
            acc["r20"] += gsw(None, d1, d2, trump, restrict="20", multi=True)
        acc["dc"] += gsw("durchmars", d1, d2, trump)
        acc["betli"] += gsw("betli", d1, d2, None)
        acc["d0"] += gsw("durchmars", d1, d2, None)

    n = float(n_det)
    return BaseProbs(
        p_parti=acc["parti"] / n, p_ulti=acc["ulti"] / n,
        p_reach100_40=acc["r40"] / n, p_reach100_20=acc["r20"] / n,
        p_duri_colored=acc["dc"] / n, p_betli=acc["betli"] / n,
        p_duri_colorless=acc["d0"] / n, has_40=has40, has_20=has20,
        trump_is_hearts=(trump == "hearts"))


def make_pimc_bid_fn():
    gp = GPTable()

    def bid_fn(cards, current, others=None):        # ignores `others` (no cheating)
        def pf(hand10, trump, talon):
            seed = sum(c.id for c in hand10) * 131 + sum(c.id for c in talon) + 7
            return pimc_base_probs(hand10, trump, talon, N_DET, seed)
        return _best_pickup(cards, pf, current, gp)  # debias 0.80, like the net

    return bid_fn
