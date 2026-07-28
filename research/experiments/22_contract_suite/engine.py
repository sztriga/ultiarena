"""One configurable engine: a contract is a weight config on the SAME multi
solver + the SAME oracle. No per-contract solvers.

A row/config turns components on/off:
  parti        → parti_pts=1                       (reproduces contract='parti')
  +silent_ulti → silent_ulti=2
  +silent_100  → score_geq_100 = silent-100 weight (gated by held marriage)

Silent-100 weight (corrected rule, milan 2026-06-14): a naked marriage scores 0,
so crossing 100 earns the FULL silent value. The live solver `score_geq_100`
tests TOTAL >= 100; for a SINGLE held marriage total = card_pts + that marriage,
so the threshold fires EXACTLY at the right card-points (40 → card_pts>=60,
20 → card_pts>=80). For MULTIPLE marriages it fires early (would need a
configurable threshold) — rare; we target the better silent-100 and flag it.
"""
from __future__ import annotations

from ulti.scoring.oracle import BidSet
from recipe import sol_marriages   # (has_40, has_20) from a 10-card hand

SILENT_40_100 = 2.0    # value of a silent 40-100 (card_pts>=60); piros doubles
SILENT_20_100 = 4.0    # value of a silent 20-100 (card_pts>=80); worth more
SILENT_DM     = 3.0    # silent colored durchmars (sweep all 10 tricks); piros
                       # doubles. = half the bid duri (6). milan 2026-06-15.
# BID 40-100 / 20-100: declared, so reaching 100 swings made↔bukott. The bid
# values are 4 / 8 (made; bukott = -that). The score_geq_100 weight = the SWING
# (made - bukott = 2x the bid value), since the -bid floor is a sunk constant.
# Policy is scale-invariant under piros (oracle doubles every component), so the
# same weight works red or not. Single declared marriage → solver's total>=100
# threshold is exact (total = card_pts + that marriage).
BID_40_100_SWING = 8.0    # made +4 / bukott -4
BID_20_100_SWING = 16.0   # made +8 / bukott -8


def silent_100_weight(has_40, has_20):
    """score_geq_100 weight = value of the best achievable silent 100.
    Single marriage → the solver's total>=100 threshold is exact. Both → returns
    the higher value (20-100) and targets it; threshold then fires early
    (approx) — handled exactly by the oracle at scoring time regardless."""
    if has_20:
        return SILENT_20_100
    if has_40:
        return SILENT_40_100
    return 0.0


def solver_weights(*, parti=True, silent_ulti=False, silent_100=False,
                   silent_dm=False, bid=None, has_40=False, has_20=False):
    """Weight vector for ``set_multi_weights`` (always contract='multi').

    ``bid`` = '40_100' or '20_100' prices a DECLARED 40-100/20-100 (the soloist
    plays to reach 100 at bid stakes). It takes the score_geq_100 slot, so it is
    mutually exclusive with ``silent_100`` for the declared marriage."""
    w = {}
    if parti:
        w['parti_pts'] = 1.0
    if silent_ulti:
        w['silent_ulti'] = 2.0
    if silent_100:
        w['score_geq_100'] = silent_100_weight(has_40, has_20)
    if silent_dm:
        # Reward a colored sweep (all 10 tricks). A sweep also banks parti +
        # silent ulti (win trick 10 with the 7) + the 100 (all card points), so
        # the objective arranges all of them together when a duri is on.
        w['silent_durchmars'] = SILENT_DM
    if bid == '40_100':
        w['score_geq_100'] = BID_40_100_SWING
    elif bid == '20_100':
        w['score_geq_100'] = BID_20_100_SWING
    return w


def oracle_bid(*, piros=False, bid=None):
    """BidSet for scoring. ``bid`` = '40_100'/'20_100' declares the 100 contract
    so the oracle scores it ±4 / ±8 (made/bukott); otherwise a plain parti deal
    where the oracle auto-credits any silent ulti / silent 100 / silent duri."""
    return BidSet(parti=True, piros=piros,
                  forty_hundred=(bid == '40_100'),
                  twenty_hundred=(bid == '20_100'))
