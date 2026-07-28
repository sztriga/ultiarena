"""Weight recipe for pricing silent 40-100 / 20-100 in the multi solver.

The multi-payoff solver (``contract='multi'`` + ``set_multi_weights``) already
has every primitive needed for the 40/20: the parti channel carries declared
marriage *points* (defenders always declare; soloist declares all held pairs),
and ``score_geq_100`` is a live component. Because a marriage holding is *static
at solve-root* (declared pre-play, soloist cannot discard into/out of a pair
mid-solve), the only policy-relevant part is the dynamic 100 threshold. We gate
``score_geq_100`` by the *marginal* GP of crossing 100 given the marriage held;
the flat holder GP is a constant added to EV outside the search.

This mirrors ``scoring.oracle`` exactly (GPTable defaults):
    hold 40, <100 → +2 (silent_forty)     ; ≥100 → +4 (forty_hundred_silent)
    hold 20, <100 → +1 (silent_twenty)    ; ≥100 → +2 (twenty_hundred_silent)
so the marginal of crossing 100 is (4-2)=2 for a 40 and (2-1)=1 for a 20.
"""
from __future__ import annotations

from ulti.card import SUITS

# Per-defender GP, matching scoring.oracle.GPTable defaults.
SILENT_40,  SILENT_40_100 = 2, 4
SILENT_20,  SILENT_20_100 = 1, 2

# Marginal GP of pushing the soloist's total to >= 100, per held marriage.
MARGIN_40 = SILENT_40_100 - SILENT_40   # 2
MARGIN_20 = SILENT_20_100 - SILENT_20   # 1


def sol_marriages(hand, trump):
    """(has_40, has_20) for a soloist 10-card ``hand`` (ulti.card.Card list).

    has_40 = holds king+upper of the trump suit.
    has_20 = holds king+upper of at least one non-trump suit.
    """
    kings  = {c.suit for c in hand if c.rank == 'king'}
    uppers = {c.suit for c in hand if c.rank == 'upper'}
    pairs  = kings & uppers
    has_40 = trump in pairs
    has_20 = any(s != trump for s in pairs)
    return has_40, has_20


def parti_multi_weights(has_40, has_20, *, silent_ulti=2.0):
    """Multi-solver weight vector for a parti soloist that should also chase
    the silent 40-100 / 20-100 when (and only when) it holds the marriage."""
    margin = 0.0
    if has_40:
        margin += MARGIN_40
    if has_20:
        margin += MARGIN_20
    return {
        'parti_pts':     1.0,
        'silent_ulti':   silent_ulti,
        'score_geq_100': margin,
    }


def base_holder_gp(has_40, has_20):
    """Constant GP/def the soloist banks just for *holding* the marriage,
    independent of play. It does not affect policy (same on every leaf); add it
    to the solved EV when aggregating at bid time."""
    g = 0.0
    if has_40:
        g += SILENT_40
    if has_20:
        g += SILENT_20
    return g
