"""The deployed bidder, defined ONCE.

Reproducing "what actually ships" was, until now, a matter of remembering to pass four
arguments. `apps/api/engine.py` passed them; every research harness constructed
`net_bid_fn(NetProvider())` instead and silently got a DIFFERENT bidder:

    isotonic calibration   OFF   (NetProvider(calibrate=False) is the default)
    exp37 realistic betli  OFF   (betli_real defaults to the module global, False)
    exp39 rebetli          OFF   (likewise)

An uncalibrated bidder with the betli heads reverted is not the frontier, so a table
measured that way describes a configuration nobody plays. It cost a 35-minute run on
2026-08-02, and it would have cost a wrong headline if the terített-betli row had not
looked implausible.

So: one factory, used by the app AND by every experiment. If the deployment profile
changes, it changes here and the harnesses follow automatically. Knobs still come from
ulti.config, so an experiment can still override any single one deliberately — the point
is that the DEFAULT is the deployed thing rather than an accident.
"""
from __future__ import annotations

import os
from typing import Optional

from ulti.config import env_bool

# Deployed promotion gates — the ONE place these are read; the app builds its bidder
# through frontier_bid_fn and inherits them.
# exp37 (promoted 2026-07-24): PLAIN betli is scored by a REALISTIC-defense make-prob
# (not the god double-dummy) → bids the cheap 5p betli that makes vs imperfect defense.
# exp39 (promoted 2026-07-25): the same realistic prob extends to REBETLI (the hidden
# 10p doubling), gated behind REBETLI_FLOOR=0.90. Set the env to 0 to revert either.
BETLI_REAL_BID = env_bool("BETLI_REAL_BID", True)
REBETLI_REAL_BID = env_bool("REBETLI_REAL_BID", True)


def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    return d


def frontier_provider(**kw):
    """NetProvider exactly as the app builds it: isotonic calibration ON, and the exp37
    realistic-betli head available (whether it is USED is the bidder's gate)."""
    from ulti.bidding.provider import NetProvider
    kw.setdefault("calibrate", True)
    kw.setdefault("betli_real_dir", os.path.join(_repo_root(), "models", "ulti", "betli"))
    return NetProvider(**kw)


def frontier_bid_fn(provider=None, **kw):
    """The deployed bid function. Extra kwargs pass through to `net_bid_fn`, so an
    experiment can vary one knob against a faithful baseline (e.g. `pickup_model=`)."""
    from ulti.bidding.auction import net_bid_fn
    kw.setdefault("betli_real", BETLI_REAL_BID)
    kw.setdefault("rebetli_real", REBETLI_REAL_BID)
    return net_bid_fn(provider or frontier_provider(), **kw)
