"""Baseline bidder #1: the exp-23 net+composer (7 calibrated base-event heads →
composer → debias-0.80 talon search). This is the reference the loop tries to beat.

`make_net_bid_fn()` is a module-level factory (picklable for fork workers) that
builds a fresh NetProvider and returns the auction's bid_fn. A new architecture =
a new make_*_bid_fn with the same signature; harness.evaluate is unchanged.
"""
from __future__ import annotations

import os
import sys

_E23 = "/Users/milansimity/Cuccok/kodok/oldtawer/experiments/23_bidding_integration"
for p in (_E23, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

CALIBRATE = os.environ.get("CALIBRATE", "1") not in ("0", "false", "no")


def make_net_bid_fn():
    from provider import NetProvider
    from auction import net_bid_fn
    return net_bid_fn(NetProvider(calibrate=CALIBRATE))


# The old deployed bidder's expressible contracts, as ladder-rung indices:
# piros parti(0), ulti(2), betli(3), colorless duri(4), piros ulti(9), rebetli(10).
OLD4 = {0, 2, 3, 4, 9, 10}


def make_old4_bid_fn():
    """Same nets/composer, but restricted to the 4-contract set the deployed
    exp17/20 bidder could express — the h2h ablation for 'value of the new contracts'."""
    from provider import NetProvider
    from auction import net_bid_fn
    return net_bid_fn(NetProvider(calibrate=CALIBRATE), allowed=OLD4)
