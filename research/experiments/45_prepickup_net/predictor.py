"""exp45 — the trained pre-pickup model, as a `pickup_model` for ulti.bidding.auction.

`net_bid_fn(..., pickup_model=fn)` calls `fn(hand10, current_rung) -> float`, the
predicted EV of the game you would end up announcing if you took the talon up. The
bidder compares that to the threshold (the −2 pass penalty when opening, the value of
defending when overcalling) and only then may it look at the talon.

The predictor is BLIND by construction — it is handed ten cards and a public rung, and
there is nowhere for a talon to enter. `datagen.assert_blind()` proves the featuriser
depends on nothing else.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

MODEL = os.path.join(_HERE, "prepickup_model.joblib")


def _exp45_featurize():
    """exp43 and exp45 both ship a `datagen.py`, and the gate puts both directories on
    sys.path — so a plain `from datagen import featurize` silently binds to whichever
    landed first. Load THIS experiment's module by absolute path instead."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp45_datagen", os.path.join(_HERE, "datagen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.featurize


def load_pickup_model(path: str = MODEL, provider=None, gp=None):
    """→ fn(hand10, current_rung) -> predicted post-pickup EV. Raises if not trained."""
    import joblib
    from ulti.bidding.ladder import GPTable
    from ulti.bidding.frontier import frontier_provider
    featurize = _exp45_featurize()

    blob = joblib.load(path)
    model, names = blob["model"], blob["names"]
    prov = provider or frontier_provider()
    table = gp or GPTable()

    def predict(hand10, current):
        cur_ix = -1 if current is None else int(current.index)
        f = featurize(list(hand10), cur_ix, prov, table)
        x = np.asarray([[f[k] for k in names]], dtype=np.float32)
        return float(model.predict(x)[0])

    return predict


def blind_ev_model(provider=None, gp=None, floor=0.0):
    """The INCUMBENT, in the same interface: the raw blind EV, uncorrected. This is what
    the cheat-free bidder does today, and it is the thing the gate has to beat."""
    from ulti.bidding.auction import _blind_best
    from ulti.bidding.ladder import GPTable
    from ulti.bidding.frontier import frontier_provider

    prov = provider or frontier_provider()
    table = gp or GPTable()

    def predict(hand10, current):
        b = _blind_best(list(hand10), lambda h, t, tal: prov.base_probs(h, t),
                        current, table, floor=floor)
        return -99.0 if b is None else float(b[0])

    return predict
