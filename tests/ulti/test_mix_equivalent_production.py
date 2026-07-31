"""The anti-tell mixer never gives anything away, on the REAL play path.

tests/ulti/test_block_equivalence.py proves the block rule on synthetic positions. This
one proves it where it actually runs: full games through apps.api.play, with the real
bidder, real PIMC, the real exp36 betli-defense net and the real exploit soloist.

Every time ``_mix_equivalent`` swaps a card, we solve the TRUE position (perfect
information — cheating on purpose, offline) and require the original and the swapped
card to have exactly the same double-dummy value. A non-zero delta means the mixer
traded a good card for a worse one.

Note what this does NOT claim: swapping an equivalent card still changes the cards on
the table, so the rest of a *sampled* PIMC game plays out differently and a final score
can move either way. That is a re-roll, not a loss — the guarantee is per-decision, and
that is exactly what is asserted here.
"""
from __future__ import annotations

import os
import tempfile

import pytest

# These are synthetic games; keep them out of the real games DB.
os.environ.setdefault("GAMES_DB",
                      os.path.join(tempfile.gettempdir(), "ulti_harness_games.db"))

# Pin every knob that affects play BEFORE importing play.py (it reads env at import).
for _k, _v in {
    "FLOOR": "0.80", "DEBIAS_PCTL": "0.85", "DURI_TERIT_MULT": "0.3", "KONTRA": "1",
    "REBETLI_FLOOR": "0.90", "EXPLOIT": "1", "EXPLOIT_EPS": "0.15", "EXPLOIT_NW": "16",
    "EXPLOIT_FRAC": "0.10", "BETLI_REAL_BID": "1", "BETLI_DEF": "1",
    "REBETLI_REAL_BID": "1", "PLAY_PIMC_N": "16", "PLAY_KONTRA_NDET": "6",
    "MIX_EQUIV": "1",
}.items():
    os.environ[_k] = _v

from apps.api import ai_play as P                    # noqa: E402  (the mixer lives here since the play.py split)
from ulti.solvers import pis                         # noqa: E402
from ultisolver._solver_core import set_multi_weights  # noqa: E402

# A slice of the golden matrix — enough to exercise several contracts and both roles.
MATRIX = [(seat, seed) for seat in (0, 1, 2) for seed in (11, 12, 13)]
TOL = 1e-5


def test_every_mixed_card_is_value_identical(monkeypatch):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
    import capture

    seen = {"picks": 0, "mixed": 0, "checked": 0}
    original = P._mix_equivalent

    def audited(sess, play_idx, card):
        out = original(sess, play_idx, card)
        seen["picks"] += 1
        if out is None or card is None or out.id == card.id:
            return out
        seen["mixed"] += 1
        if sess.p_weights is not None:
            set_multi_weights(**sess.p_weights)
        values = pis.solve_all(sess.p_pos, contract=sess.p_solve_contract)
        if card in values and out in values:
            seen["checked"] += 1
            assert values[out] == pytest.approx(values[card], abs=TOL), (
                f"mixer swapped {card} for {out} but they solve to "
                f"{values[card]} vs {values[out]} — that is real value thrown away")
        return out

    monkeypatch.setattr(P, "_mix_equivalent", audited)
    for seat, seed in MATRIX:
        capture._capture(seat, seed, "double" if seed % 2 == 0 else "pass")

    assert seen["mixed"] > 0, "mixer never fired — the test proved nothing"
    assert seen["checked"] == seen["mixed"], "some swaps could not be verified"


def test_mixer_can_be_reverted():
    """MIX_EQUIV=0 must return the pick untouched (the revert switch works)."""
    monkey = P._MIX_EQUIV
    try:
        P._MIX_EQUIV = False
        sentinel = object()
        assert P._mix_equivalent(None, 1, sentinel) is sentinel
    finally:
        P._MIX_EQUIV = monkey
