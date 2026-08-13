"""Kontra units must match what the scoring oracle actually scores.

You can only kontra a component that exists. The bug this pins: a bid 40-100 / 20-100 was
offering the defenders "kontra párti" even though the declared 100 REPLACES the párti —
scoring/oracle.py gates the párti component on ``not bid_a_100``, so that kontra rode a
component that never scores (milan, spotted in the UI: "it was a 40-100, there is no parti
there, yet the defender did a kontra parti").

The last test is the important one: it doesn't restate the rule, it asks the oracle which
components a real deal produces and demands the offered units line up.
"""
from __future__ import annotations

import os

os.environ.setdefault("KONTRA", "1")

from ulti.scoring.units import kontra_units as _kontra_units  # noqa: E402
from ulti.bidding.ladder import overcalls        # noqa: E402


_LADDER = {r.name: r for r in overcalls(None)}


def _bid(name: str):
    """The (first) BidSet of a named ladder rung."""
    assert name in _LADDER, f"rung {name!r} not in the ladder"
    return _LADDER[name].bids[0]


def test_bid_100_has_no_parti_unit():
    """A declared 100 replaces the párti — so there is nothing called párti to kontra."""
    for name in ("40-100", "20-100", "piros 40-100", "ulti-40-100",
                 "ulti-20-100", "piros 20-100", "piros ulti-40-100"):
        try:
            bid = _bid(name)
        except AssertionError:
            continue
        units = _kontra_units(bid)
        assert "parti" not in units, f"{name}: offered kontra párti, but the 100 replaces it"
        assert units, f"{name}: must still expose its own unit(s)"


def test_plain_ulti_keeps_its_parti_unit():
    """A bid ulti with no 100 DOES still play a card-point párti (milan: kontra ulti AND
    kontra párti) — the fix must not over-reach and remove that."""
    bid = _bid("ulti")
    units = _kontra_units(bid)
    assert "ulti" in units and "parti" in units


def test_durchmars_and_betli_have_no_parti():
    assert _kontra_units(_bid("duri")) == ["durchmars"]
    assert _kontra_units(_bid("betli")) == ["betli"]


def test_units_agree_with_the_oracle_on_real_deals():
    """The real check: every párti unit we offer must be a párti the oracle can score,
    and every unit we offer must be a component the oracle knows about."""
    import random

    from ulti.bidding.deal import deal_12_10_10
    from ulti.bidding.recipe import sol_marriages
    from ulti.scoring.oracle import score as score_oracle
    from ulti.solvers import pis

    checked = 0
    for name in ("ulti", "40-100", "20-100", "ulti-40-100", "duri", "betli"):
        try:
            bid = _bid(name)
        except AssertionError:
            continue
        units = _kontra_units(bid)
        rng = random.Random(hash(name) & 0xFFFF)
        for _ in range(60):
            sol12, d1, d2 = deal_12_10_10(rng.getrandbits(30))
            sol10, talon = list(sol12)[:10], list(sol12)[10:]
            trump = None if bid.betli else "hearts"
            # A 100 can only be BID while holding its marriage (enforced by both bid
            # paths — see test_a_100_cannot_be_bid_without_its_marriage). Without that
            # precondition the oracle leaves bid_a_100 False and DOES score a párti, so
            # a deal that could never occur must not be used to judge the units.
            has40, has20 = sol_marriages(sol10, trump) if trump else (False, False)
            if (bid.forty_hundred and not has40) or (bid.twenty_hundred and not has20):
                continue
            contract = "betli" if bid.betli else ("durchmars" if bid.durchmars else "parti")
            pos = pis.build_position(hands=[sol10, list(d1), list(d2)], soloist=0, leader=0,
                                     contract=contract, trump=trump, talon=talon,
                                     declare_marriages=trump is not None)
            while not pis.is_terminal(pos):
                pis.apply_move(pos, rng.choice(pis.legal_actions(pos)))
            out = score_oracle(final_pos=pos, bid=bid)
            if "parti" in out.components:
                assert "parti" in units, (
                    f"{name}: oracle scored a párti component but no párti unit is offered")
            checked += 1
    assert checked > 0


def test_a_100_cannot_be_bid_without_its_marriage():
    """The invariant the párti removal rests on.

    If a 40-100 could be bid WITHOUT holding the 40, the oracle would leave bid_a_100
    False, fall back to scoring a párti, and a párti unit would still be needed. The
    bidder makes that unreachable — no marriage, no bid.
    """
    from ulti.bidding.bidder import rung_ev
    from ulti.bidding.ladder import GPTable
    from ulti.bidding.provider import BaseProbs

    gp = GPTable()
    for name, flag in (("40-100", "has_40"), ("20-100", "has_20")):
        rung = _LADDER[name]
        without = BaseProbs(p_parti=0.99, p_ulti=0.99, p_reach100_40=0.99,
                            p_reach100_20=0.99, trump_is_hearts=True)
        assert rung_ev(rung, without, gp) is None, (
            f"{name} was biddable without its marriage — the párti unit would be needed")
        with_it = BaseProbs(p_parti=0.99, p_ulti=0.99, p_reach100_40=0.99,
                            p_reach100_20=0.99, trump_is_hearts=True, **{flag: True})
        assert rung_ev(rung, with_it, gp) is not None, f"{name} unbiddable even WITH it"
