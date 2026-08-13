"""/api/pis/explore — the analysis board's "what if I had played THIS?" fork.

Two properties matter and neither had a test:

  * VALIDATION happens in the request thread, so a bad branch is a 400 rather than
    a worker crash (building a position and replaying moves never solves).
  * the SEARCH does not run here. The Cython solver holds the GIL for its whole
    duration, so an explore solved in the web process froze every other request in
    it; it goes to the worker pool like every other AI decision.
"""
from __future__ import annotations

import pytest

from apps.api import ai_pool
from apps.api import pis as P
from ulti.bidding.deal import deal_12_10_10


def _req(**kw):
    sol12, d1, d2 = deal_12_10_10(7)
    hands = [[c.id for c in sol12[:10]], [c.id for c in d1], [c.id for c in d2]]
    base = dict(hands=hands, soloist=0, starting_leader=0, total_tricks=10,
                moves=[], forced_card_id=hands[0][0], contract="betli", trump=None,
                build_contract="betli", talon=[c.id for c in sol12[10:]],
                declare_marriages=False, marriage_restrict=None, multi_weights=None)
    base.update(kw)
    return P.PisExploreRequest(**base)


def test_explore_returns_a_full_principal_variation():
    r = P.pis_explore(_req())
    assert r["verdict"] in ("soloist", "defenders")
    assert r["alt_start"] == 0
    # forced card first, then the continuation. NOT necessarily all 30 plies: the
    # solver stops once the outcome is settled (a betli ends the moment the soloist
    # takes a trick), so only bound it.
    assert 1 <= len(r["alt_pv"]) <= 30
    assert r["alt_pv"][0]["card"]["id"] == _req().forced_card_id
    for i, step in enumerate(r["alt_pv"]):
        assert step["trick_index"] == i // 3 and step["trick_position"] == i % 3
        assert step["card"]["id"] in step["legal_card_ids"]   # every PV move was legal


def test_explore_rejects_an_illegal_forced_card():
    """A card that isn't the mover's to play is a 400 from the route, not a crash
    inside a worker."""
    other_hand_card = _req().hands[1][0]
    with pytest.raises(Exception) as e:
        P.pis_explore(_req(forced_card_id=other_hand_card))
    assert getattr(e.value, "status_code", None) == 400


def test_explore_rejects_an_illegal_replayed_move():
    with pytest.raises(Exception) as e:
        P.pis_explore(_req(moves=[_req().hands[1][0]]))   # not the leader's card
    assert getattr(e.value, "status_code", None) == 400


def test_explore_does_not_solve_in_this_process(monkeypatch):
    """The whole point of the move: no search happens in the web process. If any
    solving stayed behind, one of these would fire."""
    from ulti.solvers import pis as pis_bridge

    def _boom(*a, **k):
        raise AssertionError("solved in the web process — this belongs in a worker")

    monkeypatch.setattr(pis_bridge, "principal_variation", _boom)
    monkeypatch.setattr(pis_bridge, "solve_best", _boom)
    monkeypatch.setattr(ai_pool, "run",
                        lambda op, job: {"continuation": [], "value": 3.0} if op == "explore"
                        else pytest.fail(f"unexpected op {op!r}"))
    r = P.pis_explore(_req(contract="parti", build_contract="parti", trump="hearts",
                           declare_marriages=True))
    assert r["value"] == 3.0 and r["verdict"] == "soloist"
