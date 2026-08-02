"""The all-pass (dead deal) flow — passz is a SCORED round, not a silent re-deal.

milan's rule: pass sits on the bidding ladder like any contract, we just never play
it. Three passes (after the forehand's reclaim is declined) must end the round with
the scoring window: the forehand forfeits PASS_PENALTY per defender, the result box
appears (phase "passed"), the round lands in the match tally, and the next hand is
dealt by the same dealer. This regressed once already — the frontend's passed screen
and the golden driver's PASSED branch existed while the backend silently re-dealt."""
from __future__ import annotations

# ORDER MATTERS: the app must load FIRST. apps.api.engine applies the deployment
# profile (FLOOR=0.80 …) before the bidding library reads its knobs at import time;
# importing ulti.bidding directly here first would pin the research defaults
# (FLOOR=0.0 → the AI bids everything → an all-pass never happens).
from apps.api import play as P
from apps.api.auction_flow import _resolve_auction
from apps.api.engine import Session, _sessions, _sessions_lock

from ulti.bidding.auction import PASS_PENALTY  # noqa: E402  (after the app — see above)


def _drop(*gids):
    with _sessions_lock:
        for g in gids:
            _sessions.pop(g, None)


def test_dead_deal_scores_the_forehand_penalty():
    sess = Session(seat=0, seed=7)            # human IS the forehand → pays
    _resolve_auction(sess)                    # a_current is None → dead deal
    assert sess.phase == "passed"
    r = sess.result
    assert r["contract"] == "passz"
    assert r["seat_gp"] == [-2.0 * PASS_PENALTY, PASS_PENALTY, PASS_PENALTY]
    assert r["human_gp"] == -2.0 * PASS_PENALTY and not r["user_won"]
    assert sum(r["seat_gp"]) == 0.0           # zero-sum like every round


def test_dead_deal_from_a_defender_seat_gains():
    sess = Session(seat=1, seed=7)            # an AI forehand pays, the human collects
    _resolve_auction(sess)
    r = sess.result
    assert sess.phase == "passed"
    assert r["human_gp"] == PASS_PENALTY and r["user_won"]


def test_all_pass_reaches_the_scoring_window_through_the_real_routes():
    """Drive real deals as the forehand: pass the opening, and when both AI seats
    also pass, decline the reclaim — the snapshot must land on phase "passed" with
    the full result attached (this is what feeds the UI's scoring window)."""
    found = None
    for seed in range(1, 120):
        snap = P.play_new(P.NewRequest(seat=0, seed=seed))
        gid = snap["game_id"]
        with _sessions_lock:
            sess = _sessions[gid]
        discard_ids = [sess.a_hands[0][0].id, sess.a_hands[0][1].id]
        snap = P.play_pass(P.PassRequest(game_id=gid, discard_ids=discard_ids))
        if snap["phase"] == "bid" and sess.a_reclaim_offered:
            snap = P.play_pass(P.PassRequest(game_id=gid))      # decline the reclaim
            # resume onto the score screen: /play/state re-serves the same result
            again = P.play_state(P.StateRequest(game_id=gid))
            assert again["phase"] == "passed"
            assert again["result"]["contract"] == "passz"
            found = snap
            _drop(gid)
            break
        _drop(gid)
    assert found is not None, "no all-pass deal in 120 seeds — bidder change?"
    assert found["phase"] == "passed"
    r = found["result"]
    assert r["contract"] == "passz"
    assert r["human_gp"] == -2.0 * PASS_PENALTY                 # forehand pays
