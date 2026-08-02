"""Per-session serialization (engine._hold) — one game is a critical section.

Every play route mutates its Session under check-then-act (validate turn → apply), so
two requests on the SAME game (two tabs, a double-fired click) must serialize, while
different games must never wait on each other. The first test pins the mechanism
deterministically; the second races a real endpoint the way a double-click would."""
from __future__ import annotations

import threading
import time

from fastapi import HTTPException

from apps.api import play as P
from apps.api.engine import Session, _hold, _sessions, _sessions_lock


def _register(seat=0, seed=1) -> Session:
    sess = Session(seat=seat, seed=seed)
    with _sessions_lock:
        _sessions[sess.id] = sess
    return sess


def _drop(*ids):
    with _sessions_lock:
        for gid in ids:
            _sessions.pop(gid, None)


def test_hold_serializes_one_game_and_parallelizes_two():
    a, b = _register(seed=101), _register(seed=102)
    try:
        events = []
        a_held = threading.Event()

        def holder():
            with _hold(a.id):
                events.append("A-in")
                a_held.set()
                time.sleep(0.25)
                events.append("A-out")

        def same_game():
            a_held.wait(2.0)
            with _hold(a.id):
                events.append("B-in")

        def other_game():
            a_held.wait(2.0)
            t0 = time.monotonic()
            with _hold(b.id):
                events.append(("C-waited", time.monotonic() - t0))

        threads = [threading.Thread(target=f) for f in (holder, same_game, other_game)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5.0)

        # same game: B could only enter after A released
        assert events.index("B-in") > events.index("A-out")
        # different game: C entered while A was still holding (no cross-game wait)
        c = next(e for e in events if isinstance(e, tuple))
        assert c[1] < 0.2, f"different game waited {c[1]:.3f}s on another game's lock"
    finally:
        _drop(a.id, b.id)


def test_double_fired_request_has_exactly_one_winner():
    # Seat 1: the AI forehand acts, then the human is in the AUCTION step, where
    # Felveszem (pickup) is legal exactly once. Fire it twice concurrently — the
    # double-click case. Serialized, the loser must see the already-mutated state
    # and get a clean 400; it must never double-apply.
    snap = P.play_new(P.NewRequest(seat=1, seed=13))
    gid = snap["game_id"]
    try:
        results = []
        barrier = threading.Barrier(2)

        def fire():
            barrier.wait(2.0)
            try:
                P.play_pickup(P.PickupRequest(game_id=gid))
                results.append("ok")
            except HTTPException as e:
                results.append(e.status_code)

        threads = [threading.Thread(target=fire) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10.0)

        assert sorted(results, key=str) == [400, "ok"], results
        with _sessions_lock:
            sess = _sessions[gid]
        assert sess.a_awaiting_bid and sess.a_picked_up   # applied exactly once, cleanly
    finally:
        _drop(gid)
