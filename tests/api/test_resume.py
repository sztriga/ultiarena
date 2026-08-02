"""Resume (/play/mine + /play/state) — the splash's "Folyamatban lévő játékaid" list.

Identity model under test: device_id (a browser-generated uuid) tags games at
creation and is the ONLY key the listing answers to — one device never sees another
device's games, malformed ids are rejected at the schema, and resuming is a plain
full-snapshot state fetch. Seat-0 games are dealt without any AI work, so these run
in milliseconds."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from apps.api import play as P
from apps.api.engine import _sessions, _sessions_lock

DEV_A = "aaaaaaaa-1111-2222-3333-444444444444"
DEV_B = "bbbbbbbb-1111-2222-3333-444444444444"


def _drop(*snaps):
    with _sessions_lock:
        for s in snaps:
            _sessions.pop(s["game_id"], None)


def test_mine_lists_only_this_device_newest_first_and_resumes():
    s1 = P.play_new(P.NewRequest(seat=0, seed=201, device_id=DEV_A))
    s2 = P.play_new(P.NewRequest(seat=0, seed=202, device_id=DEV_A))
    s3 = P.play_new(P.NewRequest(seat=0, seed=203, device_id=DEV_B))
    try:
        games = P.play_mine(P.MineRequest(device_id=DEV_A))["games"]
        assert {g["game_id"] for g in games} == {s1["game_id"], s2["game_id"]}
        assert all(g["phase"] == "bid" and g["contract"] is None for g in games)

        # ordering follows activity: touching s1 (a state fetch) makes it newest
        P.play_state(P.StateRequest(game_id=s1["game_id"]))
        games = P.play_mine(P.MineRequest(device_id=DEV_A))["games"]
        assert games[0]["game_id"] == s1["game_id"]

        # resume is a plain full-snapshot fetch of the listed id
        snap = P.play_state(P.StateRequest(game_id=s2["game_id"]))
        assert snap["game_id"] == s2["game_id"]
        assert snap["phase"] == "bid"
    finally:
        _drop(s1, s2, s3)


def test_unknown_device_sees_nothing():
    s1 = P.play_new(P.NewRequest(seat=0, seed=204, device_id=DEV_A))
    try:
        assert P.play_mine(P.MineRequest(device_id=DEV_B))["games"] == []
    finally:
        _drop(s1)


def test_untagged_games_are_not_listable():
    # No device_id (an old client, or storage-less private mode) → playable but not
    # resumable; it must never leak into anyone's listing.
    s1 = P.play_new(P.NewRequest(seat=0, seed=205))
    try:
        assert P.play_mine(P.MineRequest(device_id=DEV_A))["games"] == []
    finally:
        _drop(s1)


@pytest.mark.parametrize("bad", ["", "short", "x" * 65, "not valid!", "<script>"])
def test_malformed_device_ids_are_rejected_at_the_schema(bad):
    with pytest.raises(ValidationError):
        P.MineRequest(device_id=bad)
    with pytest.raises(ValidationError):
        P.NewRequest(seat=0, device_id=bad)
