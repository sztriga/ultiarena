"""Game recording (apps/api/recording.py) — the data-collection side of the deployment.

Two properties matter for the friends-test:
  * the should_record gate: in-process drivers (golden, pytest, tournaments) must never
    write — before the gate, test runs buried the real games 121 rows to 14
  * a recorded row round-trips with the human seat's client IP in the players JSON,
    which is how "who played what" is answered until real auth lands

All inserts go to a temp DB — the real data/games.db is never touched from tests."""
from __future__ import annotations

import json
import sqlite3

import pytest

from apps.api import recording


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the module at a scratch DB and reset its cached connection."""
    path = str(tmp_path / "games_test.db")
    monkeypatch.setattr(recording, "_DB_PATH", path)
    monkeypatch.setattr(recording, "_conn", None)
    yield path
    if recording._conn is not None:
        recording._conn.close()


def _rec(gid="abc123", ip="1.2.3.4"):
    return {
        "id": gid, "created_at": 1000.0, "seed": 42, "contract": "piros ulti",
        "trump": "hearts", "soloist_seat": 1, "human_seat": 0, "kontra_level": 0,
        "winner": "soloist", "made": True, "seat_gp": [-4, 8, -4],
        "players": [
            {"seat": 0, "kind": "human", "user_id": None, "ip": ip, "agent": None},
            {"seat": 1, "kind": "ai", "user_id": None, "ip": None, "agent": "frontier"},
            {"seat": 2, "kind": "ai", "user_id": None, "ip": None, "agent": "frontier"},
        ],
        "transcript": {"deal": {"hands": [[], [], []], "talon": []},
                       "auction": [], "plays": [], "kontra": {}, "marriages": []},
    }


# ── the gate: what gets recorded at all ─────────────────────────────────────────

def test_in_process_sessions_are_never_recorded():
    assert not recording.should_record(None)         # pre-guard session (shouldn't exist)
    assert not recording.should_record("local")      # golden / pytest / tournaments


def test_real_clients_are_recorded():
    assert recording.should_record("188.36.1.2")     # internet visitor (CF header)
    assert recording.should_record("127.0.0.1")      # milan on the dev UI


# ── the row: round-trip + identity ──────────────────────────────────────────────

def test_record_round_trips_with_client_ip(tmp_db):
    recording.record_game(_rec(ip="188.36.9.9"))
    con = sqlite3.connect(tmp_db)
    row = con.execute("SELECT contract, human_seat, players FROM games").fetchone()
    con.close()
    assert row[0] == "piros ulti"
    players = json.loads(row[2])
    assert players[row[1]]["ip"] == "188.36.9.9"     # human seat carries the client
    assert all(p["ip"] is None for p in players if p["kind"] == "ai")


def test_same_game_id_replaces_not_duplicates(tmp_db):
    recording.record_game(_rec())
    recording.record_game(_rec())                     # e.g. _finish re-entered
    con = sqlite3.connect(tmp_db)
    assert con.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    con.close()


def test_recording_failure_is_swallowed_but_visible(tmp_db, capsys):
    bad = _rec()
    del bad["contract"]                               # KeyError inside record_game
    recording.record_game(bad)                        # must not raise
    assert "NOT recorded" in capsys.readouterr().err


def test_db_is_wal(tmp_db):
    recording.record_game(_rec())
    con = sqlite3.connect(tmp_db)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    con.close()
