"""Shared fixture for the API tests.

Every HTTP-level test runs against the REAL app (apps.api.main) with temp DBs and
all in-memory state cleared — no test can pollute data/ or leak state into the
next one. This fixture existed as four near-identical copies before; keep it here
and only here."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api import apikeys, limits, live, recording, users
from apps.api.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(users, "_DB_PATH", str(tmp_path / "users.db"))
    monkeypatch.setattr(users, "_conn", None)
    monkeypatch.setattr(recording, "_DB_PATH", str(tmp_path / "games.db"))
    monkeypatch.setattr(recording, "_conn", None)
    monkeypatch.setattr(limits, "RATE_LIMIT_RPM", 0)   # tests share one client IP
    monkeypatch.setattr(live, "_chat_seq", 0)
    users._token_cache.clear()
    users._auth_fails.clear()
    apikeys._cache.clear()
    apikeys._hits.clear()
    live._members.clear()
    live._tables.clear()
    live._chat.clear()
    with TestClient(app) as c:
        yield c
    for m in (users, recording):
        if m._conn is not None:
            m._conn.close()
