"""Accounts (/auth/*) and the lobby (/live/*) — stage 1 of multiplayer.

Everything runs through a real FastAPI TestClient against a temp users DB, so the
whole chain is exercised: bearer tokens, presence-by-poll, the chat cursor, and
milan's table rules — free join / host kicks / invites / host inheritance."""
from __future__ import annotations

from apps.api import live, users


def _register(client, name, password="hunter22") -> dict:
    r = client.post("/api/auth/register", json={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _poll(client, hdr, after=0):
    r = client.post("/api/live/poll", json={"chat_after": after}, headers=hdr)
    assert r.status_code == 200, r.text
    return r.json()


# ── accounts ────────────────────────────────────────────────────────────────────

def test_register_login_roundtrip(client):
    _register(client, "milan", "jelszo123")
    r = client.post("/api/auth/login", json={"username": "milan", "password": "jelszo123"})
    assert r.status_code == 200 and r.json()["username"] == "milan"


def test_wrong_password_and_duplicate_name(client):
    _register(client, "milan", "jelszo123")
    assert client.post("/api/auth/login",
                       json={"username": "milan", "password": "rossz1"}).status_code == 401
    assert client.post("/api/auth/register",
                       json={"username": "MILAN", "password": "masikjelszo"}).status_code == 409


def test_hungarian_usernames_ok_junk_rejected(client):
    _register(client, "Öcsi_űrhajós")
    r = client.post("/api/auth/register", json={"username": "rossz név!", "password": "jelszo123"})
    assert r.status_code == 400


def test_token_required_for_live(client):
    assert client.post("/api/live/poll", json={"chat_after": 0}).status_code == 401
    assert client.post("/api/live/poll", json={"chat_after": 0},
                       headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_brute_force_budget(client):
    _register(client, "milan", "jelszo123")
    for _ in range(users._AUTH_MAX_FAILS):
        client.post("/api/auth/login", json={"username": "milan", "password": "rossz1"})
    r = client.post("/api/auth/login", json={"username": "milan", "password": "jelszo123"})
    assert r.status_code == 429            # even the RIGHT password waits now


# ── lobby: presence + chat ──────────────────────────────────────────────────────

def test_presence_and_chat_cursor(client):
    a, b = _register(client, "anna"), _register(client, "bela")
    assert set(_poll(client, a)["members"]) == {"anna"}     # bela hasn't polled yet
    _poll(client, b)
    assert set(_poll(client, a)["members"]) == {"anna", "bela"}

    client.post("/api/live/chat", json={"text": "szia!"}, headers=a)
    client.post("/api/live/chat", json={"text": "helló"}, headers=b)
    chat = _poll(client, a)["chat"]
    assert [(c["user"], c["text"]) for c in chat] == [("anna", "szia!"), ("bela", "helló")]
    # the cursor: polling after the first seq returns only the second message
    assert [c["text"] for c in _poll(client, a, after=chat[0]["seq"])["chat"]] == ["helló"]


def test_presence_expires_and_frees_seats(client, monkeypatch):
    a, b = _register(client, "anna"), _register(client, "bela")
    _poll(client, a); _poll(client, b)
    client.post("/api/live/table/create", headers=b)
    # bela vanishes: age his heartbeat past the TTL
    live._members[next(u for u, m in live._members.items()
                       if m["username"] == "bela")]["last_seen"] -= live.PRESENCE_TTL + 1
    view = _poll(client, a)
    assert view["members"] == ["anna"]
    assert view["tables"] == []            # his empty table dissolved with him


# ── tables: join / kick / invite / host inheritance ─────────────────────────────

def _三(client):                            # three seated players
    hdrs = [_register(client, n) for n in ("anna", "bela", "cili")]
    for h in hdrs:
        _poll(client, h)
    tid = client.post("/api/live/table/create", headers=hdrs[0]).json()["table_id"]
    return hdrs, tid


def test_join_fills_seats_and_fourth_is_refused(client):
    (a, b, c), tid = _三(client)
    d = _register(client, "dani"); _poll(client, d)
    assert client.post("/api/live/table/join", json={"table_id": tid}, headers=b).status_code == 200
    assert client.post("/api/live/table/join", json={"table_id": tid}, headers=c).status_code == 200
    assert client.post("/api/live/table/join", json={"table_id": tid}, headers=d).status_code == 409
    t = _poll(client, a)["tables"][0]
    assert t["full"] and [s["username"] for s in t["seats"]] == ["anna", "bela", "cili"]


def test_host_kicks_non_host_cannot(client):
    (a, b, _c), tid = _三(client)
    client.post("/api/live/table/join", json={"table_id": tid}, headers=b)
    bela_id = next(u for u, m in live._members.items() if m["username"] == "bela")
    anna_id = next(u for u, m in live._members.items() if m["username"] == "anna")
    assert client.post("/api/live/table/kick",
                       json={"table_id": tid, "user_id": anna_id}, headers=b).status_code == 403
    assert client.post("/api/live/table/kick",
                       json={"table_id": tid, "user_id": bela_id}, headers=a).status_code == 200
    seats = _poll(client, a)["tables"][0]["seats"]
    assert [s["username"] if s else None for s in seats] == ["anna", None, None]


def test_invite_flags_the_table_for_the_invitee(client):
    (a, b, _c), tid = _三(client)
    assert client.post("/api/live/table/invite",
                       json={"table_id": tid, "username": "BELA"}, headers=a).status_code == 200
    assert _poll(client, b)["tables"][0]["invited_me"] is True
    assert _poll(client, a)["tables"][0]["invited_me"] is False


def test_host_leaves_next_occupant_inherits(client):
    (a, b, _c), tid = _三(client)
    client.post("/api/live/table/join", json={"table_id": tid}, headers=b)
    client.post("/api/live/table/leave", json={"table_id": tid}, headers=a)
    t = _poll(client, b)["tables"][0]
    assert t["is_host"] and t["host"] == "bela"


def test_one_table_per_player(client):
    (a, _b, _c), _tid = _三(client)
    assert client.post("/api/live/table/create", headers=a).status_code == 400


def test_start_is_host_only(client):
    (a, b, _c), tid = _三(client)
    client.post("/api/live/table/join", json={"table_id": tid}, headers=b)
    assert client.post("/api/live/table/start",
                       json={"table_id": tid}, headers=b).status_code == 403
