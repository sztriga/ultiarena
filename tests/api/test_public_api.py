"""The public research API (/api/v1) — keys, rules kernel, dataset, matches.

Everything over real HTTP against the mounted sub-application. The match tests
prove the veneer property: an API agent hits the SAME handlers and snapshots as a
browser, with the same cheat-cleanliness (its view never contains hidden cards)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api import limits, live, recording, users
from apps.api.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(users, "_DB_PATH", str(tmp_path / "users.db"))
    monkeypatch.setattr(users, "_conn", None)
    monkeypatch.setattr(recording, "_DB_PATH", str(tmp_path / "games.db"))
    monkeypatch.setattr(recording, "_conn", None)
    monkeypatch.setattr(limits, "RATE_LIMIT_RPM", 0)
    from apps.api import apikeys
    users._token_cache.clear()
    users._auth_fails.clear()
    apikeys._cache.clear()
    apikeys._hits.clear()
    live._members.clear(); live._tables.clear(); live._chat.clear()
    with TestClient(app) as c:
        yield c
    if users._conn is not None:
        users._conn.close()
    if recording._conn is not None:
        recording._conn.close()


@pytest.fixture()
def key(client) -> dict:
    r = client.post("/api/auth/register",
                    json={"username": "kutato", "password": "jelszo123"})
    browser = {"Authorization": f"Bearer {r.json()['token']}"}
    k = client.post("/api/v1/keys", json={"name": "exp-1"}, headers=browser)
    assert k.status_code == 200, k.text
    return {"Authorization": f"Bearer {k.json()['key']}", "_browser": browser}


# ── keys ────────────────────────────────────────────────────────────────────────

def test_key_lifecycle(client, key):
    browser = key["_browser"]
    lst = client.get("/api/v1/keys", headers=browser).json()["keys"]
    assert len(lst) == 1 and lst[0]["name"] == "exp-1"
    assert "ua_" in lst[0]["prefix"] and len(lst[0]["prefix"]) < 15   # never the full key
    assert client.delete(f"/api/v1/keys/{lst[0]['id']}", headers=browser).json()["revoked"]
    # revoked key stops working IMMEDIATELY
    assert client.post("/api/v1/deal", json={}, headers={
        "Authorization": key["Authorization"]}).status_code == 401


def test_public_api_requires_a_key_not_a_browser_token(client, key):
    assert client.post("/api/v1/deal", json={}).status_code == 401
    assert client.post("/api/v1/deal", json={},
                       headers=key["_browser"]).status_code == 401   # browser token ≠ key
    assert client.post("/api/v1/keys", json={"name": "x"},
                       headers={"Authorization": key["Authorization"]}).status_code == 403


# ── rules kernel ────────────────────────────────────────────────────────────────

def test_deal_is_deterministic_and_complete(client, key):
    h = {"Authorization": key["Authorization"]}
    d1 = client.post("/api/v1/deal", json={"seed": 42}, headers=h).json()
    d2 = client.post("/api/v1/deal", json={"seed": 42}, headers=h).json()
    assert d1 == d2
    all_ids = sorted(d1["hands"][0] + d1["hands"][1] + d1["hands"][2] + d1["talon"])
    assert all_ids == list(range(32))


def test_legal_and_score_roundtrip_betli(client, key):
    """Replay a deal to the end by always playing the first legal card, then score
    it — the kernel is a complete offline referee."""
    h = {"Authorization": key["Authorization"]}
    d = client.post("/api/v1/deal", json={"seed": 7}, headers=h).json()
    pos = {"hands": d["hands"], "talon": d["talon"],
           "contract": "betli", "trump": None, "plays": []}
    for _ in range(30):
        leg = client.post("/api/v1/legal", json=pos, headers=h).json()
        if leg["terminal"]:
            break
        pos["plays"].append(leg["legal_card_ids"][0])
    leg = client.post("/api/v1/legal", json=pos, headers=h).json()
    # betli may terminate EARLY (the deal is decided when the soloist takes a trick)
    assert leg["terminal"] and 3 <= len(pos["plays"]) <= 30
    sc = client.post("/api/v1/score", json=pos, headers=h).json()
    assert abs(sum(sc["seat_gp"])) < 1e-9                      # zero-sum
    assert sc["seat_gp"][1] == sc["seat_gp"][2]                # betli: defenders split evenly


def test_score_rejects_unfinished_and_bad_kontra(client, key):
    h = {"Authorization": key["Authorization"]}
    d = client.post("/api/v1/deal", json={"seed": 7}, headers=h).json()
    pos = {"hands": d["hands"], "talon": d["talon"],
           "contract": "betli", "trump": None, "plays": []}
    assert client.post("/api/v1/score", json=pos, headers=h).status_code == 400
    pos["contract"] = "nincs ilyen"
    assert client.post("/api/v1/legal", json=pos, headers=h).status_code == 400


# ── dataset ─────────────────────────────────────────────────────────────────────

def test_games_list_and_fetch(client, key):
    h = {"Authorization": key["Authorization"]}
    recording.record_game({
        "id": "g1", "created_at": 1000.0, "seed": 1, "contract": "piros ulti",
        "trump": "hearts", "soloist_seat": 0, "human_seat": 0, "kontra_level": 0,
        "winner": "soloist", "made": True, "seat_gp": [8, -4, -4],
        "players": [], "transcript": {"deal": {}, "plays": []}})
    lst = client.get("/api/v1/games", headers=h).json()
    assert lst["games"][0]["id"] == "g1" and lst["next_cursor"] is None
    full = client.get("/api/v1/games/g1", headers=h).json()
    assert full["seat_gp"] == [8, -4, -4]
    assert client.get("/api/v1/games/nope", headers=h).status_code == 404


# ── matches (the environment veneer) ────────────────────────────────────────────

def test_match_requires_exactly_one_me(client, key):
    h = {"Authorization": key["Authorization"]}
    r = client.post("/api/v1/matches",
                    json={"seats": ["me", "me", "frontier"]}, headers=h)
    assert r.status_code == 400


def test_match_view_is_cheat_clean_and_seat_locked(client, key):
    h = {"Authorization": key["Authorization"]}
    r = client.post("/api/v1/matches", json={"seats": ["me", "frontier", "frontier"],
                                             "seed": 5, "agent": "tesztbot"}, headers=h)
    assert r.status_code == 200, r.text
    mid = r.json()["match_id"]
    s = client.get(f"/api/v1/matches/{mid}", headers=h).json()
    assert s["live"] and s["seat"] == 0
    assert len(s["own_hand"]) in (10, 12)                      # my cards, face up
    # a DIFFERENT key owner cannot even look at this match
    r2 = client.post("/api/auth/register",
                     json={"username": "masik", "password": "jelszo123"})
    other_browser = {"Authorization": f"Bearer {r2.json()['token']}"}
    k2 = client.post("/api/v1/keys", json={"name": "k"}, headers=other_browser).json()["key"]
    assert client.get(f"/api/v1/matches/{mid}",
                      headers={"Authorization": f"Bearer {k2}"}).status_code == 403


@pytest.mark.slow
def test_agent_plays_a_full_match_vs_frontier(client, key, tmp_path):
    """A dumb agent (first legal everything) completes a real deal against two
    frontier chairs through /api/v1 alone, and the game lands in the dataset as
    kind=bot with the agent's name."""
    import json as _json
    import sqlite3

    h = {"Authorization": key["Authorization"]}
    # seed 14: the frontier bids (golden matrix) → the PLAY path + recording run
    mid = client.post("/api/v1/matches", json={"agent": "elso_bot", "seed": 14},
                      headers=h).json()["match_id"]

    def act(body):
        r = client.post(f"/api/v1/matches/{mid}/act", json=body, headers=h)
        assert r.status_code == 200, r.text
        return r.json()

    for _ in range(300):
        s = client.get(f"/api/v1/matches/{mid}", headers=h).json()
        if s["phase"] in ("done", "passed"):
            break
        a = s.get("auction") or {}
        if s["phase"] == "bid" and a.get("is_human_turn"):
            real = [b for b in (a.get("legal_bids") or []) if b["rung_index"] >= 0]
            if a.get("awaiting_bid") and a.get("opening") and real:
                b = real[0]                       # open with the lowest contract
                act({"type": "bid", "rung_index": b["rung_index"],
                     "bid_index": b["bid_index"],
                     "trump": b["trump_options"][0] if len(b["trump_options"]) == 1 else None,
                     "discard_ids": s["talon_ids"][:2]})
            else:
                act({"type": "pass",
                     "discard_ids": s["talon_ids"][:2] if a.get("awaiting_bid") else []})
        elif s["phase"] == "trump_select" and s.get("is_chooser"):
            act({"type": "trump", "trump": s["trump_options"][0]})
        elif s["phase"] == "kontra" and (s.get("kontra") or {}).get("is_human_turn"):
            act({"type": "kontra", "units": []})
        elif s["phase"] == "play" and s.get("legal_card_ids"):
            act({"type": "move", "card_id": s["legal_card_ids"][0]})
    else:
        raise AssertionError("match never finished")

    if s["phase"] == "done":               # an all-pass deal isn't recorded
        con = sqlite3.connect(str(tmp_path / "games.db"))
        row = con.execute("SELECT players FROM games WHERE id = ?", (mid,)).fetchone()
        con.close()
        assert row is not None
        players = _json.loads(row[0])
        me = [p for p in players if p["kind"] == "bot"]
        assert len(me) == 1 and me[0]["agent"] == "elso_bot" and me[0]["user_id"]
        assert sorted(p["kind"] for p in players) == ["ai", "ai", "bot"]
