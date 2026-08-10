"""Live-table games end to end — 3 humans, and 1 human + 2 AI chairs.

The whole point of the design (docs/MULTIPLAYER.md): a live game is the SAME
engine as the solo game with `humans` = the occupied seats, and every snapshot is
rendered per viewer. So these tests drive real HTTP against apps.api.main.app with
three bearer tokens: create/join/start a table, then each player polls their own
state and acts only when it says it's their turn — exactly what three browsers do.

The 3-human game involves no AI at all and runs in ~a second. The 1-human table
exercises milan's "port in the AI" requirement: empty chairs are AI seats running
the untouched frontier code path (slow — real PIMC — kept to one deal)."""
from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from apps.api import limits, live, recording, users
from apps.api.engine import _sessions, _sessions_lock
from apps.api.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(users, "_DB_PATH", str(tmp_path / "users.db"))
    monkeypatch.setattr(users, "_conn", None)
    monkeypatch.setattr(recording, "_DB_PATH", str(tmp_path / "games.db"))
    monkeypatch.setattr(recording, "_conn", None)
    monkeypatch.setattr(limits, "RATE_LIMIT_RPM", 0)   # 3 pollers share one test IP
    users._token_cache.clear()
    users._auth_fails.clear()
    live._members.clear()
    live._tables.clear()
    live._chat.clear()
    with TestClient(app) as c:
        yield c
    if users._conn is not None:
        users._conn.close()
    if recording._conn is not None:
        recording._conn.close()


def _register(client, name) -> dict:
    r = client.post("/api/auth/register", json={"username": name, "password": "jelszo123"})
    assert r.status_code == 200, r.text
    hdr = {"Authorization": f"Bearer {r.json()['token']}"}
    client.post("/api/live/poll", json={"chat_after": 0}, headers=hdr)
    return hdr


def _state(client, gid, hdr) -> dict:
    r = client.post("/api/play/state", json={"game_id": gid}, headers=hdr)
    assert r.status_code == 200, r.text
    return r.json()


def _act(client, gid, hdr, s) -> bool:
    """One player's browser: act iff my snapshot says it's my decision. Strategy:
    the forehand bids the lowest rung, everyone else passes; no kontra; first
    legal card. Returns True if an action was taken."""
    phase = s["phase"]
    if phase == "bid" and s["auction"]["is_human_turn"]:
        a = s["auction"]
        if a["awaiting_bid"]:
            real = [b for b in (a["legal_bids"] or []) if b["rung_index"] >= 0]
            if a["opening"] and real:
                b = real[0]
                trump = b["trump_options"][0] if len(b["trump_options"]) == 1 else None
                r = client.post("/api/play/bid", json={
                    "game_id": gid, "rung_index": b["rung_index"], "bid_index": b["bid_index"],
                    "trump": trump, "discard_ids": s["talon_ids"][:2]}, headers=hdr)
            else:   # holder's re-look → accept (pass starts play)
                r = client.post("/api/play/pass", json={"game_id": gid,
                                "discard_ids": s["talon_ids"][:2] if a["opening"] else []},
                                headers=hdr)
        else:
            r = client.post("/api/play/pass", json={"game_id": gid, "discard_ids": []},
                            headers=hdr)
        assert r.status_code == 200, r.text
        return True
    if phase == "trump_select" and s.get("is_chooser"):
        r = client.post("/api/play/trump",
                        json={"game_id": gid, "trump": s["trump_options"][0]}, headers=hdr)
        assert r.status_code == 200, r.text
        return True
    if phase == "kontra" and (s.get("kontra") or {}).get("is_human_turn"):
        r = client.post("/api/play/kontra", json={"game_id": gid, "units": []}, headers=hdr)
        assert r.status_code == 200, r.text
        return True
    if phase == "play" and s.get("legal_card_ids"):
        r = client.post("/api/play/move",
                        json={"game_id": gid, "card_id": s["legal_card_ids"][0]}, headers=hdr)
        assert r.status_code == 200, r.text
        return True
    return False


def _drive(client, gid, hdrs, max_steps=400) -> list:
    """Poll-and-act for every player until the deal ends; returns final states."""
    for _ in range(max_steps):
        states = [_state(client, gid, h) for h in hdrs]
        if states[0]["phase"] in ("done", "passed"):
            return states
        if not any(_act(client, gid, h, s) for h, s in zip(hdrs, states)):
            raise AssertionError(
                f"deadlock: nobody could act (phase={states[0]['phase']})")
    raise AssertionError("game did not finish")


def _table(client, hdrs) -> str:
    tid = client.post("/api/live/table/create", headers=hdrs[0]).json()["table_id"]
    for h in hdrs[1:]:
        assert client.post("/api/live/table/join",
                           json={"table_id": tid}, headers=h).status_code == 200
    return tid


def test_three_humans_play_a_full_deal(client, tmp_path):
    hdrs = [_register(client, n) for n in ("anna", "bela", "cili")]
    tid = _table(client, hdrs)
    r = client.post("/api/live/table/start", json={"table_id": tid}, headers=hdrs[0])
    assert r.status_code == 200, r.text
    gid = r.json()["game_id"]

    # mid-game perspective: each viewer sees ONLY their own auction hand
    for h in hdrs:
        s = _state(client, gid, h)
        assert len(s["own_hand"]) in (10, 12)

    finals = _drive(client, gid, hdrs)

    # every viewer agrees on the outcome, sees their OWN gp, and it's zero-sum
    assert len({f["phase"] for f in finals}) == 1
    seat_gp = finals[0]["result"]["seat_gp"]
    assert abs(sum(seat_gp)) < 1e-9
    for f in finals:
        assert f["result"]["human_gp"] == seat_gp[f["seat"]]
    seats = {f["seat"] for f in finals}
    assert seats == {0, 1, 2}

    # a PLAYED deal was recorded with all three user_ids (an all-pass isn't played)
    if finals[0]["phase"] == "done":
        con = sqlite3.connect(str(tmp_path / "games.db"))
        row = con.execute("SELECT players FROM games WHERE id = ?", (gid,)).fetchone()
        con.close()
        assert row is not None, "live game was not recorded"
        players = json.loads(row[0])
        assert all(p["kind"] == "human" and p["user_id"] for p in players)


def test_outsiders_cannot_touch_a_live_game(client):
    hdrs = [_register(client, n) for n in ("anna", "bela", "cili")]
    tid = _table(client, hdrs[:2])          # only anna + bela seated
    gid = client.post("/api/live/table/start", json={"table_id": tid},
                      headers=hdrs[0]).json()["game_id"]
    # cili is registered but NOT at the table; anonymous has no token at all
    assert client.post("/api/play/state", json={"game_id": gid},
                       headers=hdrs[2]).status_code == 403
    assert client.post("/api/play/state", json={"game_id": gid}).status_code == 403


def test_restart_only_after_the_deal_ends(client):
    hdrs = [_register(client, n) for n in ("anna", "bela", "cili")]
    tid = _table(client, hdrs)
    client.post("/api/live/table/start", json={"table_id": tid}, headers=hdrs[0])
    assert client.post("/api/live/table/start", json={"table_id": tid},
                       headers=hdrs[0]).status_code == 409


@pytest.mark.slow
def test_one_human_two_ai_chairs(client, tmp_path):
    """milan's requirement: empty chairs are the frontier AI via the SAME path.
    One real deal with actual net bidding + PIMC play — slow, kept to one."""
    hdrs = [_register(client, "solo_hero")]
    tid = client.post("/api/live/table/create", headers=hdrs[0]).json()["table_id"]
    r = client.post("/api/live/table/start", json={"table_id": tid}, headers=hdrs[0])
    assert r.status_code == 200, r.text
    gid = r.json()["game_id"]
    finals = _drive(client, gid, hdrs)
    assert finals[0]["phase"] in ("done", "passed")
    if finals[0]["phase"] == "done":
        con = sqlite3.connect(str(tmp_path / "games.db"))
        row = con.execute("SELECT players FROM games WHERE id = ?", (gid,)).fetchone()
        con.close()
        players = json.loads(row[0])
        kinds = sorted(p["kind"] for p in players)
        assert kinds == ["ai", "ai", "human"]
    with _sessions_lock:
        _sessions.pop(gid, None)
