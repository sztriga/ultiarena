"""The profile backend (/me/*): my games by user OR device, stats, nickname,
and analysis of a recorded game rebuilt from its transcript."""
from __future__ import annotations

import time

from apps.api import recording

DEV = "dddddddd-1111-2222-3333-444444444444"


def _rec(gid, ts, contract="piros parti", *, user_id=None, device=None, seat=0,
         sol_seat=0, made=True, gp=(4, -2, -2)):
    recording.record_game({
        "id": gid, "created_at": ts, "seed": 1, "contract": contract,
        "trump": "hearts", "soloist_seat": sol_seat, "human_seat": seat,
        "kontra_level": 0, "winner": "soloist" if made else "defenders",
        "made": made, "seat_gp": list(gp),
        "players": [{"seat": s,
                     "kind": "human" if s == seat else "ai",
                     "user_id": user_id if s == seat else None,
                     "device": device if s == seat else None,
                     "agent": None if s == seat else "frontier"} for s in range(3)],
        "transcript": {"deal": {"hands": [[], [], []], "talon": []},
                       "auction": [], "plays": [], "kontra": {}, "marriages": []}})


def test_games_by_device_no_login_needed(client):
    _rec("g1", 1000.0, device=DEV)
    _rec("g2", 2000.0, device=DEV, made=False, gp=(-4, 2, 2))
    _rec("gx", 1500.0, device="e" * 8)                      # someone else's
    r = client.get(f"/api/me/games?device={DEV}")
    assert r.status_code == 200
    games = r.json()["games"]
    assert [g["id"] for g in games] == ["g2", "g1"]         # newest first
    assert games[0]["my_gp"] == -4.0 and games[1]["my_gp"] == 4.0


def test_games_pagination_cursor(client):
    for i in range(7):
        _rec(f"g{i}", 1000.0 + i, device=DEV)
    p1 = client.get(f"/api/me/games?device={DEV}&limit=5").json()
    assert len(p1["games"]) == 5 and p1["next_cursor"]
    p2 = client.get(f"/api/me/games?device={DEV}&limit=5"
                    f"&cursor={p1['next_cursor']}").json()
    assert len(p2["games"]) == 2 and p2["next_cursor"] is None
    assert {g["id"] for g in p1["games"]} | {g["id"] for g in p2["games"]} == \
        {f"g{i}" for i in range(7)}


def test_login_widens_the_same_list(client):
    r = client.post("/api/auth/register",
                    json={"username": "milan", "password": "jelszo123"})
    tok, uid_hdr = r.json()["token"], {"Authorization": f"Bearer {r.json()['token']}"}
    me = client.get("/api/auth/me", headers=uid_hdr).json()["user"]
    _rec("g_dev", 1000.0, device=DEV)                       # anonymous era
    _rec("g_acc", 2000.0, user_id=me["user_id"])            # logged-in era
    both = client.get(f"/api/me/games?device={DEV}", headers=uid_hdr).json()["games"]
    assert [g["id"] for g in both] == ["g_acc", "g_dev"]


def test_stats_aggregate(client):
    _rec("s1", 1000.0, device=DEV, contract="ulti", made=True, gp=(8, -4, -4))
    _rec("s2", 2000.0, device=DEV, contract="ulti", made=False, gp=(-8, 4, 4))
    _rec("s3", 3000.0, device=DEV, contract="piros parti", seat=1, sol_seat=0,
         gp=(4, -2, -2))
    st = client.get(f"/api/me/stats?device={DEV}").json()
    assert st["games"] == 3 and st["gp_total"] == -2.0
    assert st["wins"] == 1
    assert st["as_soloist"]["n"] == 2 and st["as_soloist"]["made"] == 1
    ulti = next(c for c in st["contracts"] if c["contract"] == "ulti")
    assert ulti["n"] == 2 and ulti["sol_made"] == 1


def test_identity_required(client):
    assert client.get("/api/me/games").status_code == 401


def test_nickname_rename(client):
    r = client.post("/api/auth/register",
                    json={"username": "milan", "password": "jelszo123"})
    hdr = {"Authorization": f"Bearer {r.json()['token']}"}
    assert client.post("/api/me/nickname", json={"username": "ultikirály"},
                       headers=hdr).status_code == 200
    assert client.get("/api/auth/me", headers=hdr).json()["user"]["username"] == "ultikirály"
    client.post("/api/auth/register", json={"username": "masik", "password": "jelszo123"})
    assert client.post("/api/me/nickname", json={"username": "masik"},
                       headers=hdr).status_code == 409


def test_devlogin_gated_by_env(client, monkeypatch):
    assert client.post("/api/auth/devlogin").status_code == 404
    monkeypatch.setenv("DEV_AUTOLOGIN", "helyi_milan")
    r = client.post("/api/auth/devlogin")
    assert r.status_code == 200 and r.json()["username"] == "helyi_milan"
    r2 = client.post("/api/auth/devlogin")                  # idempotent — same account
    me1 = client.get("/api/auth/me",
                     headers={"Authorization": f"Bearer {r.json()['token']}"}).json()
    me2 = client.get("/api/auth/me",
                     headers={"Authorization": f"Bearer {r2.json()['token']}"}).json()
    assert me1["user"]["user_id"] == me2["user"]["user_id"]


def test_analysis_needs_ownership_and_plays(client):
    _rec("empty", 1000.0, device=DEV)                       # no plays recorded
    assert client.post(f"/api/me/games/empty/analysis?device={DEV}").status_code == 400
    assert client.post("/api/me/games/empty/analysis?device=" + "f" * 8).status_code == 403
    assert client.post(f"/api/me/games/nope/analysis?device={DEV}").status_code == 404
