"""The profile — MY games from the database, stats, and analysis of any of them.

Identity is layered (milan 2026-08-11: no forced sign-in): a logged-in account
(bearer token) owns its games by user_id; an anonymous browser owns its games by
device id. `/me/*` accepts either — the profile page works before an account
exists, and a login simply widens the same list. Locally, DEV_AUTOLOGIN gives a
named account with zero typing (users.devlogin).

Analysis of a RECORDED game rebuilds the position from the stored transcript and
runs the exact worker op the live post-game board uses — same verdicts, same
board, any game in the database (yours only)."""
from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ulti.bidding.ladder import BIDS_BY_NAME
from ulti.card import card_from_id
from ulti.config import env_int

from . import ai_pool
from .ai_play import analysis_payload
from .engine import solve_plan
from .recording import games_db
from .users import user_from_request

router = APIRouter()

_DEV_RE = r"^[0-9a-fA-F-]{8,64}$"


def _identity(request: Optional[Request], device: Optional[str]) -> dict:
    """{user_id?, device?} — whatever this caller can prove. 401 with neither."""
    user = user_from_request(request)
    ident = {}
    if user:
        ident["user_id"] = user["user_id"]
        ident["username"] = user["username"]
    if device:
        ident["device"] = device
    if not ident:
        raise HTTPException(status_code=401, detail="Se belépés, se eszköz-azonosító.")
    return ident


def _is_mine(players: list, ident: dict) -> Optional[dict]:
    """The player entry that belongs to this identity, or None."""
    for p in players:
        if ident.get("user_id") and p.get("user_id") == ident["user_id"]:
            return p
        if ident.get("device") and p.get("device") == ident["device"]:
            return p
    return None


def _my_rows(ident: dict, since: Optional[float] = None,
             cursor: Optional[float] = None, limit: Optional[int] = None) -> List[dict]:
    q = ("SELECT id, created_at, seed, contract, trump, soloist_seat, kontra_level, "
         "winner, made, seat_gp, players FROM games WHERE EXISTS ("
         "SELECT 1 FROM json_each(games.players) je WHERE ")
    cond, args = [], []
    if ident.get("user_id"):
        cond.append("json_extract(je.value,'$.user_id') = ?"); args.append(ident["user_id"])
    if ident.get("device"):
        cond.append("json_extract(je.value,'$.device') = ?"); args.append(ident["device"])
    q += "(" + " OR ".join(cond) + "))"
    if since:
        q += " AND created_at >= ?"; args.append(since)
    if cursor:
        q += " AND created_at < ?"; args.append(cursor)
    q += " ORDER BY created_at DESC"
    if limit:
        q += " LIMIT ?"; args.append(limit)
    con = games_db()
    rows = con.execute(q, args).fetchall()
    con.close()
    out = []
    for r in rows:
        players = json.loads(r[10])
        mine = _is_mine(players, ident)
        if mine is None:
            continue
        seat_gp = json.loads(r[9])
        seat = mine["seat"]
        out.append({"id": r[0], "created_at": r[1], "seed": r[2], "contract": r[3],
                    "trump": r[4], "soloist_seat": r[5], "kontra_level": r[6],
                    "winner": r[7], "made": bool(r[8]),
                    "my_seat": seat, "i_was_soloist": seat == r[5],
                    "my_gp": float(seat_gp[seat])})
    return out


@router.get("/me/games")
def my_games(request: Request = None, device: Optional[str] = None,
             limit: int = 15, cursor: Optional[float] = None) -> dict:
    ident = _identity(request, device)
    limit = max(1, min(50, limit))
    rows = _my_rows(ident, cursor=cursor, limit=limit)
    return {"games": rows,
            "next_cursor": rows[-1]["created_at"] if len(rows) == limit else None}


@router.get("/me/stats")
def my_stats(request: Request = None, device: Optional[str] = None) -> dict:
    ident = _identity(request, device)
    rows = _my_rows(ident)
    n = len(rows)
    if n == 0:
        return {"games": 0}
    gp = sum(r["my_gp"] for r in rows)
    sol = [r for r in rows if r["i_was_soloist"] and r["contract"] != "passz"]
    per: dict = {}
    for r in rows:
        c = per.setdefault(r["contract"], {"contract": r["contract"], "n": 0,
                                           "gp": 0.0, "sol_n": 0, "sol_made": 0})
        c["n"] += 1
        c["gp"] += r["my_gp"]
        if r["i_was_soloist"] and r["contract"] != "passz":
            c["sol_n"] += 1
            c["sol_made"] += int(r["made"])
    contracts = sorted(per.values(), key=lambda c: -c["n"])
    for c in contracts:
        c["gp_mean"] = round(c["gp"] / c["n"], 2)
        c["gp"] = round(c["gp"], 1)
    return {
        "games": n,
        "gp_total": round(gp, 1),
        "gp_per_game": round(gp / n, 2),
        "wins": sum(1 for r in rows if r["my_gp"] > 0),
        "as_soloist": {"n": len(sol),
                       "made": sum(1 for r in sol if r["made"]),
                       "gp": round(sum(r["my_gp"] for r in sol), 1)},
        "contracts": contracts,
    }


class NickRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)


@router.post("/me/nickname")
def set_nickname(req: NickRequest, request: Request = None) -> dict:
    """Rename the logged-in account (uniqueness enforced)."""
    import sqlite3

    from . import users
    from .users import _USERNAME_RE, require_user

    user = require_user(request)
    name = req.username.strip()
    if not _USERNAME_RE.match(name):
        raise HTTPException(status_code=400,
                            detail="A név 3-20 karakter: betű, szám, kötőjel, aláhúzás.")
    with users._lock:
        try:
            users._db().execute("UPDATE users SET username = ? WHERE id = ?",
                                (name, user["user_id"]))
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Ez a név már foglalt.")
        users._token_cache.clear()          # cached identities carry the old name
    return {"username": name}


# ── Analysis of a recorded game — the live board, off the database ──────────────

@router.post("/me/games/{game_id}/analysis")
def game_analysis(game_id: str, request: Request = None,
                  device: Optional[str] = None) -> dict:
    ident = _identity(request, device)
    con = games_db()
    r = con.execute("SELECT contract, trump, soloist_seat, human_seat, seed, players, "
                    "transcript FROM games WHERE id = ?", (game_id,)).fetchone()
    con.close()
    if r is None:
        raise HTTPException(status_code=404, detail="Nincs ilyen játszma.")
    contract, trump, soloist_seat, seed = r[0], r[1], r[2], r[4]
    players = json.loads(r[5])
    if _is_mine(players, ident) is None:
        raise HTTPException(status_code=403, detail="Nem a te játszmád.")
    t = json.loads(r[6])
    bid = BIDS_BY_NAME.get(contract)
    if bid is None or not t.get("plays"):
        raise HTTPException(status_code=400, detail="Ez a játszma nem elemezhető.")

    # transcript deal is PLAY space (soloist first); the contract maps to the solver
    # objective through the same solve_plan the live game used
    hands0 = t["deal"]["hands"]
    talon = t["deal"]["talon"]
    sol_cards = [card_from_id(c) for c in hands0[0]]
    solve_c, build_c, tr, restrict, weights = solve_plan(bid, trump, sol_cards)

    job = {"hands0": hands0, "talon": talon, "build_c": build_c, "solve_c": solve_c,
           "trump": tr, "restrict": restrict, "has_ulti": bool(bid.ulti),
           "weights": weights, "voids": None,
           "history": [(p, c) for p, c, _ti in t["plays"]],
           "bid": bid, "seed": seed,
           "analysis_worlds": env_int("ANALYSIS_WORLDS", 8)}
    raw = ai_pool.run("analysis", job)

    # by_ai per play index: which seats were people (human/bot) in this recording
    human_pis = {(p["seat"] - soloist_seat) % 3 for p in players
                 if p.get("kind") in ("human", "bot")}
    return analysis_payload(
        raw, game_id=game_id, contract=contract,
        solve_c=solve_c, build_c=build_c, restrict=restrict, weights=weights,
        trump=tr, hands0=[[card_from_id(i) for i in h] for h in hands0],
        talon=[card_from_id(c) for c in talon],
        human_pi=min(human_pis) if human_pis else 0,
        by_ai=lambda row: row["player_id"] not in human_pis)
