"""Játék élőben — the lobby: presence, chat, and the Asztal (3-seat tables).

Design (docs/MULTIPLAYER.md): everything here is in-memory behind one lock, served
by short-polling — a poll is a heartbeat (presence) plus a state read with a chat
cursor, so three players polling every 2s costs dict lookups. The transport can
become WebSockets later without touching this data model.

Rules (milan 2026-08-10): anyone in the lobby may sit at an open seat — there is
no approval step, the HOST KICKS unwanted joiners instead ("subtle" control). The
host can invite; the invitee sees the table highlighted. Host leaves → the next
occupant inherits the table; empty tables dissolve; vanished players (no poll for
PRESENCE_TTL) are stood up from their seats. One table per player.

Starting the game is stage 2 (the 3-human engine); /live/table/start exists but
answers 501 so the UI can show an honest "hamarosan".
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from threading import RLock
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .users import require_user

router = APIRouter()

PRESENCE_TTL = 15.0          # no poll for this long → you left the lobby
CHAT_KEEP = 200              # ring buffer length

_lock = RLock()
_members: Dict[str, dict] = {}       # user_id -> {username, last_seen}
_chat: deque = deque(maxlen=CHAT_KEEP)
_chat_seq = 0
_tables: Dict[str, dict] = {}        # table_id -> table dict (see design doc)


def _now() -> float:
    return time.time()


def _reap() -> None:
    """Presence expiry + table hygiene. Called under _lock on every poll."""
    cutoff = _now() - PRESENCE_TTL
    gone = [uid for uid, m in _members.items() if m["last_seen"] < cutoff]
    for uid in gone:
        del _members[uid]
    for t in list(_tables.values()):
        # stand vanished players up from their seats
        t["seats"] = [s if (s is None or s in _members) else None for s in t["seats"]]
        if t["host_id"] not in _members or t["host_id"] not in t["seats"]:
            occupants = [s for s in t["seats"] if s is not None]
            if occupants:
                t["host_id"] = occupants[0]          # the next occupant inherits
            else:
                del _tables[t["id"]]                 # empty table dissolves


def _table_of(user_id: str) -> Optional[dict]:
    for t in _tables.values():
        if user_id in t["seats"]:
            return t
    return None


def _table_view(t: dict, viewer_id: str) -> dict:
    return {
        "table_id": t["id"],
        "host": _members.get(t["host_id"], {}).get("username", "?"),
        "is_host": t["host_id"] == viewer_id,
        "seats": [
            None if s is None else {
                "username": _members.get(s, {}).get("username", "?"),
                "user_id": s,
                "is_host": s == t["host_id"],
                "is_me": s == viewer_id,
            } for s in t["seats"]
        ],
        "full": all(s is not None for s in t["seats"]),
        "invited_me": viewer_id in t["invited"],
        "state": t["state"],
    }


# ── Endpoints (all token-authed) ────────────────────────────────────────────────

class PollRequest(BaseModel):
    chat_after: int = Field(0, ge=0)


@router.post("/live/poll")
def live_poll(req: PollRequest, request: Request = None) -> dict:
    user = require_user(request)
    with _lock:
        _members[user["user_id"]] = {"username": user["username"], "last_seen": _now()}
        _reap()
        mine = _table_of(user["user_id"])
        return {
            "me": user["username"],
            "members": sorted(m["username"] for m in _members.values()),
            "chat": [c for c in _chat if c["seq"] > req.chat_after],
            "tables": [_table_view(t, user["user_id"])
                       for t in sorted(_tables.values(), key=lambda t: t["created_at"])],
            "my_table": mine["id"] if mine else None,
        }


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=300)


@router.post("/live/chat")
def live_chat(req: ChatRequest, request: Request = None) -> dict:
    user = require_user(request)
    global _chat_seq
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Üres üzenet.")
    with _lock:
        _chat_seq += 1
        _chat.append({"seq": _chat_seq, "user": user["username"],
                      "text": text, "ts": _now()})
    return {"ok": True}


class TableRequest(BaseModel):
    table_id: str


@router.post("/live/table/create")
def table_create(request: Request = None) -> dict:
    user = require_user(request)
    with _lock:
        if _table_of(user["user_id"]) is not None:
            raise HTTPException(status_code=400, detail="Már ülsz egy asztalnál.")
        t = {"id": uuid.uuid4().hex[:8], "host_id": user["user_id"],
             "seats": [user["user_id"], None, None], "invited": set(),
             "state": "open", "game_id": None, "created_at": _now()}
        _tables[t["id"]] = t
        return {"table_id": t["id"]}


@router.post("/live/table/join")
def table_join(req: TableRequest, request: Request = None) -> dict:
    user = require_user(request)
    with _lock:
        t = _tables.get(req.table_id)
        if t is None or t["state"] != "open":
            raise HTTPException(status_code=404, detail="Nincs ilyen asztal.")
        if _table_of(user["user_id"]) is not None:
            raise HTTPException(status_code=400, detail="Már ülsz egy asztalnál.")
        for i, s in enumerate(t["seats"]):
            if s is None:
                t["seats"][i] = user["user_id"]
                t["invited"].discard(user["user_id"])
                return {"table_id": t["id"], "seat": i}
        raise HTTPException(status_code=409, detail="Az asztal megtelt.")


@router.post("/live/table/leave")
def table_leave(req: TableRequest, request: Request = None) -> dict:
    user = require_user(request)
    with _lock:
        t = _tables.get(req.table_id)
        if t is None:
            return {"ok": True}
        t["seats"] = [None if s == user["user_id"] else s for s in t["seats"]]
        _reap()                                       # host inheritance / dissolve
    return {"ok": True}


class KickRequest(BaseModel):
    table_id: str
    user_id: str


@router.post("/live/table/kick")
def table_kick(req: KickRequest, request: Request = None) -> dict:
    user = require_user(request)
    with _lock:
        t = _tables.get(req.table_id)
        if t is None:
            raise HTTPException(status_code=404, detail="Nincs ilyen asztal.")
        if t["host_id"] != user["user_id"]:
            raise HTTPException(status_code=403, detail="Csak a házigazda tehet ki játékost.")
        if req.user_id == user["user_id"]:
            raise HTTPException(status_code=400, detail="Magadat nem teheted ki — hagyd el az asztalt.")
        t["seats"] = [None if s == req.user_id else s for s in t["seats"]]
        t["invited"].discard(req.user_id)
    return {"ok": True}


class InviteRequest(BaseModel):
    table_id: str
    username: str


@router.post("/live/table/invite")
def table_invite(req: InviteRequest, request: Request = None) -> dict:
    user = require_user(request)
    with _lock:
        t = _tables.get(req.table_id)
        if t is None:
            raise HTTPException(status_code=404, detail="Nincs ilyen asztal.")
        if t["host_id"] != user["user_id"]:
            raise HTTPException(status_code=403, detail="Csak a házigazda hívhat meg.")
        target = next((uid for uid, m in _members.items()
                       if m["username"].lower() == req.username.lower()), None)
        if target is None:
            raise HTTPException(status_code=404, detail="Nincs ilyen játékos a lobbyban.")
        t["invited"].add(target)
    return {"ok": True}


@router.post("/live/table/start")
def table_start(req: TableRequest, request: Request = None) -> dict:
    """Stage 2 — the 3-human game engine (docs/MULTIPLAYER.md). Honest 501 until
    it lands so the UI can label the button 'hamarosan'."""
    require_user(request)
    raise HTTPException(status_code=501, detail="A játszma indítás hamarosan érkezik.")
