"""API keys — the public API's identity (docs/PUBLIC_API.md, D1).

A key is `ua_` + 40 hex, shown ONCE at mint and stored as a SHA-256 hash (unlike
browser tokens, keys are long-lived — a DB leak must not yield usable credentials).
Keys are minted from a logged-in account (browser token), named, individually
revocable, and carry per-key rate budgets by COST CLASS (D3): `free` for the rules
kernel and dataset reads, `metered` for anything that makes the AI think.

`resolve_key(token)` is plugged into users.user_from_request, so a key IS an
identity everywhere a browser token is — including the match engine's viewer
resolution. That's the whole trick: an agent authenticates like a player.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from threading import RLock
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import users

router = APIRouter()

_lock = RLock()
_cache: dict[str, dict] = {}          # raw token -> {user_id, username, key_id}

# Per-key sliding-window budgets by cost class (D3). IP limits still apply on top.
RATE_FREE = 120        # rpm — deal/legal/score/dataset reads
RATE_METERED = 30      # rpm — match steps, bot decisions (each may cost AI seconds)
_WINDOW = 60.0
_hits: dict[str, list] = {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    name       TEXT NOT NULL,
    key_hash   BLOB NOT NULL UNIQUE,
    prefix     TEXT NOT NULL,          -- display: ua_ab12… (never the full key)
    created_at REAL NOT NULL,
    last_used  REAL,
    calls      INTEGER NOT NULL DEFAULT 0
);
"""


_schema_on = None                  # the connection the DDL has run on


def _db():
    global _schema_on
    con = users._db()
    if con is not _schema_on:      # once per CONNECTION (tests swap the DB), not per request
        con.executescript(_SCHEMA)
        _schema_on = con
    return con


def _hash(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


def resolve_key(token: str) -> Optional[dict]:
    """{user_id, username, key_id} for a valid `ua_…` key; None otherwise."""
    with users._lock:
        hit = _cache.get(token)
        if hit is None:
            row = _db().execute(
                "SELECT k.id, k.user_id, u.username FROM api_keys k "
                "JOIN users u ON u.id = k.user_id WHERE k.key_hash = ?",
                (_hash(token),)).fetchone()
            if row is None:
                return None
            hit = _cache[token] = {"key_id": row[0], "user_id": row[1],
                                   "username": row[2], "_touched": 0.0}
        now = time.time()
        if now - hit["_touched"] > 60.0:
            hit["_touched"] = now
            _db().execute("UPDATE api_keys SET last_used = ?, calls = calls + 1 "
                          "WHERE id = ?", (now, hit["key_id"]))
    return {"user_id": hit["user_id"], "username": hit["username"],
            "key_id": hit["key_id"]}


def check_key_rate(key_id: str, cls: str) -> None:
    """Per-key budget for a cost class; 429 beyond it. Cheap dict work."""
    limit = RATE_METERED if cls == "metered" else RATE_FREE
    now = time.time()
    with _lock:
        h = _hits.setdefault(f"{key_id}:{cls}", [])
        h[:] = [t for t in h if now - t < _WINDOW]
        if len(h) >= limit:
            raise HTTPException(status_code=429,
                                detail=f"API-kulcs limit ({cls}: {limit}/perc) — lassíts.")
        h.append(now)


def require_key(request: Request) -> dict:
    """The calling key's identity, or 401. Browser tokens are NOT keys — the public
    API is key-only so usage is always attributable to a mintable, revocable thing."""
    token = users.bearer_token(request)
    if not token.startswith("ua_"):
        raise HTTPException(status_code=401,
                            detail="API-kulcs kell (Authorization: Bearer ua_…).")
    ident = resolve_key(token)
    if ident is None:
        raise HTTPException(status_code=401, detail="Érvénytelen API-kulcs.")
    return ident


# ── Key management (browser-token authed — the web app's profile surface) ───────

class MintRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)


@router.post("/keys")
def mint_key(req: MintRequest, request: Request = None) -> dict:
    if users.bearer_token(request).startswith("ua_"):
        raise HTTPException(status_code=403, detail="Kulcs nem hozhat létre kulcsot.")
    user = users.require_user(request)
    token = "ua_" + os.urandom(20).hex()
    with users._lock:
        _db().execute(
            "INSERT INTO api_keys (id, user_id, name, key_hash, prefix, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (uuid.uuid4().hex[:12], user["user_id"], req.name.strip(),
             _hash(token), token[:9] + "…", time.time()))
    # The ONLY time the full key is ever visible. Store it now.
    return {"key": token, "name": req.name.strip()}


@router.get("/keys")
def list_keys(request: Request = None) -> dict:
    user = users.require_user(request)
    with users._lock:
        rows = _db().execute(
            "SELECT id, name, prefix, created_at, last_used, calls FROM api_keys "
            "WHERE user_id = ? ORDER BY created_at", (user["user_id"],)).fetchall()
    return {"keys": [{"id": r[0], "name": r[1], "prefix": r[2], "created_at": r[3],
                      "last_used": r[4], "calls": r[5]} for r in rows]}


@router.delete("/keys/{key_id}")
def revoke_key(key_id: str, request: Request = None) -> dict:
    user = users.require_user(request)
    with users._lock:
        n = _db().execute("DELETE FROM api_keys WHERE id = ? AND user_id = ?",
                          (key_id, user["user_id"])).rowcount
        _cache.clear()                 # revocation must be immediate
    return {"revoked": bool(n)}
