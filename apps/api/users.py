"""Accounts — username + password, the minimum that is still PROPER.

Design (docs/MULTIPLAYER.md, decision 3): sqlite `data/users.db` (WAL), scrypt
password hashes (stdlib — no new deps), opaque bearer tokens persisted so a server
restart doesn't log everyone out. The client sends `Authorization: Bearer <token>`;
`user_from_request` resolves it ANYWHERE identity is useful and returns None for
anonymous — nothing outside this module ever sees a password.

Identity vs abuse-control stay separate axes (the limits keep keying on the client
IP); auth endpoints get their own tight per-IP budget below because they are the
brute-force surface.
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from threading import RLock
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ulti.config import env_str

from .limits import client_ip

router = APIRouter()

_DB_PATH = env_str("USERS_DB") or str(Path(__file__).resolve().parents[2] / "data" / "users.db")
_lock = RLock()
_conn: sqlite3.Connection | None = None
_token_cache: dict[str, dict] = {}          # token -> {user_id, username} (hot path)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    username   TEXT NOT NULL UNIQUE COLLATE NOCASE,
    pw_hash    BLOB NOT NULL,
    pw_salt    BLOB NOT NULL,
    created_at REAL NOT NULL,
    device_id  TEXT
);
CREATE TABLE IF NOT EXISTS tokens (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    last_seen  REAL NOT NULL
);
"""

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_\-áéíóöőúüűÁÉÍÓÖŐÚÜŰ]{3,20}$")

# Brute-force budget: failed credential attempts per IP per 10 minutes. Separate
# from the general rate limit — 240 rpm is far too generous for password guessing.
_AUTH_WINDOW = 600.0
_AUTH_MAX_FAILS = 20
_auth_fails: dict[str, list] = {}


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        # autocommit (isolation_level=None): every statement lands immediately.
        # Without it, python-sqlite3 silently opens a transaction on the first
        # UPDATE and HOLDS the write lock forever — the last_seen touch on every
        # authed request locked the whole DB against outside writers (found by
        # smoke-testing 2026-08-10). All writes here are single statements under
        # _lock, so autocommit is exactly right.
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False, isolation_level=None)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.executescript(_SCHEMA)
    return _conn


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def _check_auth_budget(ip: str) -> None:
    now = time.time()
    with _lock:
        fails = _auth_fails.setdefault(ip, [])
        fails[:] = [t for t in fails if now - t < _AUTH_WINDOW]
        if len(fails) >= _AUTH_MAX_FAILS:
            raise HTTPException(status_code=429,
                                detail="Túl sok próbálkozás — várj pár percet.")


def _record_fail(ip: str) -> None:
    with _lock:
        _auth_fails.setdefault(ip, []).append(time.time())


def _issue_token(user_id: str, username: str) -> str:
    token = uuid.uuid4().hex + uuid.uuid4().hex
    now = time.time()
    with _lock:
        _db().execute("INSERT INTO tokens (token, user_id, created_at, last_seen) "
                      "VALUES (?,?,?,?)", (token, user_id, now, now))
        _token_cache[token] = {"user_id": user_id, "username": username, "_touched": now}
    return token


def user_from_request(request: Optional[Request]) -> Optional[dict]:
    """{user_id, username} for a valid bearer token; None for anonymous/in-process.
    The cheap, everywhere-safe identity hook (decision: identity fills in)."""
    if request is None:
        return None
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    if token.startswith("ua_"):        # an API key IS an identity (docs/PUBLIC_API.md)
        from .apikeys import resolve_key
        ident = resolve_key(token)
        return {"user_id": ident["user_id"], "username": ident["username"]} if ident else None
    with _lock:
        hit = _token_cache.get(token)
        if hit is None:
            row = _db().execute(
                "SELECT t.user_id, u.username FROM tokens t JOIN users u ON u.id = t.user_id "
                "WHERE t.token = ?", (token,)).fetchone()
            if row is None:
                return None
            hit = _token_cache[token] = {"user_id": row[0], "username": row[1],
                                         "_touched": 0.0}
        now = time.time()
        if now - hit["_touched"] > 60.0:      # a poll every 2s must not write every 2s
            hit["_touched"] = now
            _db().execute("UPDATE tokens SET last_seen = ? WHERE token = ?", (now, token))
    return {"user_id": hit["user_id"], "username": hit["username"]}


def require_user(request: Request = None) -> dict:
    user = user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Bejelentkezés szükséges.")
    return user


# ── Endpoints ───────────────────────────────────────────────────────────────────

class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=6, max_length=128)
    device_id: Optional[str] = Field(None, pattern=r"^[0-9a-fA-F-]{8,64}$")


@router.post("/auth/register")
def register(req: Credentials, request: Request = None) -> dict:
    ip = client_ip(request)
    _check_auth_budget(ip)
    name = req.username.strip()
    if not _USERNAME_RE.match(name):
        raise HTTPException(status_code=400,
                            detail="A név 3-20 karakter: betű, szám, kötőjel, aláhúzás.")
    salt = os.urandom(16)
    uid = uuid.uuid4().hex[:12]
    with _lock:
        try:
            _db().execute(
                "INSERT INTO users (id, username, pw_hash, pw_salt, created_at, device_id) "
                "VALUES (?,?,?,?,?,?)",
                (uid, name, _scrypt(req.password, salt), salt, time.time(), req.device_id))
        except sqlite3.IntegrityError:
            _record_fail(ip)               # name-probing costs budget too
            raise HTTPException(status_code=409, detail="Ez a név már foglalt.")
    return {"token": _issue_token(uid, name), "username": name}


@router.post("/auth/login")
def login(req: Credentials, request: Request = None) -> dict:
    ip = client_ip(request)
    _check_auth_budget(ip)
    with _lock:
        row = _db().execute(
            "SELECT id, username, pw_hash, pw_salt FROM users WHERE username = ?",
            (req.username.strip(),)).fetchone()
    if row is None or _scrypt(req.password, row[3]) != row[2]:
        _record_fail(ip)
        raise HTTPException(status_code=401, detail="Hibás név vagy jelszó.")
    uid, name = row[0], row[1]
    if req.device_id:                      # attach this browser to the account
        with _lock:
            _db().execute("UPDATE users SET device_id = ? WHERE id = ?",
                          (req.device_id, uid))
    return {"token": _issue_token(uid, name), "username": name}


@router.post("/auth/logout")
def logout(request: Request = None) -> dict:
    auth = (request.headers.get("authorization", "") if request else "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if token:
        with _lock:
            _db().execute("DELETE FROM tokens WHERE token = ?", (token,))
            _token_cache.pop(token, None)
    return {"ok": True}


@router.get("/auth/me")
def me(request: Request = None) -> dict:
    return {"user": user_from_request(request)}
