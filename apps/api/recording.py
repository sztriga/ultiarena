"""Persist completed games for later analysis of how the AI plays.

SQLite for now — a real, queryable store, single-file, zero-config. When the site is
deployed for multiplayer, only THIS module changes (→ Postgres); callers stay the same.

**User-aware from day one:** every seat carries `kind` (human/ai), a nullable `user_id`,
and (for AI) the `agent` config. Today the human seat is an anonymous session id and the
others are the frontier AI. When auth + human-vs-human land, real user_ids just fill in —
no schema change, and a seat being `human` with a user_id already models a real opponent.

Each row is one finished game with the full transcript (deal + auction + every action +
kontra + marriages + scores), so a game can be replayed and re-solved offline.

Recording is BEST-EFFORT: it never raises — a logging failure must never break a game.
Env GAMES_DB overrides the path (default: <repo>/data/games.db).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from threading import RLock

from ulti.config import env_str

_DB_PATH = env_str("GAMES_DB") or str(Path(__file__).resolve().parents[2] / "data" / "games.db")
_lock = RLock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id            TEXT PRIMARY KEY,   -- session/game id
    created_at    REAL,              -- unix ts
    seed          INTEGER,
    contract      TEXT,              -- resolved game name (e.g. 'piros ulti')
    trump         TEXT,              -- suit or NULL (colorless)
    soloist_seat  INTEGER,           -- real seat 0/1/2 that won the auction
    human_seat    INTEGER,           -- real seat the (current) human played
    kontra_level  INTEGER,
    winner        TEXT,              -- 'soloist' | 'defenders'
    made          INTEGER,           -- 1/0, primary contract made
    seat_gp       TEXT,              -- json [gp_seat0, gp_seat1, gp_seat2] (zero-sum GP)
    players       TEXT,              -- json [{seat,kind,user_id,agent}, ...]  <- user-aware
    transcript    TEXT               -- json {deal, auction, plays, kontra, marriages}
);
CREATE INDEX IF NOT EXISTS ix_games_created ON games(created_at);
CREATE INDEX IF NOT EXISTS ix_games_contract ON games(contract);
-- Future: CREATE TABLE users(id TEXT PRIMARY KEY, ...); user_ids in players[].user_id reference it.
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        # WAL: analysis tooling reads this file WHILE the server writes — under the
        # default rollback journal a concurrent reader gets "database is locked".
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


def record_game(rec: dict) -> None:
    """Insert (or replace) one finished game. Never raises."""
    try:
        with _lock:
            db = _db()
            db.execute(
                "INSERT OR REPLACE INTO games "
                "(id, created_at, seed, contract, trump, soloist_seat, human_seat, "
                " kontra_level, winner, made, seat_gp, players, transcript) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rec["id"], rec["created_at"], rec["seed"], rec["contract"], rec["trump"],
                    rec["soloist_seat"], rec["human_seat"], rec["kontra_level"],
                    rec["winner"], int(bool(rec["made"])),
                    json.dumps(rec["seat_gp"]), json.dumps(rec["players"]),
                    json.dumps(rec["transcript"], separators=(",", ":")),
                ),
            )
            db.commit()
    except Exception as e:
        # Never break a game over logging — but a data-collection deployment must not
        # fail SILENTLY either: one stderr line makes a dead recorder findable.
        print(f"[recording] game {rec.get('id', '?')} NOT recorded: {e!r}",
              file=sys.stderr, flush=True)
