"""Abuse guardrails for a publicly reachable UltiArena.

Without a password the site is one certificate-transparency scan away from bots, and
every AI move costs real solver time — four worker processes saturate quickly. These
limits make "anyone can open the link" safe by making abuse cheap to REFUSE: a rejected
request is a dict lookup, while the thing worth protecting (the solver, the session
store, the puzzle threads) is never touched.

Three layers, all per real client IP:
  * request rate      — sliding window over /api/* (health exempt)
  * concurrent work   — how many AI-solving requests one IP may have in flight
  * session creation  — a global cap and a per-IP cap on new games/puzzles

The client IP comes from CF-Connecting-IP (set by Cloudflare, and the tunnel means
nothing else can reach us), falling back to X-Forwarded-For then the socket. All knobs
live in ulti.config; set RATE_LIMIT_RPM=0 to disable the whole layer for local work.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import RLock
from typing import Deque, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from ulti.config import env_int

# Generous for real play (a fast player + animation polling is well under this),
# tight enough that a script can't spin the solver.
RATE_LIMIT_RPM = env_int("RATE_LIMIT_RPM", 240)        # 0 disables everything here
MAX_INFLIGHT_PER_IP = env_int("MAX_INFLIGHT_PER_IP", 4)
MAX_SESSIONS_TOTAL = env_int("MAX_SESSIONS_TOTAL", 200)
MAX_SESSIONS_PER_IP = env_int("MAX_SESSIONS_PER_IP", 12)

_WINDOW = 60.0
_lock = RLock()
_hits: Dict[str, Deque[float]] = defaultdict(deque)
_inflight: Dict[str, int] = defaultdict(int)

# Requests that make the AI think — these are the ones worth metering hard.
_HEAVY = ("/api/play/move", "/api/play/new", "/api/play/bid", "/api/play/pass",
          "/api/play/kontra", "/api/play/trump", "/api/play/analysis",
          "/api/puzzle/new", "/api/puzzle/solve", "/api/pis/explore")


def client_ip(request: Optional[Request]) -> str:
    """The real visitor's IP. Cloudflare sets CF-Connecting-IP and, because the tunnel
    is the only way in, it cannot be spoofed by a client.

    ``None`` means an in-process caller (the golden harness, tests, tournaments) —
    those are trusted and bypass the caps entirely."""
    if request is None:
        return "local"
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _prune(dq: Deque[float], now: float) -> None:
    while dq and now - dq[0] > _WINDOW:
        dq.popleft()


def check_rate(ip: str) -> bool:
    """True if this request is within the per-IP window."""
    if RATE_LIMIT_RPM <= 0:
        return True
    now = time.time()
    with _lock:
        dq = _hits[ip]
        _prune(dq, now)
        if len(dq) >= RATE_LIMIT_RPM:
            return False
        dq.append(now)
        return True


async def limit_middleware(request: Request, call_next):
    """Rate + concurrency gate. Health and static assets pass through untouched."""
    path = request.url.path
    if RATE_LIMIT_RPM <= 0 or not path.startswith("/api/") or path == "/api/health":
        return await call_next(request)

    ip = client_ip(request)
    if not check_rate(ip):
        return JSONResponse(status_code=429,
                            content={"detail": "Túl sok kérés — várj egy kicsit."},
                            headers={"Retry-After": "30"})

    heavy = any(path.startswith(h) for h in _HEAVY)
    if heavy:
        with _lock:
            if _inflight[ip] >= MAX_INFLIGHT_PER_IP:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Egyszerre túl sok játék — várd meg az előzőt."},
                    headers={"Retry-After": "5"})
            _inflight[ip] += 1
    try:
        return await call_next(request)
    finally:
        if heavy:
            with _lock:
                _inflight[ip] = max(0, _inflight[ip] - 1)


def guard_new_session(request: Optional[Request], sessions: dict, owner_of,
                      on_evict=None) -> str:
    """Called before creating a game/puzzle; returns the owning IP so the caller can
    tag the new session.

    Per-IP cap EVICTS rather than refuses: a real player clicking "Új játék" a few
    times, or reloading the tab, would otherwise be locked out for hours by their own
    abandoned games. Dropping their OLDEST session bounds memory just as well and is
    invisible to an honest user, while a script still cannot accumulate more than the
    cap. Only the GLOBAL cap refuses — that one means the server is genuinely full.

    ``owner_of(session) -> ip``; ``on_evict(session)`` releases resources (a puzzle
    session owns a background thread).
    """
    ip = client_ip(request)
    if RATE_LIMIT_RPM <= 0 or request is None:      # local/in-process caller → no caps
        return ip
    if len(sessions) >= MAX_SESSIONS_TOTAL:
        raise HTTPException(status_code=429,
                            detail="A szerver megtelt — próbáld újra pár perc múlva.")
    mine = [(gid, s) for gid, s in sessions.items() if owner_of(s) == ip]
    if len(mine) >= MAX_SESSIONS_PER_IP:
        # play sessions track last_touch; puzzle sessions track start — either way,
        # oldest activity goes first.
        mine.sort(key=lambda kv: getattr(kv[1], "last_touch", None)
                  or getattr(kv[1], "start", 0.0))
        for gid, sess in mine[:len(mine) - MAX_SESSIONS_PER_IP + 1]:
            if on_evict is not None:
                try:
                    on_evict(sess)
                except Exception:
                    pass
            sessions.pop(gid, None)
    return ip
