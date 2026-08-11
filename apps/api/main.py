"""
FastAPI entry point for the Ulti web app (play + Villámtalon puzzle + post-game analysis).

Run locally:
    uvicorn apps.api.main:app --reload --port 8000

The web client (apps/web) proxies /api and /cards to this server.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .pis import router as pis_router
from .limits import limit_middleware
from .live import router as live_router
from .play import router as play_router
from .puzzle import router as puzzle_router
from .users import router as users_router

app = FastAPI(title="Ulti", version="0.3.0")

# Vite dev server runs on :5173; permissive CORS for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """Liveness + a glance at load. Exempt from the rate limits (see limits.py), so
    uptime monitors can poll it freely; exposes only counts, never session data."""
    from .engine import _sessions as _play_sessions
    from .puzzle import _sessions as _puzzle_sessions
    return {
        "status": "ok",
        "play_sessions": len(_play_sessions),
        "puzzle_sessions": len(_puzzle_sessions),
    }


# Abuse guardrails (see apps/api/limits.py) — the site is publicly reachable with no
# password, so per-IP rate/concurrency/session caps do the protecting.
app.middleware("http")(limit_middleware)

app.include_router(pis_router,    prefix="/api")   # /pis/explore — post-game analysis branches
app.include_router(play_router,   prefix="/api")   # /play/* — the full Ulti game
app.include_router(puzzle_router, prefix="/api")   # /puzzle/* — Villámtalon rush
app.include_router(users_router,  prefix="/api")   # /auth/* — accounts (docs/MULTIPLAYER.md)
app.include_router(live_router,   prefix="/api")   # /live/* — lobby, chat, Asztal

# The PUBLIC, versioned research API (docs/PUBLIC_API.md) — its own sub-application
# with its own OpenAPI docs at /api/v1/docs. Everything above stays internal.
from .public import v1 as public_v1  # noqa: E402
app.mount("/api/v1", public_v1)

# Hungarian Tell-deck card images at /cards (card_piece_<S><R>.jpg + card_back.png).
_CARDS_DIR = Path(__file__).resolve().parents[2] / "assets" / "cards"
app.mount("/cards", StaticFiles(directory=str(_CARDS_DIR)), name="cards")

# ── production static serving ────────────────────────────────────────────────────
# When the web app has been built (apps/web/dist exists), serve it directly — one
# process hosts both the API and the SPA, which is what sits behind the domain.
# Dev is unaffected: vite (5173) proxies /api and dist is simply not mounted if absent.
_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")
