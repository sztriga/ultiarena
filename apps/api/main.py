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
from .play import router as play_router
from .puzzle import router as puzzle_router

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
    return {"status": "ok"}


# Abuse guardrails (see apps/api/limits.py) — the site is publicly reachable with no
# password, so per-IP rate/concurrency/session caps do the protecting.
app.middleware("http")(limit_middleware)

app.include_router(pis_router,    prefix="/api")   # /pis/explore — post-game analysis branches
app.include_router(play_router,   prefix="/api")   # /play/* — the full Ulti game
app.include_router(puzzle_router, prefix="/api")   # /puzzle/* — Villámtalon rush

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
