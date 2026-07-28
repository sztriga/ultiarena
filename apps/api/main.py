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


app.include_router(pis_router,    prefix="/api")   # /pis/explore — post-game analysis branches
app.include_router(play_router,   prefix="/api")   # /play/* — the full Ulti game
app.include_router(puzzle_router, prefix="/api")   # /puzzle/* — Villámtalon rush

# Hungarian Tell-deck card images at /cards (card_piece_<S><R>.jpg + card_back.png).
_CARDS_DIR = Path(__file__).resolve().parents[2] / "assets" / "cards"
app.mount("/cards", StaticFiles(directory=str(_CARDS_DIR)), name="cards")
