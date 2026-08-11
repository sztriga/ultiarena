"""The PUBLIC API — /api/v1 (docs/PUBLIC_API.md; decisions D1-D4 confirmed by milan
2026-08-11: keys per account, bot games recorded by default, engine-not-weights).

A separate, versioned FastAPI sub-application with its own OpenAPI docs at
/api/v1/docs. Everything here is a curated veneer over machinery that already
exists — the rules kernel over the pis bridge + scoring oracle, the dataset over
games.db, and MATCHES over the multiplayer Session (an external agent occupies a
seat exactly like a human player: per-viewer snapshots make cheating structurally
impossible). No second engine, no duplicated rules — ever.

Cost classes (D3): `free` = rules kernel + dataset reads; `metered` = match
creation/steps (each step may run real AI in the empty chairs).
"""
from __future__ import annotations

import random
import sqlite3
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ulti.bidding.deal import deal_12_10_10
from ulti.bidding.ladder import overcalls, contract_name
from ulti.card import RANK_HU, RANKS, SUIT_HU, SUITS, card_from_id
from ulti.scoring.oracle import score as oracle_score
from ulti.scoring.units import kontra_units
from ulti.solvers import pis as pis_bridge

from . import play as play_routes
from . import recording
from .apikeys import check_key_rate, require_key
from .apikeys import router as keys_router
from .engine import Session, _sessions, _sessions_lock
from .auction_flow import _advance_auction

v1 = FastAPI(
    title="UltiArena public API",
    version="1.0",
    # /api/v1/docs = ReDoc (the tasteful reference page — milan vetoed Swagger's
    # default skin); /api/v1/console = Swagger UI, kept for its try-it-out.
    redoc_url="/docs",
    docs_url="/console",
    description="""
Programmatic access to UltiArena: recorded games, the rules kernel
(deal / legal moves / exact scoring), and live matches where your agent
plays a seat against the site's AI.

## Getting started

1. Register an account at ultiarena.hu.
2. Mint an API key (one time):

```
TOKEN=$(curl -s -X POST https://ultiarena.hu/api/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{"username":"NAME","password":"PASSWORD"}' | jq -r .token)

curl -s -X POST https://ultiarena.hu/api/v1/keys \\
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \\
  -d '{"name":"my-first-key"}'
```

The response contains your key (`ua_…`). It is shown **only once** — store it.

3. Send the key with every request:

```
curl -s -X POST https://ultiarena.hu/api/v1/deal \\
  -H "Content-Type: application/json" -H "Authorization: Bearer ua_..." \\
  -d '{"seed":42}'
```

## Limits

Per key: **120 req/min** on cheap endpoints (kernel, dataset, match polling),
**30 req/min** where the AI has to think (match creation and steps).

Finished matches are recorded into the public dataset under your account
and agent name.

## Cards

A card is a number from 0 to 31: `suit = id // 8`, `rank = id % 8`.

""" + "suits: " + ", ".join(f"{i} = {SUIT_HU[s]} ({s})" for i, s in enumerate(SUITS))
    + "  \nranks: " + ", ".join(f"{i} = {RANK_HU[r]}" for i, r in enumerate(RANKS)) + "\n",
)
v1.include_router(keys_router)

# name → BidSet for every game on the ladder (the wire names are the display names,
# e.g. "piros ulti", "40-100-duri", "rebetli").
_BIDS: Dict[str, object] = {}
for _r in overcalls(None):
    for _b in _r.bids:
        _BIDS.setdefault(contract_name(_b), _b)


# ── Rules kernel (class: free) ──────────────────────────────────────────────────

class DealRequest(BaseModel):
    seed: Optional[int] = Field(None, description="Deterministic redeal; omit for random.")


@v1.post("/deal", tags=["kernel"])
def deal(req: DealRequest, request: Request = None) -> dict:
    """A fresh deal: 12 cards to the forehand (seat 0, incl. the 2-card talon),
    10-10 to the others — exactly the app's dealer."""
    ident = require_key(request)
    check_key_rate(ident["key_id"], "free")
    seed = req.seed if req.seed is not None else random.randint(1, 2**31 - 1)
    sol12, d1, d2 = deal_12_10_10(seed)
    return {"seed": seed,
            "forehand12": [c.id for c in sol12],
            "hands": [[c.id for c in sol12[:10]], [c.id for c in d1], [c.id for c in d2]],
            "talon": [c.id for c in sol12[10:]]}


class PositionRequest(BaseModel):
    """A play position = the deal in PLAY space (soloist first) + the moves so far."""
    hands: List[List[int]] = Field(..., description="3 hands of card ids, soloist FIRST")
    talon: List[int] = Field(..., description="the 2 buried cards")
    contract: str = Field(..., description='ladder name, e.g. "piros ulti", "betli"')
    trump: Optional[str] = Field(None, description="hearts/acorns/leaves/bells; null for colorless")
    plays: List[int] = Field(default_factory=list, description="card ids in play order")


def _rebuild(req: PositionRequest):
    bid = _BIDS.get(req.contract)
    if bid is None:
        raise HTTPException(status_code=400,
                            detail=f"ismeretlen kontraktus: {req.contract!r}")
    if bid.betli:
        build_c, trump, restrict = "betli", None, None
    elif bid.durchmars and not (bid.ulti or bid.forty_hundred or bid.twenty_hundred) \
            and req.trump is None:
        build_c, trump, restrict = "durchmars", None, None
    else:
        if req.trump is None:
            raise HTTPException(status_code=400, detail="ehhez a játékhoz adu kell")
        build_c, trump = "parti", req.trump
        restrict = "40" if bid.forty_hundred else ("20" if bid.twenty_hundred else None)
    if sorted(len(h) for h in req.hands) != [10, 10, 10] or len(req.talon) != 2:
        raise HTTPException(status_code=400, detail="3×10 lap + 2 talon kell")
    pos = pis_bridge.build_position(
        hands=[[card_from_id(c) for c in h] for h in req.hands],
        soloist=0, leader=0, contract=build_c, trump=trump,
        talon=[card_from_id(c) for c in req.talon],
        declare_marriages=(trump is not None), marriage_restrict=restrict,
        has_ulti=bool(bid.ulti))
    for cid in req.plays:
        card = card_from_id(cid)
        if card not in pis_bridge.legal_actions(pos):
            raise HTTPException(status_code=400, detail=f"illegális lépés: {cid}")
        pis_bridge.apply_move(pos, card)
    return pos, bid


@v1.post("/legal", tags=["kernel"])
def legal(req: PositionRequest, request: Request = None) -> dict:
    """Whose turn it is and which cards they may play, after replaying `plays`."""
    ident = require_key(request)
    check_key_rate(ident["key_id"], "free")
    pos, _bid = _rebuild(req)
    terminal = pis_bridge.is_terminal(pos)
    return {"terminal": terminal,
            "current_player": None if terminal else pis_bridge.current_player(pos),
            "legal_card_ids": [] if terminal else
                [c.id for c in pis_bridge.legal_actions(pos)]}


class ScoreRequest(PositionRequest):
    kontras: Dict[str, int] = Field(default_factory=dict,
                                    description="unit → level (1=kontra, 2=rekontra), e.g. {'ulti': 1}")


@v1.post("/score", tags=["kernel"])
def score(req: ScoreRequest, request: Request = None) -> dict:
    """Exact scoring of a FINISHED deal — the same oracle the site uses (silent
    games included). seat_gp is play-index space: [soloist, defender1, defender2]."""
    ident = require_key(request)
    check_key_rate(ident["key_id"], "free")
    pos, bid = _rebuild(req)
    if not pis_bridge.is_terminal(pos):
        raise HTTPException(status_code=400, detail="a játszma nem ért véget")
    units = set(kontra_units(bid))
    bad = set(req.kontras) - units
    if bad:
        raise HTTPException(status_code=400,
                            detail=f"nem kontrázható egység: {sorted(bad)} (lehet: {sorted(units)})")
    pvec = oracle_score(final_pos=pos, bid=bid,
                        kontras={u: int(v) for u, v in req.kontras.items()})
    per_def = [-float(pvec.gp_vs(0)), -float(pvec.gp_vs(1))]
    return {"contract": req.contract,
            "seat_gp": [float(pvec.total_sol), per_def[0], per_def[1]],
            "components": {k: float(v) for k, v in pvec.components.items() if v != 0}}


# ── Dataset (class: free) ───────────────────────────────────────────────────────

def _games_db() -> sqlite3.Connection:
    recording._db()                       # ensure the file + schema exist
    return sqlite3.connect(f"file:{recording._DB_PATH}?mode=ro", uri=True)


@v1.get("/games", tags=["dataset"])
def games(request: Request = None, contract: Optional[str] = None,
          since: Optional[float] = None, limit: int = 100,
          cursor: Optional[float] = None) -> dict:
    """Recorded games, newest first (metadata only — fetch one by id for the full
    transcript). `cursor` = the last row's created_at from the previous page."""
    ident = require_key(request)
    check_key_rate(ident["key_id"], "free")
    limit = max(1, min(500, limit))
    q = "SELECT id, created_at, contract, trump, soloist_seat, kontra_level, winner, made FROM games"
    cond, args = [], []
    if contract:
        cond.append("contract = ?"); args.append(contract)
    if since:
        cond.append("created_at >= ?"); args.append(since)
    if cursor:
        cond.append("created_at < ?"); args.append(cursor)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
    con = _games_db()
    rows = con.execute(q, args).fetchall()
    con.close()
    out = [{"id": r[0], "created_at": r[1], "contract": r[2], "trump": r[3],
            "soloist_seat": r[4], "kontra_level": r[5], "winner": r[6],
            "made": bool(r[7])} for r in rows]
    return {"games": out,
            "next_cursor": out[-1]["created_at"] if len(out) == limit else None}


@v1.get("/games/{game_id}", tags=["dataset"])
def game(game_id: str, request: Request = None) -> dict:
    """One game's full record: deal, auction, every play, kontra, scores, players
    (user ids for humans/bots, 'frontier' for the AI)."""
    import json as _json
    ident = require_key(request)
    check_key_rate(ident["key_id"], "free")
    con = _games_db()
    r = con.execute("SELECT id, created_at, seed, contract, trump, soloist_seat, "
                    "human_seat, kontra_level, winner, made, seat_gp, players, transcript "
                    "FROM games WHERE id = ?", (game_id,)).fetchone()
    con.close()
    if r is None:
        raise HTTPException(status_code=404, detail="nincs ilyen játszma")
    return {"id": r[0], "created_at": r[1], "seed": r[2], "contract": r[3],
            "trump": r[4], "soloist_seat": r[5], "human_seat": r[6],
            "kontra_level": r[7], "winner": r[8], "made": bool(r[9]),
            "seat_gp": _json.loads(r[10]), "players": _json.loads(r[11]),
            "transcript": _json.loads(r[12])}


# ── Matches — the environment (class: metered) ──────────────────────────────────

class MatchRequest(BaseModel):
    seats: List[str] = Field(default=["me", "frontier", "frontier"],
                             description='3 entries of "me" / "frontier"; exactly ONE "me" '
                                         '(a key is one seat identity — self-play needs a second key)')
    seed: Optional[int] = None
    agent: str = Field("agent", max_length=40,
                       description="your agent's name — recorded with the game")


@v1.post("/matches", tags=["matches"])
def match_create(req: MatchRequest, request: Request = None) -> dict:
    """Deal a match. Your agent occupies its seat(s) EXACTLY like a human player:
    poll GET /matches/{id} for your view (never anyone else's cards), act with
    POST /matches/{id}/act when it's your turn. Empty chairs are the frontier AI.
    Finished deals are recorded into the public dataset under your account (D4)."""
    ident = require_key(request)
    check_key_rate(ident["key_id"], "metered")
    if len(req.seats) != 3 or any(s not in ("me", "frontier") for s in req.seats) \
            or req.seats.count("me") != 1:
        raise HTTPException(status_code=400,
                            detail='seats: 3× "me"/"frontier", pontosan egy "me"')
    humans = {i for i, s in enumerate(req.seats) if s == "me"}
    seed = req.seed if req.seed is not None else random.randint(1, 2**31 - 1)
    sess = Session(seat=min(humans), seed=seed, humans=humans)
    sess.live = True                       # per-viewer snapshots + bearer-seat auth
    sess.api_agent = req.agent.strip() or "agent"
    sess.players = {s: {"user_id": ident["user_id"], "username": ident["username"]}
                    for s in humans}
    sess.players_by_user = {ident["user_id"]: min(humans)}
    sess.owner_ip = f"api:{ident['key_id']}"        # recording gate + attribution
    with _sessions_lock:
        _sessions[sess.id] = sess
    _advance_auction(sess)                 # frontier chairs may act immediately
    sess.rev += 1
    return {"match_id": sess.id, "seed": seed,
            "my_seats": sorted(humans), "rev": sess.rev}


@v1.get("/matches/{match_id}", tags=["matches"])
def match_state(match_id: str, request: Request = None) -> dict:
    """Your seat's snapshot — the same shape the web client plays from. Key fields:
    phase, auction.is_human_turn / legal_bids, legal_card_ids, kontra.is_human_turn,
    result.seat_gp, rev (changes on every server-side mutation — poll on it)."""
    ident = require_key(request)
    check_key_rate(ident["key_id"], "free")     # polling must be cheap
    return play_routes.play_state(play_routes.StateRequest(game_id=match_id), request)


class ActRequest(BaseModel):
    type: str = Field(..., description="bid | pass | pickup | trump | kontra | move")
    rung_index: Optional[int] = None
    bid_index: Optional[int] = None
    trump: Optional[str] = None
    discard_ids: List[int] = Field(default_factory=list)
    units: Optional[List[str]] = None
    card_id: Optional[int] = None


@v1.post("/matches/{match_id}/act", tags=["matches"])
def match_act(match_id: str, req: ActRequest, request: Request = None) -> dict:
    """One action at your turn. Dispatches to the SAME handlers the web app uses —
    identical validation, identical snapshots."""
    ident = require_key(request)
    check_key_rate(ident["key_id"], "metered")
    P = play_routes
    if req.type == "bid":
        if req.rung_index is None:
            raise HTTPException(status_code=400, detail="bid: rung_index kell")
        return P.play_bid(P.BidRequest(game_id=match_id, rung_index=req.rung_index,
                                       bid_index=req.bid_index, trump=req.trump,
                                       discard_ids=req.discard_ids), request)
    if req.type == "pass":
        return P.play_pass(P.PassRequest(game_id=match_id,
                                         discard_ids=req.discard_ids), request)
    if req.type == "pickup":
        return P.play_pickup(P.PickupRequest(game_id=match_id), request)
    if req.type == "trump":
        if not req.trump:
            raise HTTPException(status_code=400, detail="trump: szín kell")
        return P.play_trump(P.TrumpRequest(game_id=match_id, trump=req.trump), request)
    if req.type == "kontra":
        return P.play_kontra(P.KontraRequest(game_id=match_id, units=req.units or []),
                             request)
    if req.type == "move":
        if req.card_id is None:
            raise HTTPException(status_code=400, detail="move: card_id kell")
        return P.play_move(P.MoveRequest(game_id=match_id, card_id=req.card_id), request)
    raise HTTPException(status_code=400, detail=f"ismeretlen akció: {req.type!r}")
