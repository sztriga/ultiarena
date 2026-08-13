"""
Perfect-information solver endpoint — POST /api/pis/explore.

"What if?" branch exploration for the analysis board: replay a sequence of moves
on the initial deal, force a specific card, return the optimal continuation.
Contract-aware end to end (betli / parti / durchmars / the weighted "multi").
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ulti.solvers import pis as pis_bridge
from ulti.card import card_from_id

from . import ai_pool
from .serialize import card_to_dict

router = APIRouter()


class PisExploreRequest(BaseModel):
    hands:           List[List[int]]    # 3 lists of card_ids — initial deal
    soloist:         int
    starting_leader: int
    total_tricks:    int                 # legacy; full game only
    moves:           List[int]           # card_ids already played, in order
    forced_card_id:  int                 # card to play instead of the optimal one
    contract:        str = "betli"                  # SOLVE contract (betli/parti/durchmars/multi)
    trump:           Optional[str] = None
    # Full-Ulti extras (default off → unchanged betli behavior). Needed so a branch
    # off a real play-tab deal evaluates the SAME position that was actually played:
    build_contract:    Optional[str] = None         # BUILD contract for build_position
                                                    #   ("parti" when solving "multi"); defaults to `contract`
    talon:             Optional[List[int]] = None   # the 2 set-aside cards
    declare_marriages: bool = False                 # parti/ulti/… declare K+Q marriages
    marriage_restrict: Optional[str] = None         # "40" | "20" for the 100-games
    multi_weights:     Optional[Dict] = None        # set_multi_weights(**…) for "multi"


@router.post("/pis/explore")
def pis_explore(req: PisExploreRequest) -> Dict:
    """Replay the deal up to the user's chosen branch point, force a
    specific card, return the optimal continuation. Stateless — the
    client passes the full position each time. Handles full Ulti (trump,
    marriages, talon, the weighted 'multi' objective) as well as betli."""
    contract = req.contract                           # SOLVE contract (may be "multi")
    build_c  = req.build_contract or contract         # BUILD contract for build_position
    trump    = req.trump
    talon    = [card_from_id(cid) for cid in (req.talon or [])]

    # Validate HERE, in the request thread: building a position and applying moves
    # never solves, so this is cheap and it is what turns a bad request into a 400
    # instead of a worker exception.
    def _fresh():
        p = pis_bridge.build_position(
            hands=[[card_from_id(cid) for cid in h] for h in req.hands],
            soloist=req.soloist, leader=req.starting_leader,
            contract=build_c, trump=trump, talon=list(talon),
            declare_marriages=req.declare_marriages, marriage_restrict=req.marriage_restrict,
        )
        for cid in req.moves:
            card = card_from_id(cid)
            if card not in pis_bridge.legal_actions(p):
                raise HTTPException(status_code=400,
                                    detail=f"Visszajátszási hiba: a(z) {cid} lap itt nem játszható.")
            pis_bridge.apply_move(p, card)
        return p

    pos = _fresh()
    if pis_bridge.is_terminal(pos):
        raise HTTPException(status_code=400, detail="A játszma véget ért — nincs mit elemezni.")

    forced        = card_from_id(req.forced_card_id)
    forced_player = pis_bridge.current_player(pos)
    legal_now     = pis_bridge.legal_actions(pos)
    if forced not in legal_now:
        raise HTTPException(status_code=400,
                            detail=f"A(z) {req.forced_card_id} lap itt nem játszható.")

    n_played = len(req.moves)
    forced_step: Dict = {
        "player_id":      forced_player,
        "card":           card_to_dict(forced),
        "trick_index":    n_played // 3,
        "trick_position": n_played % 3,
        "legal_card_ids": [c.id for c in legal_now],
    }

    # The SEARCH ships to a worker — it holds the GIL for its whole duration, so
    # solving it here would stall every other request in this process, not just
    # this one (ai_pool).
    res = ai_pool.run("explore", {
        "hands": req.hands, "soloist": req.soloist, "leader": req.starting_leader,
        "build_c": build_c, "solve_c": contract, "trump": trump,
        "talon": [c.id for c in talon],
        "declare_marriages": req.declare_marriages, "restrict": req.marriage_restrict,
        "weights": req.multi_weights, "moves": list(req.moves),
        "forced_card_id": req.forced_card_id,
    })

    # Walk the continuation on a fresh position to emit the legal ids per step
    # (no solving — just replay).
    pos_anno = _fresh()
    alt_pv: List[Dict] = [forced_step]
    pis_bridge.apply_move(pos_anno, forced)
    for i, (pid, cid) in enumerate(res["continuation"]):
        plies = n_played + 1 + i
        alt_pv.append({
            "player_id":      pid,
            "card":           card_to_dict(card_from_id(cid)),
            "trick_index":    plies // 3,
            "trick_position": plies % 3,
            "legal_card_ids": [c.id for c in pis_bridge.legal_actions(pos_anno)],
        })
        pis_bridge.apply_move(pos_anno, card_from_id(cid))

    alt_value = res["value"]
    if contract == "betli":
        soloist_takes_alt = int(round(req.total_tricks - alt_value))
        verdict = "soloist" if soloist_takes_alt == 0 else "defenders"
        value_out = float(req.total_tricks - soloist_takes_alt)
    else:
        # trump / parti / multi: soloist-perspective value; >0 favors the soloist.
        soloist_takes_alt = 0
        verdict = "soloist" if alt_value > 0 else "defenders"
        value_out = float(alt_value)
    return {
        "alt_pv":         alt_pv,
        "alt_start":      n_played,
        "value":          value_out,
        "soloist_takes":  soloist_takes_alt,
        "verdict":        verdict,
    }