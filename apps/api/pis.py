"""
Perfect-information solver endpoint.

POST /api/pis/probe
    Generate a (currently betli-only) biased deal, solve it face-up with
    the ultisolver Cython alpha-beta solver, return the verdict + every
    legal opening move's value + the principal variation + diagnostics.

POST /api/pis/explore
    "What if?" branch exploration. Replays a sequence of moves on the
    initial deal, then forces a specific card and returns the optimal
    continuation.

Today both endpoints solve only Betli (full 10-card, no trimming — the
Cython solver is hardcoded to 10 tricks). The plumbing is contract-aware
end to end (``solvers.pis`` accepts a ``contract`` arg), so adding the
contract dropdown later is just exposing it through the request schema.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ulti.eval.dojo import deal_betli
from ulti.solvers import pis as pis_bridge
from ulti.solvers import pimc as pimc_player
from ulti.card import Card, card_from_id

from . import ai_pool
from .serialize import card_to_dict

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────



# ──────────────────────────────────────────────────────────────────────────────
# Branch exploration
# ──────────────────────────────────────────────────────────────────────────────

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
    with ai_pool.solver_lock:
        contract = req.contract                          # SOLVE contract (may be "multi")
        build_c  = req.build_contract or contract         # BUILD contract for build_position
        trump    = req.trump
        hands    = [[card_from_id(cid) for cid in h] for h in req.hands]
        talon    = [card_from_id(cid) for cid in (req.talon or [])]

        if req.multi_weights:
            from ultisolver._solver_core import set_multi_weights
            set_multi_weights(**req.multi_weights)

        def _fresh():
            p = pis_bridge.build_position(
                hands=[list(h) for h in hands], soloist=req.soloist, leader=req.starting_leader,
                contract=build_c, trump=trump, talon=list(talon),
                declare_marriages=req.declare_marriages, marriage_restrict=req.marriage_restrict,
            )
            for cid in req.moves:
                card = card_from_id(cid)
                if card not in pis_bridge.legal_actions(p):
                    raise HTTPException(status_code=400,
                                        detail=f"Replay failed at card id {cid}: not legal here.")
                pis_bridge.apply_move(p, card)
            return p

        pos = _fresh()
        if pis_bridge.is_terminal(pos):
            raise HTTPException(status_code=400, detail="Game already over — nothing to explore.")

        forced        = card_from_id(req.forced_card_id)
        forced_player = pis_bridge.current_player(pos)
        legal_now     = pis_bridge.legal_actions(pos)
        if forced not in legal_now:
            raise HTTPException(
                status_code=400,
                detail=f"Card {req.forced_card_id} is not a legal play for player {forced_player} here.",
            )

        n_played = len(req.moves)
        forced_step: Dict = {
            "player_id":      forced_player,
            "card":           card_to_dict(forced),
            "trick_index":    n_played // 3,
            "trick_position": n_played % 3,
            "legal_card_ids": [c.id for c in legal_now],
        }

        # Apply the forced card, then walk the PV from the resulting position.
        pis_bridge.apply_move(pos, forced)
        continuation = pis_bridge.principal_variation(pos, contract=contract)

        # For the alt-PV display, rebuild a fresh position, replay moves + forced,
        # then walk the continuation emitting legal ids per step.
        pos_anno = _fresh()
        alt_pv: List[Dict] = [forced_step]
        pis_bridge.apply_move(pos_anno, forced)
        for i, (pid, card) in enumerate(continuation):
            plies = n_played + 1 + i
            alt_pv.append({
                "player_id":      pid,
                "card":           card_to_dict(card),
                "trick_index":    plies // 3,
                "trick_position": plies % 3,
                "legal_card_ids": [c.id for c in pis_bridge.legal_actions(pos_anno)],
            })
            pis_bridge.apply_move(pos_anno, card)

        # Outcome value of the alt branch.
        pos_eval = _fresh()
        pis_bridge.apply_move(pos_eval, forced)
        _, alt_value = pis_bridge.solve_best(pos_eval, contract=contract)

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


# ──────────────────────────────────────────────────────────────────────────────
# PIMC play-through with god annotations
# ──────────────────────────────────────────────────────────────────────────────
# All three seats are played by PIMC (each seeing only its own info set).
# At every ply the god solver is consulted on the *true* full-info position
# to label the chosen move as a blunder or not.
#
# Blunder (for betli, binary outcome):
#     soloist on move: best_value == 10 (could have won)
#                      but chosen_value < 10 (PIMC's pick loses).
#     defender on move: best_value < 10 (defenders could win)
#                      but chosen_value == 10 (PIMC threw it away).
# In other words: the move flipped a previously-decided outcome the wrong way.



_BETLI_WIN_VAL = 10.0




