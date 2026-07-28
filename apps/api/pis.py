"""
Perfect-information solver endpoint.

POST /api/pis/probe
    Generate a (currently betli-only) biased deal, solve it face-up with
    the trickster Cython alpha-beta solver, return the verdict + every
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

from .serialize import card_to_dict

router = APIRouter()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pv_with_legal(
    *,
    initial_hands: List[List[Card]],
    soloist:       int,
    leader:        int,
    contract:      str,
    pv_pairs:      List,
    start_index:   int,
) -> List[Dict]:
    """Walk the PV from the initial position and emit one annotated step
    per ply, including legal-card-ids at the moment of the decision (so
    the UI can render illegal options as opaque)."""
    pos = pis_bridge.build_position(
        hands=initial_hands, soloist=soloist, leader=leader, contract=contract,
    )
    out: List[Dict] = []
    for i, (pid, card) in enumerate(pv_pairs):
        legal_ids = [c.id for c in pis_bridge.legal_actions(pos)]
        plies = start_index + i
        out.append({
            "player_id":      pid,
            "card":           card_to_dict(card),
            "trick_index":    plies // 3,
            "trick_position": plies % 3,
            "legal_card_ids": legal_ids,
        })
        pis_bridge.apply_move(pos, card)
    return out


def _diagnostics(stats: Dict[str, Any], n_tricks: int,
                 best_ms: float, root_ms: float, pv_ms: float,
                 total_ms: float, pv_plies: int) -> Dict[str, Any]:
    """Map trickster.get_stats() into the shape the BetliProbe UI expects.

    The Cython solver doesn't track terminals/max_depth/early_terminations
    individually — they're zeroed for now. nodes/cutoffs/TT counters are
    real."""
    nodes = int(stats.get("nodes_explored", 0))
    cuts  = int(stats.get("cutoffs", 0))
    th    = int(stats.get("tt_hits", 0))
    return {
        "nodes":              nodes,
        "terminals":          0,
        "cutoffs":            cuts,
        "early_terminations": 0,
        "max_depth":          0,
        "cutoff_pct":         (cuts / nodes * 100.0) if nodes else 0.0,
        "tt_hits":            th,
        "tt_stores":          int(stats.get("tt_stores", 0)),
        "tt_hit_pct":         (th / nodes * 100.0) if nodes else 0.0,
        "endgame_cards":      3 * n_tricks,
        "endgame_plies":      3 * n_tricks,
        "best_ms":            best_ms,
        "root_ms":            root_ms,
        "pv_ms":              pv_ms,
        "total_ms":           total_ms,
        "pv_plies":           pv_plies,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Probe — full deal, full solve
# ──────────────────────────────────────────────────────────────────────────────

class PisProbeRequest(BaseModel):
    seed:      Optional[int] = None
    alpha:     float          = Field(0.2, ge=0.0, le=10.0)
    hand_size: int            = Field(10, ge=10, le=10)   # legacy field; full game only


@router.post("/pis/probe")
def pis_probe(req: PisProbeRequest) -> Dict:
    contract = "betli"
    n = 10

    deal = deal_betli(seed=req.seed, alpha=req.alpha)
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    soloist = 0
    leader  = 0   # soloist leads trick 1 in betli (Hungarian Ulti convention)

    pos = pis_bridge.build_position(
        hands=hands, soloist=soloist, leader=leader, contract=contract,
    )

    t0       = time.perf_counter()
    t_best_0 = time.perf_counter()
    best_card, value = pis_bridge.solve_best(pos, contract=contract)
    t_best_ms = (time.perf_counter() - t_best_0) * 1000.0

    # Rebuild a fresh position for solve_all/PV — solve_best mutates internal
    # search state but not the position object itself; still, keep TT-clear
    # behaviour predictable by passing fresh positions to each entry point.
    pos2 = pis_bridge.build_position(
        hands=hands, soloist=soloist, leader=leader, contract=contract,
    )
    t_root_0 = time.perf_counter()
    all_values = pis_bridge.solve_all(pos2, contract=contract)
    t_root_ms = (time.perf_counter() - t_root_0) * 1000.0

    pos3 = pis_bridge.build_position(
        hands=hands, soloist=soloist, leader=leader, contract=contract,
    )
    t_pv_0 = time.perf_counter()
    pv_pairs = pis_bridge.principal_variation(pos3, contract=contract)
    t_pv_ms = (time.perf_counter() - t_pv_0) * 1000.0

    solve_ms = (time.perf_counter() - t0) * 1000.0
    stats    = pis_bridge.get_stats()

    # Drop the betli depth tie-breaker for display.
    soloist_takes = int(round(n - value))
    display_value = float(n - soloist_takes)
    verdict       = "soloist" if soloist_takes == 0 else "defenders"

    pv_annotated = _pv_with_legal(
        initial_hands=hands, soloist=soloist, leader=leader,
        contract=contract, pv_pairs=pv_pairs, start_index=0,
    )

    return {
        "config": {
            "seed":      req.seed,
            "alpha":     req.alpha,
            "hand_size": n,
            "contract":  contract,
        },
        "hands":         [[card_to_dict(c) for c in h] for h in hands],
        "talon":         [card_to_dict(c) for c in deal.talon],
        "soloist":       soloist,
        "leader":        leader,
        "verdict":       verdict,
        "value":         display_value,
        "soloist_takes": soloist_takes,
        "best_lead":     card_to_dict(best_card) if best_card else None,
        "all_leads": [
            {
                "card":  card_to_dict(c),
                "value": float(n - int(round(n - v))),
            }
            for c, v in sorted(all_values.items(), key=lambda kv: -kv[1])
        ],
        "principal_variation": pv_annotated,
        "solve_ms":      solve_ms,
        "diagnostics":   _diagnostics(
            stats, n_tricks=n,
            best_ms=t_best_ms, root_ms=t_root_ms, pv_ms=t_pv_ms,
            total_ms=solve_ms, pv_plies=len(pv_annotated),
        ),
    }


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

class PimcPlayRequest(BaseModel):
    seed:      Optional[int] = None
    alpha:     float          = Field(0.2, ge=0.0, le=10.0)
    n_samples: int            = Field(16,  ge=1, le=512)


_BETLI_WIN_VAL = 10.0


def _is_blunder(side_is_soloist: bool, best_value: float, chosen_value: float) -> bool:
    """For betli: a move is a blunder when it flips the outcome the wrong way."""
    if side_is_soloist:
        return best_value >= _BETLI_WIN_VAL - 1e-6 and chosen_value < _BETLI_WIN_VAL - 1e-6
    return best_value < _BETLI_WIN_VAL - 1e-6 and chosen_value >= _BETLI_WIN_VAL - 1e-6


@router.post("/pis/pimc_play")
def pis_pimc_play(req: PimcPlayRequest) -> Dict:
    contract = "betli"
    deal     = deal_betli(seed=req.seed, alpha=req.alpha)
    hands    = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    soloist, leader = 0, 0

    pos = pis_bridge.build_position(
        hands=hands, soloist=soloist, leader=leader, contract=contract,
    )

    play_log: List[Dict] = []
    n_blunders = 0
    pimc_seed_counter = (req.seed or 0)

    t0 = time.perf_counter()
    while not pis_bridge.is_terminal(pos):
        pid       = pis_bridge.current_player(pos)
        is_solo   = (pid == soloist)
        legal_now = pis_bridge.legal_actions(pos)

        # God's full-info ranking of every legal move.
        god_values = pis_bridge.solve_all(pos, contract=contract)
        if is_solo:
            god_best_value = max(god_values.values())
        else:
            god_best_value = min(god_values.values())

        # PIMC's choice (sees only this player's hand; sample others).
        pimc_seed_counter += 1
        chosen, pimc_avg = pimc_player.pimc_decision(
            true_pos  = pos,
            contract  = contract,
            n_samples = req.n_samples,
            seed      = pimc_seed_counter,
        )
        if chosen is None:
            # Shouldn't happen in non-terminal positions; bail out safely.
            break

        chosen_value = float(god_values.get(chosen, 0.0))
        blunder      = _is_blunder(is_solo, float(god_best_value), chosen_value)
        if blunder:
            n_blunders += 1

        plies = len(play_log)
        play_log.append({
            "player_id":      pid,
            "card":           card_to_dict(chosen),
            "trick_index":    plies // 3,
            "trick_position": plies % 3,
            "legal_card_ids": [c.id for c in legal_now],
            "god_best_value":  float(god_best_value),
            "god_chosen_value": chosen_value,
            "is_blunder":      blunder,
            "pimc_avg":        {c.id: round(v, 3) for c, v in pimc_avg.items()},
        })
        pis_bridge.apply_move(pos, chosen)

    total_ms = (time.perf_counter() - t0) * 1000.0

    # Final outcome via the last position.
    final_values = pis_bridge.solve_all(
        pis_bridge.build_position(
            hands=hands, soloist=soloist, leader=leader, contract=contract,
        ),
        contract=contract,
    ) if False else None   # not needed; we have it from the play log

    # For betli: the last logged play decided the outcome. Read off:
    last_chosen = play_log[-1]["god_chosen_value"] if play_log else 0.0
    soloist_takes = int(round(_BETLI_WIN_VAL - last_chosen))
    verdict = "soloist" if soloist_takes == 0 else "defenders"

    return {
        "config": {
            "seed":      req.seed,
            "alpha":     req.alpha,
            "n_samples": req.n_samples,
            "contract":  contract,
        },
        "hands":         [[card_to_dict(c) for c in h] for h in hands],
        "talon":         [card_to_dict(c) for c in deal.talon],
        "soloist":       soloist,
        "leader":        leader,
        "play_log":      play_log,
        "verdict":       verdict,
        "soloist_takes": soloist_takes,
        "diagnostics": {
            "total_ms":     total_ms,
            "n_decisions":  len(play_log),
            "n_blunders":   n_blunders,
            "blunder_rate": (n_blunders / len(play_log)) if play_log else 0.0,
        },
    }
