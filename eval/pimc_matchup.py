"""Shared PIMC ↔ god matchup primitives for the research pipeline.

The defender-side correction is centralised here so callers cannot
accidentally run anti-optimal PIMC defense. ``pimc_decision`` returns
``argmax`` (the soloist-perspective best move); a defender uses the
*minimum* averaged soloist value across the sampled determinizations.

Public API:
    pimc_pick(...)   — defender-aware PIMC move picker
    god_pick(...)    — god solver (already player-perspective-aware)
    play_one(...)    — play one head-to-head match with named strategies

Each experiment glues these into its own α-sweep loop (typically inside a
``multiprocessing.Pool`` since games are independent and the Cython
transposition table is process-global).

History
-------
Before this module existed, several experiment scripts called
``solvers.pimc.pimc_decision`` directly and used the returned ``chosen``
move as-is for both seats. ``chosen`` is always ``argmax`` over the
soloist-perspective averaged values, so PIMC defenders were systematically
playing the *worst* move available — anti-optimal. The bug skewed any
"PIMC defender vs X" comparison; ``experiments/10_betli_pimc_audit``
documents the impact (~36pp under-reported betli def_stop at α=0.7).
"""
from __future__ import annotations

import random
from typing import Any, Callable, Optional

from solvers import determinize as _det
from solvers import pimc as _pimc
from solvers import pis as pis_bridge
from ulti.card import Card


# ──────────────────────────────────────────────────────────────────────────────
# Per-strategy move pickers
# ──────────────────────────────────────────────────────────────────────────────

def god_pick(*, pos: Any, contract: str, **_unused) -> Card:
    """God solver pick. ``solve_best`` is already player-perspective-aware
    (MAX for soloist, MIN for defenders), so no flip needed."""
    chosen, _ = pis_bridge.solve_best(pos, contract=contract)
    return chosen


def pimc_pick(
    *,
    pos: Any,
    contract: str,
    n_samples: int,
    seed: int,
    voids_dict: Optional[dict] = None,
    must_hold: Optional[dict] = None,
) -> Optional[Card]:
    """PIMC pick that correctly minimises soloist value for defenders.

    ``solvers.pimc.pimc_decision`` returns ``(argmax, averaged)`` — the
    soloist-perspective best move. For seats 1 and 2 we flip to ``argmin``.

    ``must_hold`` pins known cards to a player in every sampled world (``None`` =
    the contract's registered rule). A terített-game defender passes
    ``{0: <soloist's revealed remaining hand>}`` so the open hand is fixed, not sampled.
    """
    chosen, averaged = _pimc.pimc_decision(
        true_pos=pos, contract=contract, n_samples=n_samples,
        seed=seed, voids=voids_dict, must_hold=must_hold,
    )
    viewer = pis_bridge.current_player(pos)
    if viewer != 0 and averaged:
        chosen = min(averaged, key=lambda c: averaged[c])
    return chosen


def random_pick(*, pos: Any, rng: random.Random, **_unused) -> Card:
    return rng.choice(pis_bridge.legal_actions(pos))


# ──────────────────────────────────────────────────────────────────────────────
# Match runner
# ──────────────────────────────────────────────────────────────────────────────

def _pick(
    *,
    strategy: str, pos: Any, contract: str,
    voids_dict: dict, pimc_n: int, seed: int, rng: random.Random,
) -> Optional[Card]:
    if strategy == "god":
        return god_pick(pos=pos, contract=contract)
    if strategy == "pimc":
        return pimc_pick(
            pos=pos, contract=contract, n_samples=pimc_n,
            seed=seed, voids_dict=voids_dict,
        )
    if strategy == "random":
        return random_pick(pos=pos, rng=rng)
    raise ValueError(f"unknown strategy {strategy!r}")


def play_one(
    *,
    deal: Any,
    contract: str,
    sol_strategy: str,
    def_strategy: str,
    pimc_n: int = 0,
    seed: int = 0,
    soloist: int = 0,
    leader: int = 0,
) -> Any:
    """Play one head-to-head game. Returns the final ``GameState`` so the
    caller can inspect ``captured`` / ``scores`` to compute the outcome
    (each contract has its own win condition).

    ``deal`` is a :class:`eval.dojo.BiasedDeal`. Talon discards are
    forwarded as a separate ``talon`` to :func:`solvers.pis.build_position`
    so the position starts at trick 1 with the correct hand sizes.
    """
    hands = [list(deal.sol_hand), list(deal.def1_hand), list(deal.def2_hand)]
    pos = pis_bridge.build_position(
        hands=hands, soloist=soloist, leader=leader, contract=contract,
        trump=getattr(deal, "trump", None), talon=list(deal.talon),
    )
    voids = _det.Voids()
    voids_dict = voids.as_dict()
    rng = random.Random(seed)

    move_i = 0
    while not pis_bridge.is_terminal(pos):
        p = pis_bridge.current_player(pos)
        strat = sol_strategy if p == soloist else def_strategy
        chosen = _pick(
            strategy=strat, pos=pos, contract=contract,
            voids_dict=voids_dict, pimc_n=pimc_n,
            seed=seed * 31337 + move_i, rng=rng,
        )
        if chosen is None:
            chosen = rng.choice(pis_bridge.legal_actions(pos))
        voids.observe(pos, p, chosen)
        voids_dict.clear(); voids_dict.update(voids.as_dict())
        pis_bridge.apply_move(pos, chosen)
        move_i += 1
    return pos


# ──────────────────────────────────────────────────────────────────────────────
# Per-contract god-label + outcome helpers
# ──────────────────────────────────────────────────────────────────────────────
# Each fundamental contract has a binary outcome at α-sweep granularity.
# Centralise the boundary so experiments don't reinvent the thresholds.

# Binary contracts (ulti / betli / durchmars): solver returns 10 = sol
# WIN, 0 = LOSE. Anything above 5 counts as a win prediction.
_BINARY_WIN_VAL = 1.0

# Parti, like the other binary contracts, now uses the uniform 0/10
# solver scale (10 = sol won, 0 = sol lost). The win comparison is
# therefore the same as for ulti/betli/durchmars.


def god_says_soloist_wins(pos: Any, contract: str) -> bool:
    """Does god predict a soloist win from ``pos``?"""
    vals = pis_bridge.solve_all(pos, contract=contract)
    if not vals:
        return False
    best = max(vals.values())
    return best > _BINARY_WIN_VAL / 2 - 1e-6


def defenders_won(pos: Any, contract: str, soloist: int = 0) -> bool:
    """Did the defenders win the *played-out* position?

    Per-contract win condition:
        betli      — soloist must take 0 tricks; def wins iff sol took ≥1
        durchmars  — soloist must take all 10 tricks; def wins iff sol < 10
        ulti       — soloist must take last trick with trump-7; def wins iff not
        parti      — sol must score ≥ 50 trick points (delegates to
                     trickster's ``soloist_won_simple``, which also
                     credits the talon's points to the defenders per the
                     rulebook).
    """
    if contract == "betli":
        return len(pos.captured[soloist]) > 0
    if contract == "durchmars":
        return len(pos.captured[soloist]) < 30
    if contract == "ulti":
        # Defenders win if soloist did NOT take trick 10 with trump-7.
        from ultisolver.games.ulti.game import last_trick_ulti_check
        side, won = last_trick_ulti_check(pos)
        return not (side == "soloist" and won)
    if contract == "parti":
        from ultisolver.games.ulti.game import soloist_won_simple
        return not soloist_won_simple(pos)
    raise ValueError(f"unknown contract {contract!r}")
