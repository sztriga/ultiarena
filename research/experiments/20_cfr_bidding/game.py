"""Pure bidding-tree logic for the standard Ulti auction.

State is the round-robin action history: a tuple of (pid, action) where
action is one of ACTIONS or PASS. Turn order is strict round-robin from P0;
the current holder auto-passes its own turn; 3 consecutive passes after a bid
end the auction. Everything else (holder, level, terminality, payoffs) is
derived from the history, so states are hashable and the infoset key is just
(bucket, history).

A `ctx` (deal context) supplies:
    ev[player][action]    god-exact EV/def if `player` solos `action` (float)
    avail[player][action] bool, deployable availability from the raw 10
    bucket[player]        int, the player's hand bucket (infoset key only)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from common import ACTIONS, ACTION_INDEX, RANK, RANK_TO_ACTION, PASS, N_ACTIONS

PASS_OUT_PAYOFF = (-4.0, 2.0, 2.0)   # P0 forced-opener penalty (2×-2 / +2 / +2)


# ── derivations from history ─────────────────────────────────────────────────

def to_move(hist: Tuple) -> int:
    return len(hist) % 3


def _holder_level(hist: Tuple):
    holder, level = None, 0
    for pid, a in hist:
        if a != PASS:
            holder, level = pid, RANK[a]
    return holder, level


def _trailing_passes(hist: Tuple) -> int:
    n = 0
    for pid, a in reversed(hist):
        if a == PASS:
            n += 1
        else:
            break
    return n


def is_pass_out(hist: Tuple) -> bool:
    return len(hist) >= 1 and hist[0][1] == PASS


def is_terminal(hist: Tuple) -> bool:
    if is_pass_out(hist):
        return True
    holder, _ = _holder_level(hist)
    return holder is not None and _trailing_passes(hist) >= 3


def legal_actions(hist: Tuple, ctx) -> List[str]:
    """Actions available to the player on the move."""
    p = to_move(hist)
    av = ctx['avail'][p]
    if len(hist) == 0:                       # P0 open
        return [PASS] + [a for a in ACTIONS if av[ACTION_INDEX[a]]]
    holder, level = _holder_level(hist)
    if p == holder:                          # holder auto-passes its own turn
        return [PASS]
    return [PASS] + [a for a in ACTIONS
                     if RANK[a] > level and av[ACTION_INDEX[a]]]


def apply(hist: Tuple, action: str) -> Tuple:
    return hist + ((to_move(hist), action),)


def payoffs(hist: Tuple, ctx) -> Tuple[float, float, float]:
    """Zero-sum GP vector [P0, P1, P2] at a terminal history."""
    if is_pass_out(hist):
        return PASS_OUT_PAYOFF
    holder, level = _holder_level(hist)
    action = RANK_TO_ACTION[level]
    ev = float(ctx['ev'][holder][ACTION_INDEX[action]])
    out = [-ev, -ev, -ev]
    out[holder] = 2.0 * ev
    return tuple(out)


def infoset_key(hist: Tuple, ctx) -> Tuple:
    """Key for the player on the move: (bucket, availability, public history).

    Availability (ulti / ulti_piros — the only variable ones) is part of the
    key so that every state sharing an infoset has the SAME legal-action set;
    a bucket alone does not pin down availability."""
    p = to_move(hist)
    av = ctx['avail'][p]
    return (int(ctx['bucket'][p]), (bool(av[1]), bool(av[4])), hist)
