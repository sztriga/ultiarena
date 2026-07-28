"""AI-only play restrictions — heuristic filters applied during training/eval/API.

These are NOT game rules (those live in rules.py / game.py).  They are soft
strategic guides that narrow the action space for the AI.  If all cards get
filtered out, the unrestricted set is used as fallback.
"""

from __future__ import annotations

from typing import Callable

from trickster.games.ulti.cards import BETLI_STRENGTH, Card
from trickster.games.ulti.game import GameState

Restriction = Callable[[GameState, int, list[Card]], list[Card]]


def apply_restrictions(
    restrictions: list[Restriction],
    gs: GameState,
    player: int,
    cards: list[Card],
) -> list[Card]:
    """Apply AI restrictions sequentially with safety fallback."""
    filtered = list(cards)
    for fn in restrictions:
        result = fn(gs, player, filtered)
        if result:  # never filter to empty
            filtered = result
    return filtered


# ---------------------------------------------------------------------------
#  Built-in restrictions
# ---------------------------------------------------------------------------


def betli_play_highest(gs: GameState, player: int, cards: list[Card]) -> list[Card]:
    """Betli soloist: only play the highest card per suit to shed danger."""
    if not gs.betli or player != gs.soloist or gs.training_mode == "durchmars":
        return cards

    # Group by suit, keep only the highest-strength card per suit
    best: dict[str, Card] = {}
    for c in cards:
        key = c.suit
        if key not in best or BETLI_STRENGTH[c.rank] > BETLI_STRENGTH[best[key].rank]:
            best[key] = c
    return list(best.values())


DEFAULT_AI_RESTRICTIONS: list[Restriction] = [betli_play_highest]
