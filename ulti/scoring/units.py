"""Kontra units — which parts of a game can be doubled, and what each one means.

A played Ulti game breaks into UNITS that are kontrázható independently: a piros ulti is
"kontra ulti" AND "kontra párti"; a combined game exposes each committed unit separately.
Two consumers need this vocabulary and they must never disagree:

  * ``ulti.scoring.oracle`` — maps each scoring component onto its unit, to apply the
    right multiplier at scoring time (after the game is played);
  * ``apps.api.play`` — offers the defenders their kontra buttons (before it is played).

They used to state it separately, and drifted: play.py offered "kontra párti" on a bid
40-100 while the oracle never scored a párti component there, so the kontra doubled
nothing (milan, from the UI: "it was a 40-100, there is no parti there"). Both now import
from here, so the rule has one home.

Nothing in this module imports the engine or the bidder — it is deliberately a leaf, so
anyone can use it without an import cycle.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Canonical unit ordering — display order, and the tie-break for "which unit first".
UNITS_ORDER: Tuple[str, ...] = ("parti", "ulti", "40_100", "20_100", "durchmars", "betli")

# Scoring components that ride the PÁRTI unit rather than one of their own. Silent /
# def_silent 100s and duris REPLACE the párti, so "kontra párti" scales them too
# (milan 2026-07-05). Silent ulti rides NO unit — unbid, so nobody could kontra it.
PARTI_RIDERS = frozenset({
    "parti", "silent_40_100", "silent_20_100", "silent_durchmars",
    "def_silent_40_100", "def_silent_20_100", "def_silent_durchmars",
})

_OWN_UNIT = frozenset({"40_100", "20_100", "ulti", "betli", "durchmars"})

# Per-unit god objective: (solver, multi-weights, marriage_restrict). How to ask the
# solver "does this unit make?" — the 100s are solved on the multi objective with a
# score_geq_100 weight. Engine knowledge, so it lives with the units, not in the API.
UNIT_OBJECTIVE: Dict[str, Tuple[str, Optional[dict], Optional[str]]] = {
    "parti":     ("parti", None, None),
    "ulti":      ("ulti", None, None),
    "betli":     ("betli", None, None),
    "durchmars": ("durchmars", None, None),
    "40_100":    ("multi", {"score_geq_100": 1.0}, "40"),
    "20_100":    ("multi", {"score_geq_100": 1.0}, "20"),
}


def unit_of(component: str) -> Optional[str]:
    """The kontra unit a scoring component rides, or None if it rides none."""
    if component in PARTI_RIDERS:
        return "parti"
    if component in _OWN_UNIT:
        return component
    return None


def game_has_parti(bid: Any, *, hundred_stands: bool = True) -> bool:
    """Does this game play a card-point párti of its own?

    No, whenever something replaces it:
      * betli / durchmars — not a card-point game at all;
      * a BID 40-100 / 20-100 — the declared 100 IS the point objective and REPLACES
        the párti.

    ``hundred_stands`` exists for the one case where a bid 100 does NOT replace it: the
    soloist bid it without actually holding the marriage, so the 100 never scores and the
    párti falls back in. Only the scoring path can know that (it needs the played hand),
    and it passes ``hundred_stands=bid_a_100``. The kontra-offer path leaves it True —
    the bidder and the human bid path both refuse a 100 without its marriage.
    """
    if bid.betli or bid.durchmars:
        return False
    if bid.forty_hundred or bid.twenty_hundred:
        return not hundred_stands
    return True


def kontra_units(bid: Any) -> List[str]:
    """Every kontra-able unit of this game, in UNITS_ORDER — each kontrázható separately.

    A simple game has one; a combined game exposes each committed unit. A bid ulti also
    exposes the card-point párti (milan: a piros ulti → kontra ulti AND kontra párti).
    """
    if bid.betli:
        return ["betli"]
    units: List[str] = []
    if bid.ulti:
        units.append("ulti")
    if bid.durchmars:
        units.append("durchmars")
    if bid.forty_hundred:
        units.append("40_100")
    if bid.twenty_hundred:
        units.append("20_100")
    if game_has_parti(bid):
        units.append("parti")
    return sorted(units, key=UNITS_ORDER.index)
