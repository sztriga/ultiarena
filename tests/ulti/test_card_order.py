"""Card display order is decided in exactly ONE place.

There used to be five: the auction hand, the bid hand, the play table, Villámtalon, and a
re-sort in the web UI — and the UI one ran last, so it quietly overrode the others and the
colourless (10-low) ordering never reached the screen.

Now everything goes through ``ulti.card.sort_hand``. These tests pin both halves of that:
the rule itself, and the architecture — no second implementation anywhere.
"""
from __future__ import annotations

from pathlib import Path

from ulti.card import DECK, Card, sort_hand

_UI = Path(__file__).resolve().parents[2] / "apps" / "web" / "src" / "ui"


def _ranks(cards):
    return [c.rank for c in cards]


def test_colored_order_is_natural_power():
    """Colored games: 7 8 9 J Q K 10 A — the Ten sits just under the ace."""
    hand = [Card(suit="hearts", rank=r) for r in DECK[:8].__class__(
        ["ace", "10", "king", "upper", "lower", "9", "8", "7"])]
    assert _ranks(sort_hand(hand)) == [
        "7", "8", "9", "lower", "upper", "king", "10", "ace"]


def test_colorless_order_demotes_the_ten():
    """Betli / színtelen duri: the Ten drops under the alsó — 7 8 9 10 J Q K A."""
    hand = [Card(suit="hearts", rank=r)
            for r in ("ace", "10", "king", "upper", "lower", "9", "8", "7")]
    assert _ranks(sort_hand(hand, colorless=True)) == [
        "7", "8", "9", "10", "lower", "upper", "king", "ace"]


def test_suit_order_is_piros_first():
    """Display order is piros, tök, zöld, makk — independent of the id encoding."""
    one_each = [Card(suit=s, rank="7") for s in ("acorns", "leaves", "hearts", "bells")]
    assert [c.suit for c in sort_hand(one_each)] == ["hearts", "bells", "leaves", "acorns"]


def test_sorting_is_a_permutation():
    """Ordering must never add, drop or duplicate a card."""
    for colorless in (False, True):
        out = sort_hand(DECK, colorless)
        assert sorted(c.id for c in out) == sorted(c.id for c in DECK)


def test_ui_never_re_sorts_a_hand():
    """The web UI renders the order the API sends. A client-side sort would silently
    override the colourless rule — that is exactly the bug this consolidation fixed."""
    offenders = [p.name for p in _UI.glob("*.ts*") if ".sort(" in p.read_text()]
    assert not offenders, (
        f"{offenders} sort cards client-side; order belongs to ulti.card.sort_hand")


def test_backend_has_one_ordering_rule():
    """No hand-rolled (suit, rank) sort keys left in the API layer."""
    api = Path(__file__).resolve().parents[2] / "apps" / "api"
    bad = []
    for p in api.glob("*.py"):
        text = p.read_text()
        for marker in ("key=lambda c: (c.suit_index", "key=lambda c: c.id"):
            if marker in text:
                bad.append(f"{p.name}: {marker}")
    assert not bad, f"hand-rolled card ordering found: {bad}"
