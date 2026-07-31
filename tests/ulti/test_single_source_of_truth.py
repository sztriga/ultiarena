"""Domain rules must have exactly one home.

Every bug in this area has had the same shape: a rule the engine already knows, restated
somewhere else, and the two drifting apart —

  * the colourless 10-low ordering existed in seven places, and the web UI's copy silently
    overrode the backend's so betli hands rendered in the wrong order;
  * "a bid 100 replaces the párti" lived in both the scoring oracle and the kontra-offer
    path, and only the oracle had it right, so defenders were offered a kontra that
    doubled nothing.

These tests fail when a rule sprouts a second definition, rather than waiting for the two
copies to disagree in a game.

Where a copy is UNAVOIDABLE — ``ultisolver`` is a standalone package that deliberately does
not import ``ulti``, and its Cython hot path cannot call Python at all — the copy is pinned
against the canonical one instead of removed.
"""
from __future__ import annotations

import re
from pathlib import Path

from ulti.card import COLORLESS_RANK

_ROOT = Path(__file__).resolve().parents[2]

# The literal colourless table, in either quote style. Anything matching outside the one
# canonical definition (and the solver package's own copy) is a new duplicate.
_TABLE_RE = re.compile(r"""['"]lower['"]\s*:\s*4\s*,\s*['"]upper['"]\s*:\s*5""")


def test_solver_package_agrees_with_the_canonical_rank_order():
    """ultisolver keeps its own copy (it must not import ulti) — so pin it here."""
    from ultisolver.games.ulti.cards import BETLI_STRENGTH, Rank

    # Rank enum → the rank names ulti.card uses.
    names = {Rank.SEVEN: "7", Rank.EIGHT: "8", Rank.NINE: "9", Rank.TEN: "10",
             Rank.JACK: "lower", Rank.QUEEN: "upper", Rank.KING: "king", Rank.ACE: "ace"}
    solver_order = {names[rank]: value for rank, value in BETLI_STRENGTH.items()}
    assert solver_order == COLORLESS_RANK, (
        "ultisolver's betli ordering drifted from ulti.card.COLORLESS_RANK — "
        "the solver and the rest of the app would rank colourless hands differently")


def test_no_new_copies_of_the_colorless_rank_table():
    """One definition in ulti/, one in the standalone solver package. No others."""
    allowed = {
        _ROOT / "ulti" / "card.py",                       # the canonical definition
        _ROOT / "ultisolver" / "games" / "ulti" / "cards.py",   # standalone package
        Path(__file__),                                   # this test's own regex
    }
    offenders = []
    for path in list((_ROOT / "ulti").rglob("*.py")) + \
                list((_ROOT / "apps").rglob("*.py")) + \
                list((_ROOT / "ultisolver").rglob("*.py")) + \
                list((_ROOT / "tests").rglob("*.py")):
        if path in allowed:
            continue
        if _TABLE_RE.search(path.read_text()):
            offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, (
        f"colourless rank order re-defined in {offenders} — "
        f"import ulti.card.COLORLESS_RANK instead")


def test_cython_rank_order_is_bound_by_behaviour():
    """The Cython _str_b cannot import Python, so it is bound behaviourally instead.

    ulti.solvers.blocks builds equivalence blocks with COLORLESS_RANK; the Cython solver
    values the moves with _str_b. tests/ulti/test_block_equivalence.py asserts every card
    in a block solves to the identical value — which can only hold if the two orderings
    agree. This test documents that link and checks the pieces are still wired together.
    """
    from ulti.solvers import blocks

    assert blocks.COLORLESS_RANK is COLORLESS_RANK
    hand = sorted(COLORLESS_RANK, key=COLORLESS_RANK.get)
    assert hand == ["7", "8", "9", "10", "lower", "upper", "king", "ace"]


def test_kontra_units_are_not_redefined_in_the_api_layer():
    """apps.api.play must IMPORT the unit rules, not restate them."""
    play = (_ROOT / "apps" / "api" / "play.py").read_text()
    assert "from ulti.scoring.units import" in play
    for banned in ("def kontra_units", "def _kontra_units",
                   '_UNIT_OBJ = {', '_UNITS_ORDER = ('):
        assert banned not in play, (
            f"apps/api/play.py redefines {banned!r} — it belongs in ulti.scoring.units")


def test_oracle_and_kontra_offer_share_the_parti_rule():
    """Both paths must call game_has_parti rather than each spelling the rule out."""
    from ulti.scoring import oracle, units

    assert oracle.game_has_parti is units.game_has_parti
    assert oracle._unit_of is units.unit_of
    # The párti component itself must come from the shared predicate. (oracle.py still
    # uses bid_a_100 for the SILENT-rider gate — a different rule, legitimately its own.)
    src = (_ROOT / "ulti" / "scoring" / "oracle.py").read_text()
    parti_line = [l for l in src.splitlines() if 'out.components["parti"]' in l]
    assert parti_line, "could not find the párti component assignment"
    idx = src.splitlines().index(parti_line[0])
    guard = "\n".join(src.splitlines()[max(0, idx - 5):idx])
    assert "game_has_parti" in guard, (
        f"the párti component is not gated by the shared predicate:\n{guard}")


def test_the_two_card_models_agree_on_everything():
    """``ulti.card`` and ``ultisolver.games.ulti.cards`` are two independent card models,
    bridged by ulti.solvers.pis. The split is deliberate (the solver package stands alone),
    so the models cannot be merged — but every rule they BOTH encode must match, or the
    app and the solver would disagree about the same card.

    Checks all 32 cards: the translation round-trips, and points / natural strength /
    colourless strength are identical on both sides.
    """
    from ulti.card import DECK, RANK_POINTS
    from ulti.solvers.pis import _to_o, _to_t

    for card in DECK:
        t = _to_t(card)
        assert _to_o(t).id == card.id, f"{card} does not survive the round trip"
        assert t.points() == RANK_POINTS[card.rank], f"{card}: points disagree"
        assert t.strength() == card.rank_index, f"{card}: natural strength disagrees"
        assert t.strength(betli=True) == COLORLESS_RANK[card.rank], (
            f"{card}: colourless strength disagrees")


def test_translation_covers_the_whole_deck():
    """Both decks are 32 distinct cards and map onto each other exactly."""
    from ulti.card import DECK
    from ulti.solvers.pis import _to_t

    assert len({c.id for c in DECK}) == 32
    assert len({(_to_t(c).suit, _to_t(c).rank) for c in DECK}) == 32
