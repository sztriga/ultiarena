"""Tests for AI-only play restrictions."""

from trickster.games.ulti.cards import Card, Rank, Suit
from trickster.games.ulti.game import GameState, deal, pickup_talon, set_contract, discard_talon
from trickster.games.ulti.restrictions import (
    apply_restrictions,
    betli_play_highest,
)


def _make_betli_gs(soloist: int = 0) -> GameState:
    """Create a minimal betli GameState with soloist's hand set up."""
    gs, talon = deal(seed=42, dealer=2)  # soloist = next_player(2) = 0
    gs.soloist = soloist
    pickup_talon(gs, soloist, talon)
    set_contract(gs, soloist, trump=None, betli=True)
    import random
    rng = random.Random(42)
    discards = rng.sample(gs.hands[soloist], 2)
    discard_talon(gs, discards)
    return gs


class TestBetliPlayHighest:
    def test_keeps_highest_per_suit(self):
        """Should keep only the highest-strength card per suit."""
        gs = _make_betli_gs(soloist=0)
        # Construct a hand with known cards
        cards = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.BELLS, Rank.ACE),
            Card(Suit.BELLS, Rank.NINE),
        ]
        result = betli_play_highest(gs, 0, cards)
        # Should keep King (strength 6) from hearts, Ace (strength 7) from bells
        assert Card(Suit.HEARTS, Rank.KING) in result
        assert Card(Suit.BELLS, Rank.ACE) in result
        assert len(result) == 2

    def test_noop_for_defenders(self):
        """Defenders should not be restricted."""
        gs = _make_betli_gs(soloist=0)
        cards = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.KING),
        ]
        result = betli_play_highest(gs, 1, cards)
        assert result == cards

    def test_noop_for_non_betli(self):
        """Non-betli games should not be restricted."""
        gs, talon = deal(seed=42, dealer=2)
        gs.soloist = 0
        pickup_talon(gs, 0, talon)
        set_contract(gs, 0, trump=Suit.HEARTS)
        import random
        discards = random.Random(42).sample(gs.hands[0], 2)
        discard_talon(gs, discards)

        cards = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.KING),
        ]
        result = betli_play_highest(gs, 0, cards)
        assert result == cards

    def test_ten_below_jack_in_betli(self):
        """In betli ordering, Ten < Jack, so Jack should be kept over Ten."""
        gs = _make_betli_gs(soloist=0)
        cards = [
            Card(Suit.LEAVES, Rank.TEN),
            Card(Suit.LEAVES, Rank.JACK),
        ]
        result = betli_play_highest(gs, 0, cards)
        assert result == [Card(Suit.LEAVES, Rank.JACK)]


class TestApplyRestrictions:
    def test_fallback_on_empty(self):
        """If a restriction would filter to empty, keep previous set."""
        gs = _make_betli_gs(soloist=0)
        cards = [Card(Suit.HEARTS, Rank.ACE)]

        def kill_all(gs, player, cards):
            return []  # would leave empty

        result = apply_restrictions([kill_all], gs, 0, cards)
        assert result == cards

    def test_chains_restrictions(self):
        """Multiple restrictions should apply sequentially."""
        gs = _make_betli_gs(soloist=0)
        cards = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.KING),
            Card(Suit.BELLS, Rank.ACE),
        ]
        # First restriction: betli_play_highest → [King, Ace]
        # Second restriction: keep only hearts
        def only_hearts(gs, player, cards):
            return [c for c in cards if c.suit == Suit.HEARTS]

        result = apply_restrictions(
            [betli_play_highest, only_hearts], gs, 0, cards,
        )
        assert result == [Card(Suit.HEARTS, Rank.KING)]
