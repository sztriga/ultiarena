"""Kontra scoring (milan 2026-06-26), simple contracts.

kontra ×2 / rekontra ×4 on the stake. Ulti bukott "duplán fizet" interacts
specially: made ulti = 2^level × base; bukott ulti = (2^level + 1) × base →
2x / 3x / 5x of base for none/kontra/rekontra. The accompanying parti rider (and
all symmetric contracts) just scale ×2^level. piros ×2 stacks.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
from trickster.games.ulti.cards import Card, Rank, Suit
from scoring.oracle import BidSet, score

H, A, B = Suit.HEARTS, Suit.ACORNS, Suit.BELLS
FILL = Card(B, Rank.SEVEN)


def mk(scores, *, sol_tricks=5, trump=A, last_trick=None):
    return SimpleNamespace(
        hands=[[], [], []], trump=trump, betli=False, soloist=0, dealer=2,
        captured=[[FILL] * (sol_tricks * 3), [], []], scores=list(scores),
        leader=0, trick_no=10, trick_cards=[], last_trick=last_trick,
        marriages=[], marriages_declared=True, talon_discards=[], has_ulti=True,
    )


def ltrick(player_of_7, winner, trump=A):
    cards = [Card(trump, Rank.SEVEN), Card(B, Rank.EIGHT), Card(B, Rank.NINE)]
    players = [player_of_7, (player_of_7 + 1) % 3, (player_of_7 + 2) % 3]
    return SimpleNamespace(cards=cards, players=players, winner=winner)


def U(level, piros=False):
    return BidSet(ulti=True, piros=piros, kontra_level=level)


# state, bid, expected total GP/def
CASES = [
    # ulti MADE + parti won: base +5 → ×2^level on both parts
    ("ulti made+parti, none",   mk([50, 30, 0], last_trick=ltrick(0, 0)), U(0), +5),
    ("ulti made+parti, kontra", mk([50, 30, 0], last_trick=ltrick(0, 0)), U(1), +10),
    ("ulti made+parti, rekontra", mk([50, 30, 0], last_trick=ltrick(0, 0)), U(2), +20),
    # ulti BUKOTT + parti won: ulti = -(2^level+1)*4, parti = +1*2^level
    ("ulti buk+parti won, none",   mk([50, 30, 0], last_trick=ltrick(0, 1)), U(0), -7),
    ("ulti buk+parti won, kontra", mk([50, 30, 0], last_trick=ltrick(0, 1)), U(1), -10),
    ("ulti buk+parti won, rekontra", mk([50, 30, 0], last_trick=ltrick(0, 1)), U(2), -16),
    # ulti BUKOTT + parti lost
    ("ulti buk+parti lost, none",   mk([30, 50, 0], last_trick=ltrick(0, 1)), U(0), -9),
    ("ulti buk+parti lost, kontra", mk([30, 50, 0], last_trick=ltrick(0, 1)), U(1), -14),
    ("ulti buk+parti lost, rekontra", mk([30, 50, 0], last_trick=ltrick(0, 1)), U(2), -24),
    # piros ulti (×2 stacks)
    ("piros ulti made+parti, kontra",
        mk([50, 30, 0], last_trick=ltrick(0, 0, H), trump=H), U(1, piros=True), +20),
    ("piros ulti buk+parti won, none",
        mk([50, 30, 0], last_trick=ltrick(0, 1, H), trump=H), U(0, piros=True), -14),
    ("piros ulti buk+parti won, kontra",
        mk([50, 30, 0], last_trick=ltrick(0, 1, H), trump=H), U(1, piros=True), -20),
    ("piros ulti buk+parti won, rekontra",
        mk([50, 30, 0], last_trick=ltrick(0, 1, H), trump=H), U(2, piros=True), -32),
]


def main():
    fails = 0
    for name, state, bid, want in CASES:
        got = score(final_pos=state, bid=bid).total_per_def
        ok = (got == want)
        fails += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} got {got:+d} want {want:+d}")
    print("\nALL PASS" if not fails else f"\n{fails} FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
