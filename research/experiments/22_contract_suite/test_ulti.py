"""Bid ulti scoring (milan 2026-06-16): a bid ulti is ALWAYS ulti + parti (same
color). ulti = 4 (piros 8) made / -8 (piros -16) bukott; the accompanying parti
= ±1 (piros ±2) and is scored alongside. Silent duri / 40-100 / 20-100 REPLACE
the parti part if silently done; silent ulti doesn't apply (you bid it)."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from trickster.games.ulti.cards import Card, Rank, Suit
from ulti.scoring.oracle import BidSet, score

H, A, L, B = Suit.HEARTS, Suit.ACORNS, Suit.LEAVES, Suit.BELLS
FILL = Card(B, Rank.SEVEN)


def mk(scores, marriages, *, sol_tricks=5, trump=A, talon=(), last_trick=None):
    return SimpleNamespace(
        hands=[[], [], []], trump=trump, betli=False, soloist=0, dealer=2,
        captured=[[FILL] * (sol_tricks * 3), [], []], scores=list(scores),
        leader=0, trick_no=10, trick_cards=[], last_trick=last_trick,
        marriages=list(marriages), marriages_declared=True, talon_discards=list(talon),
        has_ulti=True,
    )


def ltrick(player_of_7, winner, trump=A):
    """Last trick where `player_of_7` played the trump-7; `winner` won it."""
    cards = [Card(trump, Rank.SEVEN), Card(B, Rank.EIGHT), Card(B, Rank.NINE)]
    players = [player_of_7, (player_of_7 + 1) % 3, (player_of_7 + 2) % 3]
    return SimpleNamespace(cards=cards, players=players, winner=winner)


U  = BidSet(ulti=True, piros=False)   # colored bid ulti
Up = BidSet(ulti=True, piros=True)    # piros bid ulti

CASES = [
    # name, state, bid, expected total GP/def
    ("ulti made + parti won",        mk([50, 30, 0], [], last_trick=ltrick(0, 0)), U, +5),   # +4 +1
    ("ulti made + parti lost",       mk([30, 50, 0], [], last_trick=ltrick(0, 0)), U, +3),   # +4 -1
    ("ulti bukott + parti won",      mk([50, 30, 0], [], last_trick=ltrick(0, 1)), U, -7),   # -8 +1
    ("ulti bukott + parti lost",     mk([30, 50, 0], [], last_trick=ltrick(0, 1)), U, -9),   # -8 -1
    ("piros ulti made + parti won",  mk([50, 30, 0], [], last_trick=ltrick(0, 0, H), trump=H), Up, +10),  # +8 +2
    ("piros ulti bukott + parti won",mk([50, 30, 0], [], last_trick=ltrick(0, 1, H), trump=H), Up, -14),  # -16 +2
    ("ulti made + silent 40-100",
        mk([100, 0, 0], [(0, A, 40)], last_trick=ltrick(0, 0)), U, +6),     # +4 ulti, +2 (40-100 replaces parti)
    ("ulti made + silent duri (sweep)",
        mk([90, 0, 0], [], sol_tricks=10, last_trick=ltrick(0, 0)), U, +7),  # +4 ulti, +3 (duri replaces parti)
]


def main():
    ok = True
    for name, state, bid, want in CASES:
        pv = score(final_pos=state, bid=bid)
        got = pv.total_per_def
        flag = (got == want)
        ok &= flag
        print(f"  [{'PASS' if flag else 'FAIL'}] {name:<34} got {got:+d} (want {want:+d})"
              + ("" if flag else f"   comps={pv.components}"))
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
