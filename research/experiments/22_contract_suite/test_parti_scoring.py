"""Parti-family scoring spec (milan 2026-06-15). Checks total GP/def against
milan's numbers across: plain & piros parti; silent 100/duri (REPLACE parti,
stack with each other); silent ulti (STACKS, bukott doubles); defender side
(sweep / 100 = negative); talon counts for defenders; BID 40-100/20-100 in both
colors (leave parti out)."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from trickster.games.ulti.cards import Card, Rank, Suit
from ulti.scoring.oracle import BidSet, score

H, L, A, B = Suit.HEARTS, Suit.LEAVES, Suit.ACORNS, Suit.BELLS
FILL = Card(B, Rank.SEVEN)


def mk(scores, marriages, *, sol_tricks=5, trump=H, talon=(), last_trick=None):
    return SimpleNamespace(
        hands=[[], [], []], trump=trump, betli=False, soloist=0, dealer=2,
        captured=[[FILL] * (sol_tricks * 3), [], []], scores=list(scores),
        leader=0, trick_no=10, trick_cards=[], last_trick=last_trick,
        marriages=list(marriages), marriages_declared=True,
        talon_discards=list(talon),
    )


def ltrick(player_of_7, winner, trump=H):
    """Last trick where `player_of_7` played the trump-7; `winner` won it."""
    cards = [Card(trump, Rank.SEVEN), Card(B, Rank.EIGHT), Card(B, Rank.NINE)]
    players = [player_of_7, (player_of_7 + 1) % 3, (player_of_7 + 2) % 3]
    return SimpleNamespace(cards=cards, players=players, winner=winner)


def gp(state, bid):
    return score(final_pos=state, bid=bid).total_per_def


P  = BidSet(parti=True, piros=True)            # piros parti
Pp = BidSet(parti=True, piros=False)           # plain/colored parti
B40 = BidSet(forty_hundred=True, piros=True)   # bid piros 40-100  (8)
B20 = BidSet(twenty_hundred=True, piros=True)  # bid piros 20-100  (16)
B40c = BidSet(forty_hundred=True, piros=False) # bid colored 40-100 (4)
B20c = BidSet(twenty_hundred=True, piros=False)# bid colored 20-100 (8)

CASES = [
    # name, state, bid, expected total GP/def
    # ── piros parti: silent 100/duri REPLACE parti, silent ulti STACKS ──
    ("piros parti win",            mk([50, 30, 0], []), P, +2),
    ("piros parti loss",           mk([30, 50, 0], []), P, -2),
    ("parti win + silent ulti",    mk([50, 30, 0], [], last_trick=ltrick(0, 0)), P, +6),
    ("parti win + ulti bukott",    mk([50, 30, 0], [], last_trick=ltrick(0, 1)), P, -6),
    ("sol silent 40-100",          mk([100, 0, 0], [(0, H, 40)], sol_tricks=9), P, +4),
    ("sol silent 20-100",          mk([100, 0, 0], [(0, L, 20)], sol_tricks=9), P, +8),
    ("sol silent duri",            mk([90, 0, 0], [], sol_tricks=10), P, +6),
    ("sol duri + 40-100",          mk([130, 0, 0], [(0, H, 40)], sol_tricks=10), P, +10),
    ("def sweep, no marriage",     mk([0, 45, 45], [], sol_tricks=0), P, -6),
    ("def sweep + 40",             mk([0, 85, 45], [(1, H, 40)], sol_tricks=0), P, -10),
    ("def sweep + 20",             mk([0, 65, 45], [(1, L, 20)], sol_tricks=0), P, -14),
    ("def sweep + 40 & 20",        mk([0, 85, 65], [(1, H, 40), (2, L, 20)], sol_tricks=0), P, -14),
    ("talon counts for def (loss)",
        mk([50, 40, 0], [], trump=A, talon=[Card(H, Rank.ACE), Card(L, Rank.ACE)]), Pp, -1),
    # ── bid 40-100 / 20-100: NO parti, only the 100 (rest of silents hold) ──
    ("bid 40-100 made (no parti)",      mk([100, 0, 0], [(0, H, 40)], sol_tricks=9), B40, +8),
    ("bid 40-100 bukott",               mk([99, 0, 0], [(0, H, 40)], sol_tricks=9), B40, -8),
    ("bid 20-100 made (no parti)",      mk([100, 0, 0], [(0, L, 20)], sol_tricks=9), B20, +16),
    ("bid 20-100 bukott",               mk([99, 0, 0], [(0, L, 20)], sol_tricks=9), B20, -16),
    ("bid 40-100 + silent ulti stacks",
        mk([100, 0, 0], [(0, H, 40)], sol_tricks=9, last_trick=ltrick(0, 0)), B40, +12),
    ("bid 40-100 + sweep (duri holds)", mk([130, 0, 0], [(0, H, 40)], sol_tricks=10), B40, +14),
    # ── COLORED (non-piros) 40-100 / 20-100 — the normal way they're played ──
    ("sol silent 40-100 colored",  mk([100, 0, 0], [(0, A, 40)], sol_tricks=9, trump=A), Pp, +2),
    ("sol silent 20-100 colored",  mk([100, 0, 0], [(0, L, 20)], sol_tricks=9, trump=A), Pp, +4),
    ("bid 40-100 colored made",    mk([100, 0, 0], [(0, A, 40)], sol_tricks=9, trump=A), B40c, +4),
    ("bid 40-100 colored bukott",  mk([99, 0, 0], [(0, A, 40)], sol_tricks=9, trump=A), B40c, -4),
    ("bid 20-100 colored made",    mk([100, 0, 0], [(0, L, 20)], sol_tricks=9, trump=A), B20c, +8),
    ("bid 20-100 colored bukott",  mk([99, 0, 0], [(0, L, 20)], sol_tricks=9, trump=A), B20c, -8),
]


def main():
    ok = True
    for name, state, bid, want in CASES:
        got = gp(state, bid)
        flag = (got == want)
        ok &= flag
        comps = score(final_pos=state, bid=bid).components
        print(f"  [{'PASS' if flag else 'FAIL'}] {name:<32} got {got:+d} (want {want:+d})"
              + ("" if flag else f"   comps={comps}"))
    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
