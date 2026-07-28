"""Binary contracts — betli (5) / piros betli aka REBETLI (10), and durchmars
colored (6) / piros (12). No parti, no silent riders (not points-based)."""
import sys
from types import SimpleNamespace
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
from trickster.games.ulti.cards import Card, Rank, Suit
from scoring.oracle import BidSet, score

FILL = Card(Suit.BELLS, Rank.SEVEN)

def mk(sol_tricks, *, trump=None, betli=False):
    return SimpleNamespace(
        hands=[[], [], []], trump=trump, betli=betli, soloist=0, dealer=2,
        captured=[[FILL] * (sol_tricks * 3), [], []], scores=[0, 0, 0], leader=0,
        trick_no=10, trick_cards=[], last_trick=None, marriages=[],
        marriages_declared=True, talon_discards=[])

CASES = [
    # betli: take 0 tricks. piros betli = REBETLI (the double rung)
    ("betli won (0 tricks)",     mk(0, betli=True), BidSet(betli=True), +5),
    ("betli lost (1 trick)",     mk(1, betli=True), BidSet(betli=True), -5),
    ("rebetli won (piros betli)",mk(0, betli=True), BidSet(betli=True, piros=True), +10),
    ("rebetli lost",             mk(1, betli=True), BidSet(betli=True, piros=True), -10),
    # durchmars colored (6) / piros (12)
    ("duri made (sweep)",        mk(10, trump=Suit.ACORNS), BidSet(durchmars=True), +6),
    ("duri failed (9 tricks)",   mk(9, trump=Suit.ACORNS), BidSet(durchmars=True), -6),
    ("piros duri made",          mk(10, trump=Suit.HEARTS), BidSet(durchmars=True, piros=True), +12),
    ("piros duri failed",        mk(9, trump=Suit.HEARTS), BidSet(durchmars=True, piros=True), -12),
]

ok = True
for name, state, bid, want in CASES:
    got = score(final_pos=state, bid=bid).total_per_def
    f = (got == want); ok &= f
    print(f"  [{'PASS' if f else 'FAIL'}] {name:<26} got {got:+d} (want {want:+d})")
print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
