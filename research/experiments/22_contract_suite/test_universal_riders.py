"""Universal silent riders (milan 2026-06-16): silent ulti + silent 100 attach to
ANY game by achievement-minus-bid, incl. durchmars combos. silent durchmars stays
points-based (betli's 0-tricks / a bid duri aren't silent duris)."""
import sys
from types import SimpleNamespace
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
from trickster.games.ulti.cards import Card, Rank, Suit
from ulti.scoring.oracle import BidSet, score
A,B,L = Suit.ACORNS, Suit.BELLS, Suit.LEAVES; F=Card(B,Rank.SEVEN)
def mk(scores, marr, sol_tricks=10, lt=None, trump=A):
    return SimpleNamespace(hands=[[],[],[]], trump=trump, betli=False, soloist=0, dealer=2,
        captured=[[F]*(sol_tricks*3),[],[]], scores=scores, leader=0, trick_no=10,
        trick_cards=[], last_trick=lt, marriages=marr, marriages_declared=True, talon_discards=[], has_ulti=True)
def lt(p7,w,trump=A): return SimpleNamespace(cards=[Card(trump,Rank.SEVEN),Card(B,Rank.EIGHT),Card(B,Rank.NINE)],players=[p7,(p7+1)%3,(p7+2)%3],winner=w)
CASES = [
  # ulti-duri now CARRIES the silent 100 (was +10, the gap)
  ("ulti-duri sweep + 40 (silent 40-100)", mk([130,0,0],[(0,A,40)],10,lt(0,0)), BidSet(ulti=True,durchmars=True), +12),
  ("ulti-duri sweep + 20 (silent 20-100)", mk([110,0,0],[(0,L,20)],10,lt(0,0)), BidSet(ulti=True,durchmars=True), +14),
  # 40-100-duri: bid the 40-100, sweep, take the 7 → silent ulti rides (no 2nd 100)
  ("40-100-duri sweep + win 7 (silent ulti)", mk([130,0,0],[(0,A,40)],10,lt(0,0)), BidSet(durchmars=True,forty_hundred=True), +12),
  # pure colored durchmars (hypothetical) carries silent ulti + silent 100
  ("colored duri sweep + 40 + win 7", mk([130,0,0],[(0,A,40)],10,lt(0,0)), BidSet(durchmars=True), +10),
  # colorless durchmars: no trump, no marriages → clean, riders can't trigger
  ("colorless duri sweep (no riders)", SimpleNamespace(hands=[[],[],[]],trump=None,betli=False,soloist=0,dealer=2,captured=[[F]*30,[],[]],scores=[90,0,0],leader=0,trick_no=10,trick_cards=[],last_trick=None,marriages=[],marriages_declared=True,talon_discards=[],has_ulti=False), BidSet(durchmars=True), +6),
  # betli: soloist 0 tricks must NOT register a defender duri
  ("betli won (no spurious def-duri)", SimpleNamespace(hands=[[],[],[]],trump=None,betli=True,soloist=0,dealer=2,captured=[[],[],[]],scores=[0,0,0],leader=0,trick_no=10,trick_cards=[],last_trick=None,marriages=[],marriages_declared=True,talon_discards=[],has_ulti=False), BidSet(betli=True), +5),
  # TRIPLES: ulti + a 100 + duri all bid → NO silent riders left (everything bid)
  ("ulti-40-100-duri all made (14)", mk([130,0,0],[(0,A,40)],10,lt(0,0)), BidSet(ulti=True,forty_hundred=True,durchmars=True), +14),
  ("ulti-20-100-duri all made (18)", mk([110,0,0],[(0,L,20)],10,lt(0,0)), BidSet(ulti=True,twenty_hundred=True,durchmars=True), +18),
  ("ulti-40-100-duri all bukott", mk([99,0,0],[(0,A,40)],9,lt(0,1)), BidSet(ulti=True,forty_hundred=True,durchmars=True), -18),
]
ok=True
for nm,st,bid,want in CASES:
    pv=score(final_pos=st,bid=bid); f=(pv.total_per_def==want); ok&=f
    print(f"  [{'PASS' if f else 'FAIL'}] {nm:<40} {pv.total_per_def:+d} (want {want:+d})" + ("" if f else f"  {pv.components}"))
print("\n"+("ALL PASS" if ok else "SOME FAILED")); sys.exit(0 if ok else 1)
