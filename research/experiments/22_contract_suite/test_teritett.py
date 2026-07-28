"""Terített (open-cards) contracts (milan 2026-06-16): the soloist plays face-up;
the DURI/betli component DOUBLES — ×2 colored, ×4 colorless (terített betli 20,
terített colorless duri 24). piros's ×2 (all components) stacks on top."""
import sys
from types import SimpleNamespace
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
from trickster.games.ulti.cards import Card, Rank, Suit
from scoring.oracle import BidSet, score
A,B,L,H = Suit.ACORNS, Suit.BELLS, Suit.LEAVES, Suit.HEARTS; F=Card(B,Rank.SEVEN)
def mk(scores, marr, sol_tricks=10, lt=None, trump=A, betli=False):
    return SimpleNamespace(hands=[[],[],[]], trump=trump, betli=betli, soloist=0, dealer=2,
        captured=[[F]*(sol_tricks*3),[],[]], scores=scores, leader=0, trick_no=10,
        trick_cards=[], last_trick=lt, marriages=marr, marriages_declared=True, talon_discards=[], has_ulti=True)
def lt7(p7,w,t=A): return SimpleNamespace(cards=[Card(t,Rank.SEVEN),Card(B,Rank.EIGHT),Card(B,Rank.NINE)],players=[p7,(p7+1)%3,(p7+2)%3],winner=w)
no7 = SimpleNamespace(cards=[Card(A,Rank.ACE),Card(B,Rank.EIGHT),Card(B,Rank.NINE)],players=[0,1,2],winner=0)
T=lambda **k: BidSet(teritett=True, **k)
CASES = [
  ('terített betli won (×4)',         mk([0,0,0],[],0,trump=None,betli=True), T(betli=True), +20),
  ('terített betli lost',             mk([0,0,0],[],1,trump=None,betli=True), T(betli=True), -20),
  ('terített colorless duri (×4)',    mk([90,0,0],[],10,no7,trump=None), T(durchmars=True), +24),
  ('terített ulti-duri (combo ×2)',   mk([90,0,0],[],10,lt7(0,0)), T(ulti=True,durchmars=True), +16),
  ('terített 20-100-duri (clean)',    mk([110,0,0],[(0,L,20)],10,no7), T(durchmars=True,twenty_hundred=True), +20),
  ('terített 20-100-duri + silent ulti', mk([110,0,0],[(0,L,20)],10,lt7(0,0)), T(durchmars=True,twenty_hundred=True), +22),
  ('piros terített duri (×2·×2)',     mk([90,0,0],[],10,no7,trump=H), T(durchmars=True,piros=True), +24),
  ('piros terített ulti-duri',        mk([90,0,0],[],10,lt7(0,0,H),trump=H), T(ulti=True,durchmars=True,piros=True), +32),
]
ok=True
for nm,st,bid,want in CASES:
    pv=score(final_pos=st,bid=bid); f=(pv.total_per_def==want); ok&=f
    print(f"  [{'PASS' if f else 'FAIL'}] {nm:<36} {pv.total_per_def:+d} (want {want:+d})" + ("" if f else f"  {pv.components}"))
print("\n"+("ALL PASS" if ok else "SOME FAILED")); sys.exit(0 if ok else 1)
