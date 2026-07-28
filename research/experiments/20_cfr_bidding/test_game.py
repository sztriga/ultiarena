"""Sanity tests for the bidding-tree logic (game.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

import game as G
from common import ACTIONS, ACTION_INDEX, PASS

# ev[player][action_idx]; avail[player][action_idx]
ev = np.array([
    [-2.0,  4.0, -5.0, -6.0,  np.nan],  # P0: parti -2, ulti +4, betli -5, duri -6, no piros-ulti
    [ 2.0,  4.0, -5.0, -6.0, -16.0],    # P1
    [-2.0, -8.0, +5.0, -6.0,  np.nan],  # P2: betli winnable!
], dtype=np.float32)
avail = np.array([
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
], dtype=bool)
ctx = {'ev': ev, 'avail': avail, 'bucket': [0, 0, 0]}

def ok(cond, msg):
    print(("  ok  " if cond else "FAIL  ") + msg)
    assert cond, msg

print("=== open node ===")
h = ()
ok(G.to_move(h) == 0, "P0 to move at open")
la = G.legal_actions(h, ctx)
ok(la == [PASS, 'parti', 'ulti', 'betli', 'duri'], f"P0 open actions (no piros-ulti): {la}")
ok(not G.is_terminal(h), "open not terminal")

print("\n=== pass-out ===")
h2 = G.apply(h, PASS)
ok(G.is_terminal(h2), "P0 pass-out terminal")
ok(G.payoffs(h2, ctx) == (-4.0, 2.0, 2.0), f"pass-out payoff {G.payoffs(h2, ctx)}")

print("\n=== open parti, both pass → P0 plays parti ===")
h = G.apply((), 'parti')
ok(G.to_move(h) == 1 and not G.is_terminal(h), "after open, P1 to move")
la = G.legal_actions(h, ctx)
ok(la == [PASS, 'ulti', 'betli', 'duri', 'ulti_piros'], f"P1 raises above rank2: {la}")
h = G.apply(h, PASS)          # P1 pass
h = G.apply(h, PASS)          # P2 pass
ok(G.to_move(h) == 0, "back to holder P0")
ok(G.legal_actions(h, ctx) == [PASS], "holder auto-pass only")
h = G.apply(h, PASS)          # P0 auto-pass → 3 passes
ok(G.is_terminal(h), "3 passes → terminal")
pay = G.payoffs(h, ctx)
ok(pay == (-4.0, 2.0, 2.0), f"P0 parti ev-2 → soloist 2*-2=-4, defs +2: {pay}")

print("\n=== P0 opens ulti(+4), nobody overtakes → P0 wins ulti ===")
h = G.apply((), 'ulti')
h = G.apply(h, PASS); h = G.apply(h, PASS); h = G.apply(h, PASS)
ok(G.is_terminal(h), "terminal")
ok(G.payoffs(h, ctx) == (8.0, -4.0, -4.0), f"ulti +4 → (8,-4,-4): {G.payoffs(h, ctx)}")

print("\n=== overtake: P0 parti, P2 takes betli (P2 can win betli) ===")
h = G.apply((), 'parti')      # holder 0, level 2
h = G.apply(h, PASS)          # P1 pass
ok(G.to_move(h) == 2, "P2 to move")
la = G.legal_actions(h, ctx)
ok('betli' in la and 'ulti' not in la, f"P2 betli legal, ulti not avail: {la}")
h = G.apply(h, 'betli')       # P2 overtakes → holder 2, level 4
ok(not G.is_terminal(h), "after overtake not terminal (passes reset)")
ok(G.to_move(h) == 0, "P0 to move after overtake")
h = G.apply(h, PASS); h = G.apply(h, PASS); h = G.apply(h, PASS)
ok(G.is_terminal(h), "terminal after 3 passes")
pay = G.payoffs(h, ctx)
ok(pay == (-5.0, -5.0, 10.0), f"P2 betli +5 → (−5,−5,10): {pay}")

print("\n=== infoset key uses bucket + history ===")
ctx2 = {'ev': ev, 'avail': avail, 'bucket': [3, 7, 5]}
h = G.apply((), 'parti')
ok(G.infoset_key(h, ctx2) == (7, ((0, 'parti'),)), f"P1 infoset {G.infoset_key(h, ctx2)}")

print("\nALL GAME TESTS PASSED")
