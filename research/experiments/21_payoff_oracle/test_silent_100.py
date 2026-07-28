"""Unit test: corrected silent-100 oracle logic against milan's examples.

card points = trick points only (marriages excluded); a silent 100 = ONE
marriage + card points >= 100; only the better one scores; naked marriage = 0.
  40-100: card_pts >= 60, value 2 (piros 4)
  20-100: card_pts >= 80, value 4 (piros 8)
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')

from trickster.games.ulti.game import GameState, Suit
from scoring.oracle import BidSet, score

ACORNS, LEAVES, BELLS, HEARTS = Suit.ACORNS, Suit.LEAVES, Suit.BELLS, Suit.HEARTS


def mk(scores, marriages, trump=ACORNS, sol_tricks=5):
    # sol_tricks must be non-degenerate: soloist_tricks==0 is a DEFENDER sweep
    # (def durchmars replaces parti), 10 is a soloist sweep. These 100-threshold
    # cases are about the soloist's silent-100, so keep the soloist mid-table.
    return GameState(
        hands=[[], [], []], trump=trump, betli=False, soloist=0, dealer=2,
        captured=[[0] * (sol_tricks * 3), [], []], scores=list(scores),
        leader=0, trick_no=10, trick_cards=[], last_trick=None,
        marriages=list(marriages), marriages_declared=True,
    )


def comp(state, **bidkw):
    bid = BidSet(parti=True, **bidkw)
    return score(final_pos=state, bid=bid).components


def check(name, got, key, expect):
    v = got.get(key, 0)
    ok = (v == expect)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {key}={v} (expect {expect})")
    return ok


def main():
    allok = True
    # 1. 40 only, card_pts=60 → silent_40_100 = +2
    s = mk([100, 0, 0], [(0, ACORNS, 40)])             # 60 card + 40 marriage
    allok &= check("40 only, card 60", comp(s), "silent_40_100", 2)
    # 2. 40 only, card_pts=59 → none
    s = mk([99, 0, 0], [(0, ACORNS, 40)])
    allok &= check("40 only, card 59", comp(s), "silent_40_100", 0)
    # 3. 20 only, card_pts=80 → silent_20_100 = +4
    s = mk([100, 0, 0], [(0, LEAVES, 20)])             # 80 card + 20 marriage
    allok &= check("20 only, card 80", comp(s), "silent_20_100", 4)
    # 4. 20 only, card_pts=79 → none
    s = mk([99, 0, 0], [(0, LEAVES, 20)])
    allok &= check("20 only, card 79", comp(s), "silent_20_100", 0)
    # 5. 40 & 20, card_pts=60 → only 40-100 (20 needs 80)
    s = mk([120, 0, 0], [(0, ACORNS, 40), (0, LEAVES, 20)])  # 60 card + 60 marr
    g = comp(s)
    allok &= check("40&20, card 60", g, "silent_40_100", 2)
    allok &= check("40&20, card 60 (no 20-100)", g, "silent_20_100", 0)
    # 6. 40 & 20, card_pts=80 → pick better → 20-100 = 4
    s = mk([140, 0, 0], [(0, ACORNS, 40), (0, LEAVES, 20)])  # 80 card + 60 marr
    g = comp(s)
    allok &= check("40&20, card 80 (pick 20-100)", g, "silent_20_100", 4)
    allok &= check("40&20, card 80 (no 40-100)", g, "silent_40_100", 0)
    # 7. milan's stack: 40+20+20+20 = 100 marriages, card_pts=0 → NO silent 100
    s = mk([100, 0, 0], [(0, ACORNS, 40), (0, LEAVES, 20),
                         (0, BELLS, 20), (0, HEARTS, 20)])
    g = comp(s)
    allok &= check("40+20+20+20, card 0 (no 40-100)", g, "silent_40_100", 0)
    allok &= check("40+20+20+20, card 0 (no 20-100)", g, "silent_20_100", 0)
    allok &= check("40+20+20+20, card 0 (parti won)", g, "parti", 1)
    # 8. piros: trump=HEARTS, 40, card_pts=60, bid.piros → silent_40_100 ×2 = 4
    s = mk([100, 0, 0], [(0, HEARTS, 40)], trump=HEARTS)
    allok &= check("piros 40, card 60", comp(s, piros=True), "silent_40_100", 4)
    # 9. piros: 20, card_pts=80 → silent_20_100 ×2 = 8
    s = mk([100, 0, 0], [(0, LEAVES, 20)], trump=HEARTS)
    allok &= check("piros 20, card 80", comp(s, piros=True), "silent_20_100", 8)

    print()
    print("ALL PASS" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
