"""Combined-game kontra (milan 2026-07-05):
  • per-component: "kontra ulti" doubles only ulti, "kontra parti" only parti.
  • silent substitutes ride the parti unit: kontra parti scales silent 40/20-100
    AND silent duri (both if both land).
  • colorless (betli / standalone duri) = SEPARATE per-defender counters → the
    two defenders can differ (asymmetric gp_vs); colored stays shared (együtt
    sírunk → symmetric, no def_split).
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, '/Users/milansimity/Cuccok/kodok/oldtawer')
from trickster.games.ulti.cards import Card, Rank, Suit
from scoring.oracle import BidSet, score

H, A, B = Suit.HEARTS, Suit.ACORNS, Suit.BELLS
FILL = Card(B, Rank.SEVEN)


def mk_u(scores, marriages, *, sol_tricks=5, trump=A, last_trick=None):
    return SimpleNamespace(
        hands=[[], [], []], trump=trump, betli=False, soloist=0, dealer=2,
        captured=[[FILL] * (sol_tricks * 3), [], []], scores=list(scores),
        leader=0, trick_no=10, trick_cards=[], last_trick=last_trick,
        marriages=list(marriages), marriages_declared=True, talon_discards=[],
        has_ulti=True)


def mk_b(sol_tricks, *, trump=None, betli=False):
    return SimpleNamespace(
        hands=[[], [], []], trump=trump, betli=betli, soloist=0, dealer=2,
        captured=[[FILL] * (sol_tricks * 3), [], []], scores=[0, 0, 0], leader=0,
        trick_no=10, trick_cards=[], last_trick=None, marriages=[],
        marriages_declared=True, talon_discards=[])


def ltrick(player_of_7, winner, trump=A):
    cards = [Card(trump, Rank.SEVEN), Card(B, Rank.EIGHT), Card(B, Rank.NINE)]
    players = [player_of_7, (player_of_7 + 1) % 3, (player_of_7 + 2) % 3]
    return SimpleNamespace(cards=cards, players=players, winner=winner)


def notrick():                                    # last trick with NO trump-7
    return SimpleNamespace(cards=[Card(B, Rank.EIGHT), Card(B, Rank.NINE),
                                  Card(B, Rank.TEN)], players=[0, 1, 2], winner=0)


U = BidSet(ulti=True)
PP = BidSet(piros=True)                           # piros parti
BE = BidSet(betli=True)
DU = BidSet(durchmars=True)

# each: (name, state, bid, kontras, check)  check = ("tpd", v) | ("split", (v0, v1))
CASES = [
    # ── per-component in a colored ulti+parti game (ulti made, parti won) ──
    ("kontra ulti only",  mk_u([50, 30, 0], [], last_trick=ltrick(0, 0)), U,
        {"ulti": 1}, ("tpd", +9)),          # ulti +8, parti +1
    ("kontra parti only", mk_u([50, 30, 0], [], last_trick=ltrick(0, 0)), U,
        {"parti": 1}, ("tpd", +6)),         # ulti +4, parti +2
    ("kontra both",       mk_u([50, 30, 0], [], last_trick=ltrick(0, 0)), U,
        {"ulti": 1, "parti": 1}, ("tpd", +10)),
    ("kontra ulti, bukott + parti won", mk_u([50, 30, 0], [], last_trick=ltrick(0, 1)), U,
        {"ulti": 1}, ("tpd", -11)),         # ulti bukott −12, parti +1
    # ── silent substitutes ride the parti unit (piros parti) ──
    ("kontra parti → silent piros 40-100 (milan's 8)",
        mk_u([100, 0, 0], [(0, H, 40)], trump=H, last_trick=notrick()), PP,
        {"parti": 1}, ("tpd", +8)),          # 2 ×piros ×kontra = 8
    ("kontra parti → silent piros duri",
        mk_u([90, 0, 0], [], sol_tricks=10, trump=H, last_trick=notrick()), PP,
        {"parti": 1}, ("tpd", +12)),         # 3 ×piros ×kontra = 12
    ("kontra parti → silent 40-100 AND silent duri (double both)",
        mk_u([100, 0, 0], [(0, H, 40)], sol_tricks=10, trump=H, last_trick=notrick()), PP,
        {"parti": 1}, ("tpd", +20)),         # 8 + 12
    # ── colorless SEPARATE per-defender counters (asymmetric) ──
    ("betli made, def1 kontra / def2 not", mk_b(0, betli=True), BE,
        {"betli": (1, 0)}, ("split", (+10, +5))),
    ("betli lost, def1 kontra / def2 not", mk_b(1, betli=True), BE,
        {"betli": (1, 0)}, ("split", (-10, -5))),
    ("colorless duri made, def1 rekontra / def2 kontra", mk_b(10, trump=None), DU,
        {"durchmars": (2, 1)}, ("split", (+24, +12))),
    # ── együtt sírunk: colored kontra is SHARED → symmetric, no split ──
    ("colored: shared ulti kontra → symmetric (no split)",
        mk_u([50, 30, 0], [], last_trick=ltrick(0, 0)), U, {"ulti": 1}, ("nosplit", None)),
]


def main():
    fails = 0
    for name, state, bid, kontras, (kind, want) in CASES:
        pv = score(final_pos=state, bid=bid, kontras=kontras)
        if kind == "tpd":
            got = pv.total_per_def
            ok = (got == want)
            detail = f"total_per_def {got:+d} (want {want:+d})"
        elif kind == "split":
            got = (pv.gp_vs(0), pv.gp_vs(1))
            ok = (got == want)
            detail = f"gp_vs {got} (want {want})"
        else:  # nosplit
            ok = (len(pv.def_split) == 0)
            detail = f"def_split={pv.def_split}"
        fails += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
    print("\n" + ("ALL PASS" if not fails else f"{fails} FAILED"))
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
