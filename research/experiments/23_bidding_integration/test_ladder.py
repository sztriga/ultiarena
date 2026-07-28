"""Validate the generated ladder against milan's confirmed table.

Ground truth = the (value, Σsq, piros) sequence from `reference_bidding_ladder`,
walked up with milan 2026-06-16. The generator must reproduce it EXACTLY, in
order, with the two equivalent-pairs sharing a rung.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ladder import (  # noqa: E402
    LADDER, BidSet, GPTable, build_ladder, rung_for, overcalls,
    nominal_value, sigma_sq, nominal_components, contract_name,
)

# (value, Σsq, piros) — milan's confirmed strict order, pass → 36.
EXPECTED = [
    (2, 4, True),      # piros parti
    (4, 16, False),    # 40-100
    (5, 17, False),    # ulti (4+1)
    (5, 25, False),    # betli
    (6, 36, False),    # colorless duri
    (8, 32, False),    # 40-100-ulti
    (8, 64, True),     # piros 40-100
    (8, 64, False),    # 20-100
    (10, 52, False),   # ulti-duri ≡ 40-100-duri  (TIE)
    (10, 68, True),    # piros ulti
    (10, 100, True),   # rebetli (= piros betli, scored ×2)
    (12, 80, False),   # 20-100-ulti
    (14, 68, False),   # ulti-40-100-duri
    (14, 100, False),  # 20-100-duri
    (16, 128, True),   # piros 40-100-ulti
    (16, 256, True),   # piros 20-100
    (18, 116, False),  # ulti-20-100-duri
    (20, 208, True),   # piros-ulti-duri ≡ piros-40-100-duri  (TIE)
    (24, 320, True),   # piros 20-100-ulti
    (28, 272, True),   # piros-ulti-40-100-duri
    (28, 400, True),   # piros-20-100-duri
    (36, 464, True),   # piros-ulti-20-100-duri
]

# rung index → set of expected interchangeable contract names
EXPECTED_TIES = {
    8: {"ulti-duri", "40-100-duri"},
    17: {"piros ulti-duri", "piros 40-100-duri"},
}


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    fails = 0
    # the non-terített rungs must still reproduce milan's confirmed 22-rung table
    # (terített is a parallel layer interleaved by value; check it as a subsequence)
    non_ter = [r for r in LADDER if not any(b.teritett for b in r.bids)]

    # 1. non-terített rung count
    try:
        _check(len(non_ter) == len(EXPECTED),
               f"non-terített rung count {len(non_ter)} != {len(EXPECTED)}")
        print(f"PASS  non-terített rungs = {len(non_ter)}  (full ladder = {len(LADDER)})")
    except AssertionError as e:
        print(f"FAIL  {e}"); fails += 1

    # 2. exact ordered (value, Σsq, piros) sequence of the non-terített rungs
    got = [(r.value, r.sigma_sq, r.piros) for r in non_ter]
    for i, (g, e) in enumerate(zip(got, EXPECTED)):
        if g != e:
            print(f"FAIL  non-ter rung {i}: got {g} expected {e}  ({non_ter[i].name})")
            fails += 1
    if got == EXPECTED:
        print("PASS  non-terített sequence matches milan's table")

    # 2t. terített layer: ladder tops at the 48-point game, known values present
    top = LADDER[-1]
    _check(top.value == 48, f"top rung is {top.value} not 48 ({top.name})")
    print(f"{'PASS' if top.value==48 else 'FAIL'}  ladder tops at 48: {top.name}")
    tb = [r for r in LADDER if r.value == 20 and any(b.betli and b.teritett for b in r.bids)]
    tcd = [r for r in LADDER if r.value == 24 and any(b.durchmars and b.teritett for b in r.bids)]
    _check(bool(tb), "terített betli (20) missing")
    _check(bool(tcd), "terített colorless duri (24) missing")
    print(f"{'PASS' if tb and tcd else 'FAIL'}  terített betli=20 & colorless duri=24 present")

    # 3. equivalent pairs share a rung with identical components (find by content,
    #    since terített interleaving shifts absolute indices)
    for names in EXPECTED_TIES.values():
        match = [r for r in LADDER if set(r.names) == names]
        if not match:
            print(f"FAIL  equivalent pair {names} not a shared rung"); fails += 1
        elif len(set(nominal_components(b) for b in match[0].bids)) != 1:
            print(f"FAIL  {names} equivalents have differing components"); fails += 1
        else:
            print(f"PASS  equivalent pair (rung {match[0].index}): {names}")

    # 4. plain non-piros parti is NOT biddable; piros parti is rung 0
    if rung_for(BidSet()) is not None:
        print("FAIL  plain non-piros parti is biddable (should not be)"); fails += 1
    else:
        print("PASS  plain non-piros parti not biddable")
    r0 = rung_for(BidSet(piros=True))
    if r0 is None or r0.index != 0 or r0.value != 2:
        print(f"FAIL  piros parti not rung 0/value 2: {r0}"); fails += 1
    else:
        print("PASS  piros parti = rung 0 (value 2)")

    # 5. piros standalone duri is NOT biddable; colorless duri is (value 6, no-trump)
    if rung_for(BidSet(durchmars=True, piros=True)) is not None:
        print("FAIL  piros standalone duri is biddable (should not be)"); fails += 1
    else:
        print("PASS  piros standalone duri not biddable")
    rcd = rung_for(BidSet(durchmars=True))
    if rcd is None or rcd.value != 6 or not rcd.colorless:
        print(f"FAIL  colorless duri wrong: {rcd}"); fails += 1
    else:
        print("PASS  colorless duri = value 6, colorless")

    # 6. betli double-rung: betli (5) and rebetli (10) both present
    rb = rung_for(BidSet(betli=True))
    rr = rung_for(BidSet(betli=True, piros=True))
    if rb is None or rb.value != 5 or rr is None or rr.value != 10:
        print(f"FAIL  betli/rebetli double-rung wrong: {rb} / {rr}"); fails += 1
    else:
        _check(rr.index > rung_for(BidSet(ulti=True, piros=True)).index,
               "rebetli should rank above piros ulti")
        print("PASS  betli (5) + rebetli (10) double-rung; rebetli > piros ulti")

    # 7. overcalls are strictly-higher rungs; equivalents not re-offered
    cur = LADDER[8]                       # the ulti-duri ≡ 40-100-duri tie
    oc = overcalls(cur)
    if any(o.index <= 8 for o in oc):
        print("FAIL  overcalls included current/lower rung"); fails += 1
    elif oc[0].index != 9:
        print(f"FAIL  first overcall not rung 9: {oc[0]}"); fails += 1
    else:
        print("PASS  overcalls above the equivalent-tie rung start at rung 9")

    # 8. terített standalone values (confirmed): betli 20, colorless duri 24
    tb = nominal_value(BidSet(betli=True, teritett=True))
    tcd = nominal_value(BidSet(durchmars=True, teritett=True))
    ptcd = nominal_value(BidSet(durchmars=True, teritett=True, piros=False))
    if tb != 20:
        print(f"FAIL  terített betli = {tb} expected 20"); fails += 1
    else:
        print("PASS  terített betli = 20")
    if tcd != 24:
        print(f"FAIL  terített colorless duri = {tcd} expected 24"); fails += 1
    else:
        print("PASS  terített colorless duri = 24")

    print()
    if fails == 0:
        print(f"ALL CHECKS PASS ({len(LADDER)} rungs)")
    else:
        print(f"{fails} CHECK(S) FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
