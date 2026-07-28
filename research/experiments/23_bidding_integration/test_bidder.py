"""Sanity tests for the base-event composer (bidder.py).

Stubbed base-event probabilities → assert the bidder picks sensible rungs and
respects the gates (marriage holdings, hearts-for-piros, equivalent-pair choice,
pass economics).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bidder import BaseProbs, best_bid, rung_ev  # noqa: E402
from ladder import LADDER, rung_for, BidSet  # noqa: E402


def _bid_name(probs):
    b = best_bid(probs)
    return None if b is None else b[0].name


def main():
    fails = 0

    def check(cond, msg):
        nonlocal fails
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            fails += 1

    # 1. truly hopeless (hearts hand, p_parti=0) → PASS (every rung ≤ −2 floor)
    check(_bid_name(BaseProbs(p_parti=0.0, trump_is_hearts=True)) is None,
          "hopeless hearts hand passes")

    # 2. a HEARTS hand with any real parti chance bids piros parti (−0.8 beats
    #    −2 pass) — your exp-20 economics, on a hand that CAN go hearts.
    check(_bid_name(BaseProbs(p_parti=0.30, trump_is_hearts=True)) == "piros parti",
          "thin parti(♥) bids piros parti (pass only when hopeless)")

    # 2b. OPEN QUESTION (milan): a NON-hearts hand whose only game is a weak parti
    #     currently PASSES (piros parti needs hearts; plain parti not biddable).
    #     If a soloist may open piros parti by *choosing* hearts, this gate is wrong
    #     for parti specifically — flagged in STATUS.md, not asserted here.
    print(f"NOTE  non-♥ thin-parti → {_bid_name(BaseProbs(p_parti=0.30))} "
          f"(floor question — see STATUS.md)")

    # 3. strong ulti with hearts → climbs to a piros-ulti-family rung
    b = best_bid(BaseProbs(p_parti=0.75, p_ulti=0.80, trump_is_hearts=True))
    check(b is not None and "ulti" in b[0].name and b[0].piros,
          f"strong ulti(♥) bids a piros ulti rung ({b[0].name if b else None})")

    # 4. SAME hand without hearts cannot bid any colored-piros rung
    b = best_bid(BaseProbs(p_parti=0.75, p_ulti=0.80))
    check(b is not None and not b[0].piros and "ulti" in b[0].name,
          f"ulti(non-♥) bids a non-piros ulti rung ({b[0].name if b else None})")

    # 5. holding the 40 + good reach → a 40-100 rung; WITHOUT has_40 → never
    b = best_bid(BaseProbs(p_parti=0.4, p_reach100_40=0.70, has_40=True))
    check(b is not None and "40-100" in b[0].name,
          f"40-hand bids a 40-100 rung ({b[0].name if b else None})")
    # without the marriage, no 40-100 rung is ever returned
    no40 = best_bid(BaseProbs(p_reach100_40=0.99))   # high prob but no holding
    check(no40 is None or "40-100" not in no40[0].name,
          "never bids 40-100 without holding the 40")

    # 6. sweep hand → a colorless-duri rung (under the god metric a strong sweep
    #    prefers the TERÍTETT version — 4× pay at the same double-dummy risk)
    nm = _bid_name(BaseProbs(p_duri_colorless=0.80, p_betli=0.1))
    check(nm is not None and "duri" in nm and "ulti" not in nm and "100" not in nm,
          f"sweep hand bids a standalone (no-trump) duri rung ({nm})")

    # 7. equivalent-pair selection: ulti+sweep, NO marriage → must use the
    #    ulti-duri representative of the value-10 tie (not the 40-100-duri one)
    b = best_bid(BaseProbs(p_ulti=0.7, p_duri_colored=0.85, p_parti=0.9))
    check(b is not None and "ulti-duri" in b[0].name,
          f"ulti+sweep(non♥,no-40) reaches the ulti-duri equivalent ({b[0].name if b else None})")

    # 8. monotonicity: higher p_ulti ⇒ higher ulti-rung EV
    r_ulti = rung_for(BidSet(ulti=True))
    lo = rung_ev(r_ulti, BaseProbs(p_parti=0.5, p_ulti=0.3))
    hi = rung_ev(r_ulti, BaseProbs(p_parti=0.5, p_ulti=0.9))
    check(hi > lo, f"ulti EV monotone in p_ulti ({lo:+.2f} → {hi:+.2f})")

    # 9. overcall: given a held bid, only strictly-higher rungs are offered
    cur = rung_for(BidSet(ulti=True))            # rung index 2 (value 5)
    b = best_bid(BaseProbs(p_duri_colorless=0.9), current=cur)
    check(b is not None and b[0].index > cur.index,
          f"overcall returns a strictly higher rung ({b[0].index if b else None} > {cur.index})")

    print()
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
