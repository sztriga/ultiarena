"""Kontra decision + kontra-aware EV for the SIMPLE contracts (parti/ulti/betli/
durchmars). milan 2026-06-26.

Kontra is a constant multiplier on a zero-sum payoff → optimal play & makeability
are UNCHANGED. So everything here is closed-form on the make-probability `p`:

  • contract_payoffs(bid, level)  — soloist per-def GP for made / bukott of the
    PRIMARY contract at a kontra level (handles the ulti-bukott 2x/3x/5x special +
    piros). The parti rider on a bid ulti is a small correction left to the oracle
    at scoring time — here we use primary-only for clean thresholds.
  • kontra_level_for(p)           — simulate def→def→soloist decisions → final level.
    Defenders kontra iff they expect the soloist to FAIL (soloist EV<0); soloist
    rekontra iff still +EV. v1 single-p model (god: p∈{0,1}; pimc: belief).
  • kontra_adjusted_ev(p)         — soloist EV of bidding it, given the defenders
    will kontra failures. This is what makes weak bids worse than PASS (so a hand
    finally PASSES instead of always declaring piros parti).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, "/Users/milansimity/Cuccok/kodok/oldtawer"):
    if p not in sys.path:
        sys.path.insert(0, p)

from ulti.scoring.oracle import GPTable   # noqa: E402

_GP = GPTable()


def contract_payoffs(bid, level, gp=None):
    """(made_per_def, bukott_per_def) for the primary simple contract at `level`."""
    gp = gp or _GP
    pm = 2 if bid.piros else 1
    km = 2 ** level
    if bid.ulti:
        return gp.ulti_bid * km * pm, -(km + 1) * gp.ulti_bid * pm
    if bid.betli:
        t = (4 if bid.teritett else 1)
        return gp.betli * km * pm * t, -gp.betli * km * pm * t
    if bid.durchmars:
        t = (4 if not bid.piros and bid.teritett else (2 if bid.teritett else 1))
        return gp.durchmars_bid * km * pm * t, -gp.durchmars_bid * km * pm * t
    return gp.parti * km * pm, -gp.parti * km * pm     # parti


def _sol_ev(p, bid, level, gp=None):
    m, b = contract_payoffs(bid, level, gp or _GP)
    return p * m + (1.0 - p) * b


def kontra_level_for(p, bid, gp=None, p_sol=None):
    """Final kontra level after def→def→soloist decisions, by BACKWARD INDUCTION.
    `p` = defenders' belief the soloist makes it; `p_sol` = soloist's belief.
    The defender kontras only if it leaves the soloist worse off AFTER the
    soloist's best rekontra response (so it won't kontra into a profitable
    rekontra). Under a single shared belief (god, p_sol=p) rekontra never fires;
    it only appears when the soloist is more optimistic than the defenders."""
    gp = gp or _GP
    if p_sol is None:
        p_sol = p
    # soloist's response to a kontra: rekontra iff it improves the soloist's EV
    lvl_if_kontra = 2 if _sol_ev(p_sol, bid, 2, gp) > _sol_ev(p_sol, bid, 1, gp) else 1
    # defender kontras iff that final level is worse for the soloist than no-kontra
    if _sol_ev(p, bid, lvl_if_kontra, gp) < _sol_ev(p, bid, 0, gp):
        return lvl_if_kontra
    return 0


def kontra_adjusted_ev(p, bid, gp=None):
    """Soloist EV of bidding this simple contract, given defenders kontra failures."""
    gp = gp or _GP
    return _sol_ev(p, bid, kontra_level_for(p, bid, gp), gp)


if __name__ == "__main__":
    from ladder import BidSet
    print("contract        p   no-k EV   kontra-adj EV   level")
    for name, bid in [("piros parti", BidSet(piros=True)),
                      ("ulti", BidSet(ulti=True)),
                      ("betli", BidSet(betli=True))]:
        for p in (0.1, 0.25, 0.4, 0.5, 0.6, 0.8):
            nk = _sol_ev(p, bid, 0)
            ka = kontra_adjusted_ev(p, bid)
            lv = kontra_level_for(p, bid)
            print(f"{name:<14} {p:.2f}  {nk:+6.2f}    {ka:+6.2f}        {lv}")
        print()
