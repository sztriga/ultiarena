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

from ulti.scoring.oracle import GPTable

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


# ── The deployed in-game defender rule (exp27) ──────────────────────────────────
# Validated 2026-07-21 on a held-out tournament vs the old blind-makeability rule:
# +7.7 GP/deal for the defenders, cheat-clean. The old rule `_sol_ev(blind) < 0`
# wildly over-kontra'd — it sampled RANDOM soloist hands and ignored that the
# soloist BID the contract, so it "saw" ~6-11% makeability against a true ~80%,
# kontra'd makeable ulti/parti and paid double (the rekontra amplified the loss).
# Per-unit calibrated signals instead, from the defender's OWN hand only.
KONTRA_ULTI_TRUMPS = 4     # ulti: kontra iff this defender holds >=4 trumps (make ~32%; 3→76%)
KONTRA_DURI_TRUMPS = 3     # colored durchmars: kontra iff >=3 trumps (make ~2-5%; 0→50%)

# PARTI: no kontra. Removed 2026-08-03 (exp47, 11,994 defender-positions on the
# honest post-talon-fix frontier). The old rule was `blind makeability < 0.10` and
# it LOST in both rekontra worlds — change in soloist GP vs never-kontra +0.077
# (t+6.0) with no rekontra, +1.133 (t+39.7) under a re-doubling soloist. It fired
# on 58% of positions at a 53% make rate: a coin flip, doubled.
#
# No threshold on that signal can work. The estimate is biased −0.490 (mean 0.127
# against a true 0.617) and its reliability curve is so compressed that even its
# lowest bin [0.00,0.05) still makes 49.4% — it never separates below-50% from
# above-50%. Structural replacements were swept too; every one that looked good
# without a rekontra died with one, because the breakeven is 50% ONLY if the
# soloist cannot re-double:
#     trump 40 alone     fires 7.2%, makes 25.0%   -0.062 -> +0.007   (neutral)
#     two 20s alone      fires 3.9%, makes 44.4%   +0.000 -> +0.068   (loses)
#     4+ trumps alone    fires 3.5%, makes 34.1%   -0.023 -> +0.023   (loses)
#     40 AND a 20        fires 2.1%, makes 14.9%   -0.025 -> -0.012   (only survivor)
# The sole rule negative in both worlds is worth -0.012 GP per position —
# indistinguishable from abstaining, and not worth a special case. Removing it
# also deletes a PIMC solve per defender per deal: faster AND better.
#
# NB this is a verdict against the FRONTIER, which passes ~39% of deals and makes
# 65-94% of what it bids, so it hands the defenders nothing worth doubling.
# Against a human who over-bids the answer may differ.


def defender_kontras_unit(hand, trump, unit) -> bool:
    """Does a defender holding `hand` kontra `unit`? Cheat-clean: own hand only.

    THE rule — the live game (apps.api.kontra_flow) and the human-play evaluator
    (ulti.eval.human_eval) both call this, so "what the engine would have done"
    can never drift from what it actually does.

    Trump count is the decisive signal for the trick contracts (ulti/duri); parti,
    20-100 and betli/colorless-duri abstain, because no own-hand signal beats not
    kontra-ing (see the exp47 note above).
    """
    if unit == "ulti":
        return sum(1 for c in hand if c.suit == trump) >= KONTRA_ULTI_TRUMPS
    if unit == "durchmars" and trump is not None:
        return sum(1 for c in hand if c.suit == trump) >= KONTRA_DURI_TRUMPS
    if unit == "40_100" and trump is not None:
        # milan 2026-07-23: a 40-100 declares the TRUMP marriage (the "40"). Holding
        # either of its cards means the soloist cannot hold the full marriage → the
        # 40-100 is unmakeable → auto-kontra.
        return any(c.suit == trump and c.rank in ("king", "upper") for c in hand)
    # 20-100: the 20's colour is NOT declared, so the trump test doesn't apply — the
    # rule is about the NON-trump marriages and is still being pinned down with milan.
    return False               # 20-100 / betli / colorless durchmars → abstain


if __name__ == "__main__":
    from ulti.bidding.ladder import BidSet
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
