"""Composer: base-event probabilities → expected GP of every ladder rung → bid.

This is the heart of the integration thesis (milan + Claude 2026-06-17): DON'T
train a per-rung policy. Predict a handful of base events with the value nets,
then COMPOSE each rung's EV from the oracle's GP rates (which score components
INDEPENDENTLY) and the ladder arithmetic, and pick the bid off the ladder.

Each rung's EV is a SUM of per-component expectations — exactly mirroring how
`scoring/oracle` scores a made deal component-by-component:

    E[component] = p_make·(+made_rate) − (1−p_make)·(bukott_rate)

with the base-event probability supplying p_make. ulti's bukott pays double
(`ulti_bid_bukott`); the others are symmetric. piros ×2 at the end; terített ×2/×4
on the duri/betli component (already in the rung).

v0 APPROXIMATIONS (flagged for milan — all in `rung_ev`):
  • base-event probabilities are treated as MARGINAL & INDEPENDENT across a combo
    (e.g. P(ulti-duri made) uses p_ulti and p_sweep separately). Real hands
    correlate; a joint head or a copula correction is the v1.
  • SILENT riders (silent ulti / silent 100 bonuses that ride on a bid game it
    wasn't declared in) are NOT added to bid EV here — they're a positive bias,
    so this UNDER-states EV slightly. Easy to add once the nets expose them.
  • auction strategy is greedy "declare the max-EV rung that beats the pass floor";
    real play opens low, responds to raises, and bluffs (rebetli). v1.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ladder import (  # noqa: E402
    LADDER, Rung, GPTable, overcalls, PASS_VALUE_PER_DEF,
)


@dataclass
class BaseProbs:
    """Per-hand base-event make-probabilities (from the value nets) + the marriage
    gates (you can only bid a 100 you hold)."""
    p_parti:          float = 0.0
    p_ulti:           float = 0.0   # bid ulti made (trick 10 w/ trump-7)
    p_reach100_40:    float = 0.0   # force ≥100 holding the trump 40
    p_reach100_20:    float = 0.0   # force ≥100 holding a non-trump 20
    p_duri_colored:   float = 0.0   # sweep all 10 WITH trump (combos)
    p_duri_colorless: float = 0.0   # sweep all 10 no-trump (standalone duri)
    p_betli:          float = 0.0   # take 0 tricks (double-dummy / perfect defense)
    p_betli_real:     float = 0.0   # take 0 tricks vs REALISTIC (imperfect) defense — exp37.
                                    # 0 = not loaded → betli EV falls back to the god p_betli.
    has_40:           bool = False
    has_20:           bool = False
    trump_is_hearts:  bool = False  # soloist's chosen/best trump is hearts → piros
                                    # COLORED rungs are available (base-event probs
                                    # are trump-suit-specific; piros is NOT a free 2×)


import os as _os
KONTRA = _os.environ.get("KONTRA", "0").lower() in ("1", "true", "yes")
# FLOOR: don't DECLARE a contract if any of its bid components has make-prob below
# this bar — kills the rare-head over-firing (20-100 / terített combos) that bleeds.
# 0 = off. Applies only to escalations (piros parti has no components → never gated).
_FLOOR = float(_os.environ.get("FLOOR", "0"))
# DURI_TERIT_MULT (default 1.0 = OFF, deployed behaviour unchanged): terített (open-hand)
# durchmars is far harder than closed-hand — defenders see the soloist's cards and block a
# trick — but the composer scores it with the CLOSED-hand p_duri and then AMPLIFIES the value
# ×2/×4. With the duri heads being the worst-calibrated (overconfident), that over-values
# terített-duri massively (exp29: made only 13–26%, −0.7 GP/deal, the frontier's #1 leak).
# This multiplies the terített-duri make-prob down before the EV. Being tuned (exp30).
_DURI_TERIT_MULT = float(_os.environ.get("DURI_TERIT_MULT", "1.0"))
# BETLI_REAL (default OFF, deployed behaviour unchanged): score PLAIN betli by the REALISTIC-defense
# make-prob (`p_betli_real`, exp37) instead of the double-dummy `p_betli`. This lets the bidder bid
# imperfect / bluff betli (dd-lost but ~60% made vs real defenders) and stops terített betli from
# always dominating plain (terított keeps the god p_betli — open hand ≈ perfect-info defense).
_BETLI_REAL = _os.environ.get("BETLI_REAL", "0").lower() in ("1", "true", "yes")
# REBETLI_REAL (default OFF): extend the realistic-defense prob to REBETLI too — the HIDDEN 10p doubling
# (defenders are still blind, so the realistic prob is the right make estimate, NOT the god prob). rebetli
# then out-bids plain betli on a strong betli (2× stake, same prob), so it's gated behind a HIGHER floor
# (_REBETLI_FLOOR) than plain betli: only escalate to the doubled hidden betli when the realistic make is
# near-certain (milan: "go betli first, escalate to rebetli when confident"). terített keeps the god prob
# (open hand ≈ perfect-info defense). REBETLI_FLOOR default 0.90.
_REBETLI_REAL = _os.environ.get("REBETLI_REAL", "0").lower() in ("1", "true", "yes")
_REBETLI_FLOOR = float(_os.environ.get("REBETLI_FLOOR", "0.90"))


def _betli_prob(bid, probs, use_real, rebetli_real=False) -> float:
    """Which betli make-prob to use. PLAIN betli (use_real) and REBETLI (rebetli_real) are both HIDDEN
    (defenders blind) → realistic-defense prob. terített (open hand ≈ perfect-info defense) keeps the
    double-dummy p_betli. Rebetli's higher-stake gate is applied in the FLOOR check, not here."""
    if bid.teritett or probs.p_betli_real <= 0.0:
        return probs.p_betli
    if bid.piros:                                 # rebetli (hidden 10p doubling)
        return probs.p_betli_real if rebetli_real else probs.p_betli
    return probs.p_betli_real if use_real else probs.p_betli   # plain betli


def _is_simple(bid) -> bool:
    """A simple contract = parti / ulti / betli / durchmars, no 100/combo/terített."""
    if bid.forty_hundred or bid.twenty_hundred or bid.teritett:
        return False
    return (int(bid.ulti) + int(bid.durchmars) + int(bid.betli)) <= 1


def _primary_p(bid, rung, probs, use_real=False, rebetli_real=False) -> float:
    if bid.ulti:
        return probs.p_ulti
    if bid.betli:
        return _betli_prob(bid, probs, use_real, rebetli_real)
    if bid.durchmars:
        return probs.p_duri_colorless if rung.colorless else probs.p_duri_colored
    return probs.p_parti


def _bid_ev(bid, rung: Rung, probs: BaseProbs, gp: GPTable,
            floor=None, duri_mult=None, betli_real=None, rebetli_real=None) -> Optional[float]:
    """EV of one specific contract (BidSet). None if not biddable with this hand.
    `floor`/`duri_mult`/`betli_real`/`rebetli_real` override the module globals per-call (None = use the
    global) — lets a head-to-head run two bidder CONFIGS in one process."""
    floor = _FLOOR if floor is None else floor
    duri_mult = _DURI_TERIT_MULT if duri_mult is None else duri_mult
    use_real = _BETLI_REAL if betli_real is None else betli_real
    use_rebetli = _REBETLI_REAL if rebetli_real is None else rebetli_real
    if bid.forty_hundred and not probs.has_40:
        return None
    if bid.twenty_hundred and not probs.has_20:
        return None
    # a COLORED piros rung requires hearts to be the trump; rebetli (colorless
    # piros betli) is exempt — it's a doubling, not a hearts declaration.
    if bid.piros and not rung.colorless and not probs.trump_is_hearts:
        return None

    if floor > 0.0:                              # suppress uncertain declarations
        if bid.betli:
            # betli is always standalone; rebetli (the hidden 10p doubling) needs a HIGHER floor than
            # plain betli — only escalate to the double when the realistic make is near-certain.
            bp = _betli_prob(bid, probs, use_real, use_rebetli)
            bf = _REBETLI_FLOOR if (bid.piros and not bid.teritett and use_rebetli) else floor
            if bp < bf:
                return None
        else:
            cp = []
            if bid.ulti:
                cp.append(probs.p_ulti)
            if bid.durchmars:
                cp.append(probs.p_duri_colorless if rung.colorless else probs.p_duri_colored)
            if bid.forty_hundred:
                cp.append(probs.p_reach100_40)
            if bid.twenty_hundred:
                cp.append(probs.p_reach100_20)
            if cp and min(cp) < floor:
                return None

    # kontra-aware EV (simple contracts): defenders kontra failures → weak bids
    # become worse than the −2 pass, so the hand finally PASSES. milan 2026-06-26.
    if KONTRA and _is_simple(bid):
        from kontra import kontra_adjusted_ev
        return kontra_adjusted_ev(_primary_p(bid, rung, probs, use_real, use_rebetli), bid, gp)

    ev = 0.0
    if bid.ulti:
        p = probs.p_ulti
        ev += p * gp.ulti_bid - (1.0 - p) * gp.ulti_bid_bukott   # bukott doubles
    if bid.durchmars:
        p = probs.p_duri_colorless if rung.colorless else probs.p_duri_colored
        if bid.teritett:
            p *= duri_mult            # open-hand duri is much harder than the closed p_duri
        terit = (4 if rung.colorless else 2) if bid.teritett else 1
        ev += (2.0 * p - 1.0) * gp.durchmars_bid * terit          # ±6 symmetric
    if bid.betli:
        p = _betli_prob(bid, probs, use_real, use_rebetli)        # plain/rebetli→realistic, terített→god
        terit = 4 if bid.teritett else 1                          # betli is colorless
        ev += (2.0 * p - 1.0) * gp.betli * terit
    if bid.forty_hundred:
        p = probs.p_reach100_40
        ev += (2.0 * p - 1.0) * gp.forty_hundred_bid
    if bid.twenty_hundred:
        p = probs.p_reach100_20
        ev += (2.0 * p - 1.0) * gp.twenty_hundred_bid

    points_based = not (bid.durchmars or bid.betli)
    bid_a_100 = bid.forty_hundred or bid.twenty_hundred
    if points_based and not bid_a_100:                            # implicit parti rider
        p = probs.p_parti
        ev += (2.0 * p - 1.0) * gp.parti

    # ─── SILENT riders (achievement-minus-bid): the bonus you bank for a silent
    # game you did NOT declare. Approximated as p × silent_rate (a rational
    # player only banks it when favourable → upside, no bukott). The exact
    # replace-vs-stack accounting is the oracle's at scoring time; this is a
    # ranking refinement so weak-hand bids carry their true upside. milan 2026-06-21.
    trump_present = not rung.colorless
    if trump_present and not bid.ulti:
        ev += probs.p_ulti * gp.ulti_silent                 # silent ulti (+2)
    if not bid.durchmars and points_based:
        ev += probs.p_duri_colored * gp.durchmars_silent    # silent duri (+3)
    if not bid.forty_hundred and probs.has_40:
        ev += probs.p_reach100_40 * gp.forty_hundred_silent  # silent 40-100 (+2)
    if not bid.twenty_hundred and probs.has_20:
        ev += probs.p_reach100_20 * gp.twenty_hundred_silent  # silent 20-100 (+4)

    if bid.piros:
        ev *= 2.0
    return ev


def rung_ev(rung: Rung, probs: BaseProbs, gp: Optional[GPTable] = None,
            floor=None, duri_mult=None, betli_real=None, rebetli_real=None) -> Optional[float]:
    """Expected GP/def of declaring this rung — the BEST of its interchangeable
    contracts (for the equivalent pairs, you may declare EITHER, so take whichever
    this hand can actually bid and scores higher). None if none is biddable.
    `floor`/`duri_mult`/`betli_real`/`rebetli_real` override the module globals per-call."""
    if gp is None:
        gp = GPTable()
    best = None
    for bid in rung.bids:
        ev = _bid_ev(bid, rung, probs, gp, floor=floor, duri_mult=duri_mult,
                     betli_real=betli_real, rebetli_real=rebetli_real)
        if ev is not None and (best is None or ev > best):
            best = ev
    return best


def rank_bids(probs: BaseProbs, current: Optional[Rung] = None,
              gp: Optional[GPTable] = None) -> List[Tuple[float, Rung]]:
    """All legal overcalls above `current`, scored, best-EV first."""
    out = []
    for r in overcalls(current):
        ev = rung_ev(r, probs, gp)
        if ev is not None:
            out.append((ev, r))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def best_bid(probs: BaseProbs, current: Optional[Rung] = None,
             gp: Optional[GPTable] = None) -> Optional[Tuple[Rung, float]]:
    """Greedy bid for ONE trump choice: highest-EV legal rung that beats the pass
    floor. Returns None = PASS. (See v0 approximations in the module docstring.)"""
    ranked = rank_bids(probs, current, gp)
    if not ranked:
        return None
    ev, rung = ranked[0]
    if ev <= float(PASS_VALUE_PER_DEF):
        return None
    return (rung, ev)


def best_bid_over_trumps(probs_by_trump: dict, current: Optional[Rung] = None,
                         gp: Optional[GPTable] = None):
    """The REAL bid decision: you choose your trump, so evaluate every candidate
    trump's base-event probs and take the global best (rung, trump, EV).

    `probs_by_trump` maps suit → BaseProbs for declaring that suit as trump (probs
    are trump-specific; the 'hearts' entry must set trump_is_hearts=True, which
    unlocks the piros rungs). Because HEARTS is always a candidate, **piros parti
    is always available** — so for now a hand never passes (piros parti EV = 4p−2
    dominates the −2 pass; milan 2026-06-18). kontra will change that later.

    Returns (rung, trump, ev) or None (pass — only if every trump's best ≤ floor)."""
    best = None
    for trump, pr in probs_by_trump.items():
        for ev, rung in rank_bids(pr, current, gp):
            if best is None or ev > best[2]:
                best = (rung, trump, ev)
    if best is None or best[2] <= float(PASS_VALUE_PER_DEF):
        return None
    return best


if __name__ == "__main__":
    # quick demo on a few archetypes
    demos = {
        "hopeless":        BaseProbs(p_parti=0.0),
        "thin parti":      BaseProbs(p_parti=0.30),
        "solid parti(♥)":  BaseProbs(p_parti=0.80, trump_is_hearts=True),
        "ulti (non-♥)":    BaseProbs(p_parti=0.75, p_ulti=0.70),
        "ulti (♥)":        BaseProbs(p_parti=0.75, p_ulti=0.70, trump_is_hearts=True),
        "40-100 hand":     BaseProbs(p_parti=0.7, p_reach100_40=0.65, has_40=True),
        "sweep (duri)":    BaseProbs(p_duri_colorless=0.75, p_betli=0.2),
        "ulti+sweep(non♥)": BaseProbs(p_ulti=0.7, p_duri_colored=0.7, p_parti=0.9),
        "ulti+sweep(♥)":   BaseProbs(p_ulti=0.7, p_duri_colored=0.7, p_parti=0.9,
                                     trump_is_hearts=True),
    }
    for name, pr in demos.items():
        b = best_bid(pr)
        if b is None:
            print(f"{name:<14} → PASS")
        else:
            print(f"{name:<14} → {b[0].name:<28} EV/def {b[1]:+.2f}")
