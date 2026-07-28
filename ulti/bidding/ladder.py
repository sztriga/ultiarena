"""The full Ulti bidding ladder — strict value-ordered rung sequence.

A *rung* is a position in the auction. Every biddable contract maps to exactly
one rung; a rung may hold several **interchangeable** contracts (the two
"equivalent" pairs milan confirmed: ulti-duri ≡ 40-100-duri at value 10, and
their piros twins at 20).

The ladder is GENERATED from the combo basis, not hand-listed, then validated
against milan's confirmed table in `test_ladder.py`:

    independent flags   {ulti, durchmars, points ∈ (–, 40-100, 20-100), piros}
    + the betli family  {betli, rebetli}

Each contract's *nominal made value* and its *components* come straight from the
oracle's GP rates (a made contract pays exactly the sum of its component rates).
Ordering rule (milan 2026-06-16):

    sort by (net value asc, Σsq of components asc, piros-before-nonpiros)

Σsq = sum of squares of the made components ("X plus Y" additive parts). Higher
Σsq ranks higher. The two equivalent pairs fall out as identical (value, Σsq,
components) — not a special case.

This module is PURE LOGIC (no solver, no net). It answers: "what can I bid above
the current high bid, and in what order do rungs sit." See `reference_bidding_ladder`.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

from ulti.scoring.oracle import BidSet, GPTable  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Nominal made value + components of a single BidSet
# ─────────────────────────────────────────────────────────────────────

def nominal_components(bid: BidSet, gp: Optional[GPTable] = None) -> Tuple[int, ...]:
    """The made-magnitudes of each scoring component of ``bid`` (soloist made
    everything). Mirrors the oracle's made-case arithmetic — value = sum,
    Σsq = sum of squares. Sorted ascending for a canonical signature.

    - ulti           → ulti_bid (4)
    - durchmars      → durchmars_bid (6)        × terített (×2 colored / ×4 colorless)
    - betli          → betli (5)                × terített
    - 40-100         → forty_hundred_bid (4)
    - 20-100         → twenty_hundred_bid (8)
    - implicit parti → parti (1)   only in a points-based game with no bid-100
    - piros          → ×2 on ALL components (applied last)
    """
    if gp is None:
        gp = GPTable()
    comps: List[int] = []
    colorless = _is_colorless(bid)
    terit = (4 if colorless else 2) if bid.teritett else 1
    points_based = not (bid.durchmars or bid.betli)
    bid_a_100 = bid.forty_hundred or bid.twenty_hundred

    if bid.ulti:
        comps.append(gp.ulti_bid)
    if bid.durchmars:
        comps.append(gp.durchmars_bid * terit)
    if bid.betli:
        comps.append(gp.betli * terit)
    if bid.forty_hundred:
        comps.append(gp.forty_hundred_bid)
    if bid.twenty_hundred:
        comps.append(gp.twenty_hundred_bid)
    if points_based and not bid_a_100:
        comps.append(gp.parti)             # the implicit parti base ("4+1")

    if bid.piros:
        comps = [c * 2 for c in comps]
    return tuple(sorted(comps))


def nominal_value(bid: BidSet, gp: Optional[GPTable] = None) -> int:
    return sum(nominal_components(bid, gp))


def sigma_sq(bid: BidSet, gp: Optional[GPTable] = None) -> int:
    return sum(c * c for c in nominal_components(bid, gp))


def _is_colorless(bid: BidSet) -> bool:
    """A contract is played colorless (no trump) iff it is a standalone betli or
    a standalone durchmars (the "duri 6" rung). Once durchmars is combined with
    ulti or a 100 it is the COLORED duri (lives in a trump game)."""
    if bid.betli:
        return True
    if bid.durchmars and not (bid.ulti or bid.forty_hundred or bid.twenty_hundred):
        return True
    return False


def _is_legal_contract(bid: BidSet) -> bool:
    """Which BidSets are biddable at all (milan's ladder rules)."""
    # plain non-piros parti is NOT biddable — only PASS or piros parti
    is_parti_base = not (bid.ulti or bid.durchmars or bid.betli
                         or bid.forty_hundred or bid.twenty_hundred)
    if is_parti_base and not bid.piros:
        return False
    # standalone piros duri ("piros duri we don't play" — the one ×2 exception;
    # a colorless game can't be hearts). In a combo, colored piros duri is fine.
    standalone_duri = (bid.durchmars and not (bid.ulti or bid.forty_hundred
                                              or bid.twenty_hundred or bid.betli))
    if standalone_duri and bid.piros:
        return False
    # betli does not combine with anything in milan's ladder
    if bid.betli and (bid.ulti or bid.durchmars or bid.forty_hundred
                      or bid.twenty_hundred):
        return False
    return True


def contract_name(bid: BidSet) -> str:
    """Human label for a contract (display + tests)."""
    parts: List[str] = []
    if bid.ulti:
        parts.append("ulti")
    if bid.forty_hundred:
        parts.append("40-100")
    if bid.twenty_hundred:
        parts.append("20-100")
    if bid.durchmars:
        parts.append("duri")
    if bid.betli:
        return ("teritett " if bid.teritett else "") + ("rebetli" if bid.piros else "betli")
    if not parts:
        parts.append("parti")
    name = "-".join(parts)
    prefix = ""
    if bid.piros:
        prefix += "piros "
    if bid.teritett:
        prefix += "teritett "
    return prefix + name


# ─────────────────────────────────────────────────────────────────────
# Rung — a position on the ladder (one or more interchangeable contracts)
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rung:
    index: int                      # ladder position (0 = lowest bid above pass)
    value: int                      # net GP/def when made
    sigma_sq: int                   # Σ of squares of components (tiebreak)
    piros: bool
    colorless: bool                 # no-trump (betli / colorless duri)
    components: Tuple[int, ...]     # made-magnitudes (sorted)
    bids: Tuple[BidSet, ...]        # interchangeable contracts on this rung
    names: Tuple[str, ...]          # parallel human names

    @property
    def name(self) -> str:
        return " ≡ ".join(self.names)


# Sort key: value asc, Σsq asc, piros BEFORE non-piros at equal (value, Σsq).
def _order_key(value: int, ssq: int, piros: bool) -> Tuple[int, int, int]:
    return (value, ssq, 0 if piros else 1)


def _signature(bid: BidSet, gp: GPTable) -> Tuple[int, int, bool, Tuple[int, ...]]:
    """Two BidSets sharing this signature are interchangeable (same rung)."""
    return (nominal_value(bid, gp), sigma_sq(bid, gp), bid.piros,
            nominal_components(bid, gp))


def _enumerate_bidsets() -> List[BidSet]:
    """All structurally-legal biddable BidSets, INCLUDING the terített layer
    (open-hand ×2 on the duri/betli component → the ladder tops out at the
    48-point game, terített piros ulti-20-100-duri). terített only matters when a
    duri or betli component is present, so it is enumerated only there."""
    out: List[BidSet] = []
    for ulti in (False, True):
        for dm in (False, True):
            for points in (None, "forty_hundred", "twenty_hundred"):
                for piros in (False, True):
                    for teritett in (False, True):
                        if teritett and not dm:   # terit doubles the duri component
                            continue
                        bid = BidSet(
                            ulti=ulti,
                            durchmars=dm,
                            forty_hundred=(points == "forty_hundred"),
                            twenty_hundred=(points == "twenty_hundred"),
                            piros=piros,
                            teritett=teritett,
                        )
                        if _is_legal_contract(bid):
                            out.append(bid)
    for piros in (False, True):          # betli family: betli 5p, rebetli 10p, terített betli 20p
        for teritett in (False, True):
            if piros and teritett:        # no "terített rebetli" — not a real contract
                continue
            bid = BidSet(betli=True, piros=piros, teritett=teritett)
            if _is_legal_contract(bid):
                out.append(bid)
    return out


def build_ladder(gp: Optional[GPTable] = None) -> Tuple[Rung, ...]:
    """Generate the full strict-ordered ladder of rungs."""
    if gp is None:
        gp = GPTable()
    # group BidSets by signature (interchangeable contracts share a rung)
    groups: dict = {}
    for bid in _enumerate_bidsets():
        groups.setdefault(_signature(bid, gp), []).append(bid)

    rungs_raw = []
    for sig, bids in groups.items():
        value, ssq, piros, comps = sig
        rungs_raw.append((value, ssq, piros, comps, bids))
    rungs_raw.sort(key=lambda r: _order_key(r[0], r[1], r[2]))

    rungs: List[Rung] = []
    for i, (value, ssq, piros, comps, bids) in enumerate(rungs_raw):
        bids_t = tuple(bids)
        rungs.append(Rung(
            index=i, value=value, sigma_sq=ssq, piros=piros,
            colorless=all(_is_colorless(b) for b in bids_t),
            components=comps,
            bids=bids_t,
            names=tuple(contract_name(b) for b in bids_t),
        ))
    return tuple(rungs)


LADDER: Tuple[Rung, ...] = build_ladder()


# ─────────────────────────────────────────────────────────────────────
# Auction controller helpers
# ─────────────────────────────────────────────────────────────────────

def _same_contract(a: BidSet, b: BidSet) -> bool:
    return (a.ulti == b.ulti and a.durchmars == b.durchmars
            and a.forty_hundred == b.forty_hundred
            and a.twenty_hundred == b.twenty_hundred
            and a.betli == b.betli and a.piros == b.piros
            and a.teritett == b.teritett)


def rung_for(bid: BidSet) -> Optional[Rung]:
    """The ladder rung a contract sits on (None if not biddable)."""
    for r in LADDER:
        if any(_same_contract(bid, b) for b in r.bids):
            return r
    return None


def overcalls(current: Optional[Rung]) -> List[Rung]:
    """Rungs you may declare to overtake ``current`` (None = open the auction).

    You must go STRICTLY higher than the current rung index — you cannot take an
    *equivalent* contract on the same rung, you must climb past it (milan: an
    overtaker "must climb to the next step")."""
    floor = -1 if current is None else current.index
    return [r for r in LADDER if r.index > floor]


# Pass economics (milan): passing the would-be soloist forfeits −2/def — NOT a
# neutral floor. gps = [-4, +2, +2] (P0 pays 2 to each opponent). Piros parti's
# EV/def = 4p-2 ties the pass at p=0 and beats it for any p>0 → you bid the
# floor (piros parti) whenever you have a real chance; pass only when hopeless.
PASS_VALUE_PER_DEF: int = -2


def describe_ladder(gp: Optional[GPTable] = None) -> str:
    rungs = LADDER if gp is None else build_ladder(gp)
    lines = [f"{'rung':>4}  {'value':>5} {'Σsq':>5}  {'flavor':<9} {'contract(s)'}"]
    lines.append(f"{'PASS':>4}  {PASS_VALUE_PER_DEF:>5}     -  {'-':<9} (forfeit −2/def)")
    for r in rungs:
        flavor = "colorless" if r.colorless else ("piros" if r.piros else "color")
        lines.append(f"{r.index:>4}  {r.value:>5} {r.sigma_sq:>5}  {flavor:<9} {r.name}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_ladder())
    print(f"\n{len(LADDER)} rungs.")
