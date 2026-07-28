"""Scoring oracle — pure function from played-out position to payoff vector.

First sketch. Critique welcome.

The oracle answers: "given this final position and what was bid, how
many GP does each side win/lose, broken down by rule component?"

It does NOT play the game. It does not call a solver. It just looks
at ``state.captured``, ``state.marriages``, ``state.talon_discards``,
``state.last_trick`` and applies the rulebook component by component.

Sign convention: every value in the returned vector is **signed GP
from the soloist's perspective, per defender** (so total sol gain =
2 × value, and each defender pays that value). Positive = soloist
gained, negative = soloist lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ── kontra units ──────────────────────────────────────────────────────
# Each scoring component rides a KONTRA UNIT. Silent substitutes (silent /
# def_silent 100 & duri that replace the parti) ride the PARTI unit, so
# "kontra parti" scales them too (milan 2026-07-05). Silent ulti rides no unit
# (unbid → scored flat, no kontra). A component's unit is kontra'd independently.
_PARTI_UNIT = {
    "parti", "silent_40_100", "silent_20_100", "silent_durchmars",
    "def_silent_40_100", "def_silent_20_100", "def_silent_durchmars",
}


def _unit_of(key: str) -> Optional[str]:
    if key in _PARTI_UNIT:
        return "parti"
    if key in ("40_100", "20_100", "ulti", "betli", "durchmars"):
        return key
    return None                      # silent_ulti (unbid) → no kontra unit

from ultisolver.games.ulti.game import (
    GameState,
    soloist_won_durchmars,
    soloist_tricks,
    soloist_has_40,
    soloist_has_20,
    soloist_points,
    defender_points,
    defender_has_40,
    defender_has_20,
    marriage_points,
    last_trick_ulti_check,
)


def _best_silent_100(card_pts, has_40, has_20, gp, on_40, on_20):
    """The single best silent-100 a side qualifies for (only ONE marriage may
    count; 20-100 beats 40-100). Returns (component_key, value) or None.
    Non-piros units; score() applies the piros ×2."""
    if has_20 and card_pts >= 80 and on_20:
        return ("silent_20_100", gp.twenty_hundred_silent)
    if has_40 and card_pts >= 60 and on_40:
        return ("silent_40_100", gp.forty_hundred_silent)
    return None


def _parti_won(state: GameState) -> bool:
    """Soloist wins parti iff sol's total (trick pts + declared marriages)
    strictly exceeds the defenders' total. The defenders' total INCLUDES the
    talon points — the soloist's 2 discards count for the defenders by rule
    (milan 2026-06-15). `defender_points` already folds the talon in."""
    return soloist_points(state) > defender_points(state)


# ─────────────────────────────────────────────────────────────────────
# Bid set — what was announced before the deal started
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BidSet:
    """What the soloist announced. Silent variants are scored
    automatically by the oracle when not present here."""
    parti:        bool = False     # simple parti (≥ 50 trick pts)
    ulti:         bool = False     # take trick 10 with trump-7
    durchmars:    bool = False     # take all 10 tricks
    forty_hundred: bool = False    # declare trump K+Q + score ≥ 100
    twenty_hundred: bool = False   # declare non-trump K+Q + score ≥ 100
    betli:        bool = False     # take 0 tricks (no trump)
    kontra_level: int  = 0         # 0 = none, 1 = kontra, 2 = rekontra
    piros:        bool = False     # hearts as trump (doubles)
    teritett:     bool = False     # open/spread: soloist plays face-up (defenders
                                   # see the hand) → doubles the DURI/betli
                                   # component: ×2 colored, ×4 colorless
                                   # (terített betli 20, colorless duri 24). milan
                                   # 2026-06-16.


# ─────────────────────────────────────────────────────────────────────
# Per-component GP rates (per defender). Values are TODO — milan to
# confirm against the variant of Ulti you actually play. Marking the
# defaults as placeholders so we don't accidentally ship wrong rates.
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GPTable:
    parti:                int = 1
    ulti_bid:             int = 4
    ulti_bid_bukott:      int = 8   # bukott pays double
    ulti_silent:          int = 2
    ulti_silent_bukott:   int = 4   # silent bukott = 2 × silent rate
    durchmars_bid:        int = 6
    durchmars_silent:     int = 3
    # 100s (confirmed by milan 2026-06-14): 20-100 is worth MORE than 40-100 —
    # it needs 80 of the 90 card points, vs 60 for the 40. silent = half the bid.
    forty_hundred_bid:    int = 4
    forty_hundred_silent: int = 2
    twenty_hundred_bid:   int = 8
    twenty_hundred_silent: int = 4
    # A held marriage with NO 100 scores ONLY its points toward parti — no
    # separate GP bonus.
    silent_forty:         int = 0
    silent_twenty:        int = 0
    betli:                int = 5


# ─────────────────────────────────────────────────────────────────────
# Payoff vector — one entry per component the oracle considered
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PayoffVector:
    """Signed GP the SOLOIST wins, per component, per defender.

    ``components[key]`` = soloist's GP for that component vs BOTH defenders
    (the symmetric/shared case — every colored game). ``def_split[key]`` =
    ``(vs_def1, vs_def2)`` override for the ONLY asymmetric case: a colorless
    game (betli / standalone duri / terített) where the two defenders kontra'd
    differently. Positive ⟹ soloist gained. A component's sign is the same for
    both defenders (kontra only scales magnitude), so ``made`` is well-defined.
    """
    components: Dict[str, int] = field(default_factory=dict)
    def_split: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def made(self, key: str) -> bool:
        return self.components.get(key, 0) > 0

    def gp_vs(self, d: int) -> int:
        return sum(self.def_split[k][d] if k in self.def_split else v
                   for k, v in self.components.items())

    @property
    def total_per_def(self) -> int:
        """Def-1 view of the soloist's total (== each defender when symmetric)."""
        return sum(self.components.values())

    @property
    def total_sol(self) -> int:
        return self.gp_vs(0) + self.gp_vs(1)


# ─────────────────────────────────────────────────────────────────────
# The oracle
# ─────────────────────────────────────────────────────────────────────

def score(
    *,
    final_pos: GameState,
    bid:       BidSet,
    gp:        Optional[GPTable] = None,
    score_parti:       bool = True,
    silents:           bool = True,
    silent_ulti:       Optional[bool] = None,
    silent_durchmars:  Optional[bool] = None,
    silent_40_100:     Optional[bool] = None,
    silent_20_100:     Optional[bool] = None,
    silent_40:         Optional[bool] = None,
    silent_20:         Optional[bool] = None,
    kontras:           Optional[dict] = None,
) -> PayoffVector:
    """Score a played-out deal.

    ``final_pos`` must satisfy ``is_terminal(final_pos)`` — all 10 tricks
    played, or early termination on a binary contract (betli/duri).

    The ``silents`` master flag enables/disables all silent-contract
    components (silent_ulti, silent_durchmars, silent_40, silent_20,
    silent_40_100, silent_20_100). Per-component overrides take
    precedence over the master when set to True or False; pass None
    (the default) to follow the master.
    """
    if gp is None:
        gp = GPTable()
    out = PayoffVector()

    def _on(per_flag: Optional[bool]) -> bool:
        return silents if per_flag is None else per_flag

    on_silent_ulti      = _on(silent_ulti)
    on_silent_dm        = _on(silent_durchmars)
    on_silent_40_100    = _on(silent_40_100)
    on_silent_20_100    = _on(silent_20_100)
    # silent_40 / silent_20 (naked-hold) params are kept for API compatibility
    # but no longer score: a held marriage with no 100 only adds its points to
    # the parti total (milan 2026-06-14). They are intentionally unused.

    points_based = not (bid.durchmars or bid.betli)
    sol_pts      = soloist_points(final_pos)
    has_trump    = (final_pos.trump is not None)

    # ─── silent ulti / bukott — STACKS, never replaces parti ─────────
    # Scored by who played the trump-7 in trick 10 and whether they won it.
    # Bukott (played the 7 and lost) pays double against the player who tried.
    # UNIVERSAL: rides on ANY trump game it wasn't bid in — incl. a durchmars
    # combo where the soloist sweeps and takes trick 10 with the 7 (milan
    # 2026-06-16). Colorless games have no trump-7, so it can't trigger there.
    if has_trump and not bid.ulti and on_silent_ulti:
        side, won = last_trick_ulti_check(final_pos)
        if side == "soloist":
            out.components["silent_ulti"] = (
                +gp.ulti_silent if won else -gp.ulti_silent_bukott)
        elif side == "defender":
            out.components["silent_ulti"] = (
                -gp.ulti_silent if won else +gp.ulti_silent_bukott)
    elif has_trump and bid.ulti:
        # Bid ulti: making it pays ulti_bid; any failure (played 7 and lost, or
        # never played it in trick 10) pays bukott (the contract was promised).
        side, won = last_trick_ulti_check(final_pos)
        if side == "soloist" and won:
            out.components["ulti"] = +gp.ulti_bid
        else:
            out.components["ulti"] = -gp.ulti_bid_bukott

    # ─── bid durchmars / betli (binary contracts) ───────────────────
    # terített (open cards) DOUBLES the duri/betli component: ×4 colorless (no
    # trump), ×2 colored. (piros's ×2 applies to all components at the end, so a
    # colored piros terített duri = 6×2(terit)×2(piros) = 24.) milan 2026-06-16.
    terit = (4 if not has_trump else 2) if bid.teritett else 1
    if bid.durchmars:
        dm = gp.durchmars_bid * terit
        out.components["durchmars"] = (
            +dm if soloist_won_durchmars(final_pos) else -dm)
    if bid.betli:
        be = gp.betli * terit
        out.components["betli"] = (
            +be if soloist_tricks(final_pos) == 0 else -be)

    # ─── bid 100s (declared 40-100 / 20-100) ─────────────────────────
    # card points = the side's trick points EXCLUDING declared marriages; the
    # "100" is ONE marriage + card points ≥ 100 (milan 2026-06-14). Only one
    # marriage may count; 20-100 (needs card≥80) beats 40-100 (card≥60).
    sol_card  = sol_pts - marriage_points(final_pos, final_pos.soloist)
    sol_has40 = has_trump and soloist_has_40(final_pos)
    sol_has20 = soloist_has_20(final_pos)
    bid_a_100 = False
    if bid.forty_hundred and sol_has40:
        bid_a_100 = True
        out.components["40_100"] = (
            +gp.forty_hundred_bid if sol_card >= 60 else -gp.forty_hundred_bid)
    if bid.twenty_hundred and sol_has20:
        bid_a_100 = True
        out.components["20_100"] = (
            +gp.twenty_hundred_bid if sol_card >= 80 else -gp.twenty_hundred_bid)

    # ─── silent points-riders + parti ───────────────────────────────
    # Silent points-riders attach by ACHIEVEMENT minus bid:
    #   • silent durchmars (+3) — a sweep you did NOT bid; stays POINTS-BASED
    #     (betli's 0-tricks isn't a defender duri, and a bid durchmars isn't
    #     silent → both excluded via points_based).
    #   • best single silent-100 (+2/+4) — UNIVERSAL: any game where a side
    #     reaches 100 with an unbid marriage, so it RIDES on a durchmars combo
    #     too (milan 2026-06-16). Only one 100 per game (best marriage).
    # In a points-based parti game a non-empty rider set REPLACES parti; in a
    # durchmars/combo game it just STACKS. Parti (±1) scores only in a
    # points-based game with no bid-100 and no rider.
    silent_pts = {}
    if points_based and on_silent_dm:
        if soloist_won_durchmars(final_pos):
            silent_pts["silent_durchmars"] = +gp.durchmars_silent
        if soloist_tricks(final_pos) == 0:
            silent_pts["def_silent_durchmars"] = -gp.durchmars_silent
    if not bid_a_100:
        best = _best_silent_100(sol_card, sol_has40, sol_has20, gp,
                                on_silent_40_100, on_silent_20_100)
        if best:
            silent_pts[best[0]] = +best[1]
        n = len(final_pos.scores)
        def_card = (defender_points(final_pos)
                    - sum(marriage_points(final_pos, p)
                          for p in range(n) if p != final_pos.soloist))
        dbest = _best_silent_100(def_card,
                                 has_trump and defender_has_40(final_pos),
                                 defender_has_20(final_pos), gp,
                                 on_silent_40_100, on_silent_20_100)
        if dbest:
            silent_pts["def_" + dbest[0]] = -dbest[1]

    if silent_pts:
        out.components.update(silent_pts)       # replaces parti, or rides a bid/combo
    elif points_based and score_parti and not bid_a_100:
        out.components["parti"] = (
            +gp.parti if _parti_won(final_pos) else -gp.parti)

    # ─── multipliers (piros, kontra) — per-component, per-defender ───
    # Each component rides a KONTRA UNIT (see _unit_of). The unit's level scales
    # it PER DEFENDER: colored units are SHARED (együtt sírunk — both defenders
    # the same); colorless units (betli; standalone duri, i.e. no trump) may
    # differ per defender (separate counters). Silent substitutes ride the parti
    # unit, so "kontra parti" doubles them (milan 2026-07-05). Bid-ulti bukott is
    # the (2^level+1)×base special (milan 2026-06-26). Silent ulti rides no unit
    # (flat). piros ×2 on everything.
    #   `kontras` = {unit: level | (def1_level, def2_level)}; a bare int/absent
    #   unit falls back to `bid.kontra_level` for BOTH defenders (simple-game
    #   shorthand). Colorless units are the only place the two defenders diverge.
    kontras = kontras or {}
    pmult = 2 if bid.piros else 1
    ulti_base = gp.ulti_bid                       # = ulti_bid_bukott / 2

    def _lvl(unit: Optional[str], d: int) -> int:
        if unit is None:
            return 0
        kv = kontras.get(unit)
        if kv is not None:
            return kv[d] if isinstance(kv, tuple) else kv
        return bid.kontra_level

    def _scale(key: str, v: int, level: int) -> int:
        if key == "ulti" and v < 0:               # bid-ulti bukott (asymmetric)
            return -(2 ** level + 1) * ulti_base * pmult
        return v * (2 ** level) * pmult

    sym: Dict[str, int] = {}
    split: Dict[str, Tuple[int, int]] = {}
    for k, v in out.components.items():
        unit = _unit_of(k)
        v1 = _scale(k, v, _lvl(unit, 0))
        v2 = _scale(k, v, _lvl(unit, 1))
        sym[k] = v1
        if v1 != v2:                              # colorless separate counters
            split[k] = (v1, v2)
    out.components = sym
    out.def_split = split
    return out
