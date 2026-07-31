"""In-game kontra: offers, AI decisions (exp27 per-unit rules), levels, the oracle kontra dict."""


import os
import random
import sys
import time
import uuid
from typing import List

from ulti.bidding.kontra import _sol_ev
from ulti.scoring.units import UNITS_ORDER as _UNITS_ORDER, \
    UNIT_OBJECTIVE as _UNIT_OBJ, kontra_units as _kontra_units

from .engine import Session, _recipe  # noqa: E402
from . import ai_pool  # noqa: E402


# ── Kontra (simple contracts only) ──────────────────────────────────────────────

# Unit vocabulary, the kontra-able units of a game and how to solve each one all live in
# ulti.scoring.units — the same module the scoring oracle uses, so the kontra we OFFER and
# the kontra we SCORE can never describe different games. Only the display labels are the
# API layer's business.
_UNIT_HU = {"parti": "parti", "ulti": "ulti", "40_100": "40-100", "20_100": "20-100",
            "durchmars": "durchmars", "betli": "betli"}


def _unit_makeability(sess: Session, viewer: int, unit: str, salt: int) -> float:
    """P(soloist makes `unit` | viewer's own hand) — cheat-clean own-hand sampling,
    god-solved for the unit's objective in a WORKER (apps.api.ai_worker does the math)."""
    return ai_pool.run("unit_makeability", {
        "hands0": [[c.id for c in h] for h in sess.play_hands0],
        "talon": [c.id for c in sess.play_talon],
        "trump": sess.trump, "unit": unit, "viewer": viewer,
        "seed": sess.seed + salt,
    })


# exp27 per-unit defender-kontra gates (validated 2026-07-21 — held-out tournament vs
# the old blind-makeability rule: +7.7 GP/deal for the defenders, cheat-clean). The old
# rule `_sol_ev(blind_makeability) < 0` wildly over-kontra'd: it sampled RANDOM soloist
# hands and ignored that the soloist BID the contract, so it "saw" ~6-11% makeability
# when the true make is ~80% → kontra'd makeable ulti/parti and paid double (the soloist's
# rekontra amplified the loss further). Per-unit calibrated signals instead — own hand only.
_KONTRA_ULTI_TRUMPS = 4     # ulti: kontra iff this defender holds >=4 trumps (make ~32%; 3→76%)
_KONTRA_DURI_TRUMPS = 3     # colored durchmars: kontra iff >=3 trumps (make ~2-5%; 0→50%)
_KONTRA_PARTI_MAKE  = 0.10  # parti: kontra iff blind makeability ~0 (far more selective than old rule)


def _ai_defender_kontras_unit(sess: Session, pidx: int, U: str) -> bool:
    """Cheat-clean per-unit kontra from this defender's OWN hand. Trump count is the
    decisive signal for the trick contracts (ulti/duri); parti keeps a makeability
    threshold; the 100-games & betli/colorless-duri abstain (no own-hand signal beats
    not kontra-ing)."""
    own = sess.play_hands0[pidx]
    if U == "ulti":
        return sum(1 for c in own if c.suit == sess.trump) >= _KONTRA_ULTI_TRUMPS
    if U == "durchmars" and sess.trump is not None:
        return sum(1 for c in own if c.suit == sess.trump) >= _KONTRA_DURI_TRUMPS
    if U == "parti":
        return _unit_makeability(sess, pidx, "parti", 100 + pidx) < _KONTRA_PARTI_MAKE
    if U == "40_100" and sess.trump is not None:
        # milan 2026-07-23: a 40-100 declares the TRUMP marriage (the "40"). If a defender
        # holds a card of the trump marriage (K or felső of trump), the soloist can't hold the
        # full trump marriage → the 40-100 is unmakeable → auto-kontra.
        return any(c.suit == sess.trump and c.rank in ("king", "upper") for c in own)
    # 20-100: the 20's colour is NOT declared, so the trump test doesn't apply — the rule is
    # about the NON-trump marriages and is still being pinned down with milan. Abstain for now.
    return False               # 20-100 / betli / colorless durchmars → abstain


def _unit_makeability_post_trick1(sess: Session, unit: str, salt: int) -> float:
    """Post-trick-1 unit makeability (the soloist's rekontra signal) — worker-side."""
    job = _recipe(sess)
    job.update(unit=unit, viewer=0, seed=sess.seed + salt)
    return ai_pool.run("unit_makeability_post1", job)


def _ai_soloist_rekontras_unit(sess: Session, U: str) -> bool:
    # The rekontra comes AFTER trick 1 — decide from what the soloist has now seen.
    p = _unit_makeability_post_trick1(sess, U, 200 + _UNITS_ORDER.index(U))
    return _sol_ev(p, sess.bid, 0) > 0


def _recompute_k_level(sess: Session) -> None:
    lvl = 0
    for U in sess.k_units:
        for pidx in (1, 2):
            if sess.k_def.get(U, {}).get(pidx):
                lvl = max(lvl, 2 if sess.k_rekontra.get(U) else 1)
    sess.k_level = lvl


def _available_units(sess: Session, pidx: int) -> List[str]:
    """Units a defender may still kontra at their decision point. Colored units are
    shared (együtt sírunk) → drop ones ANY defender already kontra'd; colorless keep
    separate per-defender counters → drop only ones THIS defender already kontra'd."""
    out = []
    for U in sess.k_units:
        d = sess.k_def.get(U, {})
        taken = d.get(pidx) if sess.k_colorless else (d.get(1) or d.get(2))
        if not taken:
            out.append(U)
    return out


def _kontra_dict(sess: Session) -> dict:
    """Per-UNIT kontra levels for the oracle. Colored units are SHARED (együtt sírunk —
    both defenders together); colorless (betli / no-trump duri) may differ per defender
    (separate counters)."""
    if not sess.k_units:
        return {}
    out: dict = {}
    for U in sess.k_units:
        d = sess.k_def.get(U, {})
        def lvl(pidx: int) -> int:
            if not d.get(pidx):
                return 0
            return 2 if sess.k_rekontra.get(U) else 1
        d1, d2 = lvl(1), lvl(2)
        if d1 == 0 and d2 == 0:
            continue
        if sess.k_colorless:             # separate counters (def0=pidx1, def1=pidx2)
            out[U] = (d1, d2)
        else:                            # colored → shared
            out[U] = max(d1, d2)
    return out


def _next_kontra_offer(sess: Session):
    """The next kontra decision to offer given trick-1 play so far → (role, pidx,
    available_units) or None. Each defender is offered once, right after playing their
    first card (play-index 1 at ply 1, 2 at ply 2), on the units still open to them;
    the soloist's rekontra comes after trick 1 (>=3 plies) once, on the kontra'd units.
    Defenders with nothing left to kontra are auto-skipped (marks k_off)."""
    if not sess.k_units:
        return None
    plies = len(sess.p_history)
    for pidx in (1, 2):
        if plies > pidx and not sess.k_off[pidx]:
            avail = _available_units(sess, pidx)
            if avail:
                return ("def", pidx, avail)
            sess.k_off[pidx] = True       # nothing left to kontra → auto-skip
    if plies >= 3 and not sess.k_rk_off:
        kontrad = [U for U in sess.k_units
                   if sess.k_def.get(U, {}).get(1) or sess.k_def.get(U, {}).get(2)]
        if kontrad:
            return ("sol", 0, kontrad)
        sess.k_rk_off = True              # nothing was kontra'd → no rekontra
    return None


def _apply_kontra_ai(sess: Session, role: str, pidx: int, avail: List[str]) -> None:
    """A non-human seat's per-unit kontra/rekontra decision (own-hand makeability)."""
    if role == "def":
        sess.k_off[pidx] = True
        hit = [U for U in avail if _ai_defender_kontras_unit(sess, pidx, U)]
        for U in hit:
            sess.k_def[U][pidx] = True
        if hit:
            labels = ", ".join(_UNIT_HU.get(U, U) for U in hit)
            sess.bubbles.append({"player": pidx, "text": f"Kontra! ({labels})", "ply": pidx})
    else:                                # soloist rekontra
        sess.k_rk_off = True
        hit = [U for U in avail if _ai_soloist_rekontras_unit(sess, U)]
        for U in hit:
            sess.k_rekontra[U] = True
        if hit:
            labels = ", ".join(_UNIT_HU.get(U, U) for U in hit)
            sess.bubbles.append({"player": 0, "text": f"Rekontra! ({labels})", "ply": 3})
    _recompute_k_level(sess)


