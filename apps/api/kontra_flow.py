"""In-game kontra: offers, AI decisions (exp27 per-unit rules), levels, the oracle kontra dict."""


from typing import List

from ulti.bidding.kontra import _sol_ev, defender_kontras_unit
from ulti.scoring.units import UNITS_ORDER as _UNITS_ORDER

from .engine import Session, _recipe
from . import ai_pool


# ── Kontra (simple contracts only) ──────────────────────────────────────────────

# Unit vocabulary, the kontra-able units of a game and how to solve each one all live in
# ulti.scoring.units — the same module the scoring oracle uses, so the kontra we OFFER and
# the kontra we SCORE can never describe different games. Only the display labels are the
# API layer's business.
_UNIT_HU = {"parti": "parti", "ulti": "ulti", "40_100": "40-100", "20_100": "20-100",
            "durchmars": "durchmars", "betli": "betli"}


# (The pre-play `_unit_makeability` helper lived here. It had exactly one caller — the
# parti kontra rule — and went with it; `ai_worker.op_unit_makeability` remains for the
# research harnesses. The soloist's REKONTRA still uses the post-trick-1 variant below.)


def _ai_defender_kontras_unit(sess: Session, pidx: int, U: str) -> bool:
    """This session's defender applying THE rule (ulti.bidding.kontra) to their own
    hand — the exp27 gates and their evidence live there, with the evaluator that
    grades human kontras against the same function."""
    return defender_kontras_unit(sess.play_hands0[pidx], sess.trump, U)


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


def _kontra_attribution(sess: Session) -> dict:
    """WHO doubled what, by PLAY index — `{unit: {"def": [1,2], "rekontra": bool}}`.

    `_kontra_dict` above is what the ORACLE needs: per-unit levels, with colored units
    collapsed to a single shared number because együtt sírunk means one defender's kontra
    binds both. That collapse loses the attribution, and an evaluation harness needs it —
    without it, "did this player double when they should have" is unanswerable on every
    colored unit (it was 20 of 22 kontra decisions in the first recorded games).

    Only units somebody actually doubled appear; an empty dict means nobody did.
    """
    out: dict = {}
    for U in sess.k_units:
        d = sess.k_def.get(U, {})
        who = [pidx for pidx in (1, 2) if d.get(pidx)]
        if who:
            out[U] = {"def": who, "rekontra": bool(sess.k_rekontra.get(U))}
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


def _apply_kontra_choice(sess: Session, role: str, pidx: int, chosen: List[str]) -> None:
    """Apply a made kontra/rekontra decision — the ONE state mutation + announcement
    bubble, whoever decided (an AI seat below, the human via /play/kontra)."""
    if role == "def":
        sess.k_off[pidx] = True
        for U in chosen:
            sess.k_def[U][pidx] = True
    else:                                # soloist rekontra
        sess.k_rk_off = True
        for U in chosen:
            sess.k_rekontra[U] = True
    if chosen:
        labels = ", ".join(_UNIT_HU.get(U, U) for U in chosen)
        word, player, ply = (("Kontra", pidx, pidx) if role == "def"
                             else ("Rekontra", 0, 3))
        sess.bubbles.append({"player": player, "text": f"{word}! ({labels})", "ply": ply})
    _recompute_k_level(sess)


def _apply_kontra_ai(sess: Session, role: str, pidx: int, avail: List[str]) -> None:
    """A non-human seat's per-unit kontra/rekontra decision (own-hand makeability)."""
    if role == "def":
        hit = [U for U in avail if _ai_defender_kontras_unit(sess, pidx, U)]
    else:
        hit = [U for U in avail if _ai_soloist_rekontras_unit(sess, U)]
    _apply_kontra_choice(sess, role, pidx, hit)


