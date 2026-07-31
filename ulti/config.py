"""Every runtime knob, in one place.

Before this module, 30+ ``os.environ`` reads were scattered across nine files, and the
same knob could have two defaults (bidder.py ships FLOOR=0; the deployed app wants 0.80).
Now: modules read knobs through the typed helpers below — at their own import time, same
as before — and every read is recorded in ``REGISTRY``, so ``snapshot()`` shows the full
live configuration. The deployment profile is one table, applied explicitly.

Two-tier defaults, kept on purpose
----------------------------------
Library modules (ulti.bidding.*) declare NEUTRAL defaults at their read site — importing
the bidder standalone gives the un-tuned research object. The deployed app first calls
:func:`apply_deploy_defaults` (a plain ``os.environ.setdefault``), THEN imports the
bidding stack, which is exactly the ordering apps/api/play.py always had. Explicit env
vars override both tiers, which is how the golden harness pins behaviour.

Knob catalog (name — used by — meaning)
---------------------------------------
FLOOR             bidder    min P(primary) to bid a rung (deploy 0.80, exp30)
DEBIAS_PCTL       auction   argmax-over-66 debias percentile (deploy 0.85, exp20/30)
DURI_TERIT_MULT   bidder    terített-duri EV clamp (deploy 0.3, exp30)
KONTRA            bidder    kontra-aware bidding (deploy on, exp25)
REBETLI_FLOOR     bidder    min realistic P(make) to escalate betli→rebetli (exp39)
BETLI_REAL / REBETLI_REAL   bidder  realistic-defense betli/rebetli heads (exp37/39)
BETLI_REAL_DIR    provider  override dir for the exp37 betli-real head
PIMC_N, KONTRA_NDET         scorers  PIMC sizes for bidding-side evaluations
PLAY_PIMC_N       play      PIMC determinizations per AI card
PLAY_KONTRA_NDET  play      determinizations per in-game kontra decision
EXPLOIT{,_EPS,_NW,_FRAC}    play  exp31 safe-exploit soloist
BETLI_REAL_BID, REBETLI_REAL_BID, BETLI_DEF, MIX_EQUIV   play  promoted-feature gates
BETLI_DEF_MODEL   betli     path override for the exp36 defense net
GAMES_DB          recording SQLite path for finished-game recording
PUZZLE_SECONDS, PUZZLE_QUEUE   puzzle  Villámtalon duration / precompute depth

(Training-only scripts — ulti/bidding/base_head.py — keep their own epoch/lr knobs;
they are not part of the serving surface.)
"""
from __future__ import annotations

import os
from typing import Dict, Optional

# Every knob read through the helpers lands here: name -> (raw or None, parsed, default).
REGISTRY: Dict[str, tuple] = {}

_TRUE = ("1", "true", "yes")


def _record(name: str, parsed, default):
    REGISTRY[name] = (os.environ.get(name), parsed, default)
    return parsed


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    return _record(name, os.environ.get(name, default), default)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    val = default if raw is None else raw.strip().lower() in _TRUE
    return _record(name, val, default)


def env_int(name: str, default: int) -> int:
    return _record(name, int(os.environ.get(name, default)), default)


def env_float(name: str, default: float) -> float:
    return _record(name, float(os.environ.get(name, default)), default)


# ── The deployment profile ──────────────────────────────────────────────────────
# apps/api applies this BEFORE importing the bidding stack (setdefault → explicit env
# still wins). One table instead of four bare setdefault lines buried in play.py.
DEPLOY_DEFAULTS = {
    "FLOOR": "0.80",           # exp30: suppress overconfident escalations (was 0.70)
    "DEBIAS_PCTL": "0.85",     # exp30 re-tune of the exp20 debias (was 0.80)
    "DURI_TERIT_MULT": "0.3",  # exp30: fixes the terített-duri over-bid leak
    "KONTRA": "1",             # exp25: kontra-aware bidder (passes weak hands)
}


def apply_deploy_defaults() -> None:
    for k, v in DEPLOY_DEFAULTS.items():
        os.environ.setdefault(k, v)


def snapshot() -> Dict[str, object]:
    """Live value of every knob that has been read so far (for logs / debugging)."""
    return {name: parsed for name, (_raw, parsed, _d) in sorted(REGISTRY.items())}
