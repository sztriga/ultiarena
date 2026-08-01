"""
Villámtalon — a talon-picking puzzle-rush mini-game.

The player gets 12 cards and an AI-chosen contract ("Legerősebb ulti — adu: piros").
They bury 2 cards; the value net scores the kept 10 for that contract. Points reward
how close the chosen talon is to the OPTIMAL talon (best of all 66 discards). 3-minute
timer, solve as many as you can, combo multiplier for streaks, difficulty rises with score.

Everything here is NET-ONLY (no PIMC, no solver) → puzzle generation + scoring are
milliseconds. To keep the player's experience instant we PRE-GENERATE a queue of puzzles
in a per-session background thread (the 66-discard ranking is computed at generation time,
so scoring a submitted answer is an O(1) lookup — no net call on the request path).

Reuses the SAME loaded net singleton as the play tab (apps/api/play._provider) — never
loads a second copy.
"""
from __future__ import annotations

import itertools
import random
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from threading import RLock, Thread
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

# play.py has already put the exp dirs on sys.path; import it first so its side effects run,
# then reuse its net singleton + Hungarian suit map.
from .play import _provider, _SUIT_HU  # noqa: E402
from ulti.card import sort_hand         # noqa: E402
from ulti.config import env_int         # noqa: E402
from .serialize import card_to_dict     # noqa: E402
from .limits import guard_new_session   # noqa: E402

from ulti.vnet.pickup import featurize        # noqa: E402
from ulti.eval.dojo import (                  # noqa: E402
    deal_ulti_biased, deal_parti, deal_durchmars_colored,
    deal_durchmars_colorless, deal_betli,
)

router = APIRouter()

DURATION = env_int("PUZZLE_SECONDS", 180)       # 3-minute rush
QUEUE_TARGET = env_int("PUZZLE_QUEUE", 5)       # puzzles kept ready ahead
_MAX_GEN_TRIES = 40                                          # deals to try before relaxing the gate

def _deal_parti_piros(*, seed, alpha=0.5, suit_sigma=1.0):
    """Parti is only ever played in PIROS (hearts) — milan's rule (plain non-piros parti is
    not biddable). deal_biased picks the trump ~uniformly, so re-deal until it's hearts
    (an unbiased subsample — trump choice is independent of hand strength)."""
    r = random.Random(seed)
    d = None
    for _ in range(40):
        d = deal_parti(seed=r.getrandbits(31), alpha=alpha, suit_sigma=suit_sigma)
        if d.trump == "hearts":
            return d
    return d


# ── Contract catalogue ───────────────────────────────────────────────────────────
# head = provider head to score with; colored = 36-dim colored featurize else 32-dim colorless;
# dealer(seed, alpha) → BiasedDeal; label = Hungarian prompt word.
_CONTRACTS = {
    "parti":   dict(head="parti",          colored=True,  dealer=_deal_parti_piros,        label="parti"),
    "ulti":    dict(head="ulti",           colored=True,  dealer=deal_ulti_biased,         label="ulti"),
    "duri":    dict(head="duri_colored",   colored=True,  dealer=deal_durchmars_colored,   label="duri"),
    "szintelen_duri": dict(head="colorless_duri", colored=False, dealer=deal_durchmars_colorless,
                           label="színtelen duri"),
    "betli":   dict(head="betli",          colored=False, dealer=deal_betli,               label="betli"),
}


def _difficulty(score: int) -> float:
    """0.0 (start) → 1.0 (hardest), reached ~2500 pts."""
    return min(1.0, max(0.0, score / 2500.0))


def _contract_pool(d: float) -> List[str]:
    """Widen the contract set as difficulty rises (parti/ulti are easiest to read)."""
    pool = ["parti", "ulti"]
    if d >= 0.20:
        pool.append("duri")
    if d >= 0.40:
        pool += ["szintelen_duri", "betli"]
    return pool


def _target_band(d: float):
    """The optimal-talon strength window we want. Easy = strong/obvious; hard = weaker/marginal."""
    hi = 0.97 - 0.20 * d                       # 0.97 → 0.77
    lo = 0.80 - 0.35 * d                       # 0.80 → 0.45
    return lo, hi


def _alpha(contract: str, d: float) -> float:
    """Dealer strength knob. Lower with difficulty so weaker deals become common."""
    base = {"parti": 1.2, "ulti": 1.3, "duri": 1.4, "szintelen_duri": 2.2, "betli": 2.2}[contract]
    return base * (1.0 - 0.45 * d)


_MIN_SPREAD = 0.12                              # best−worst must exceed this, else the choice is a dud


# ── Puzzle generation & scoring (net-only) ────────────────────────────────────────

_net_lock = RLock()                             # serialize net inference across session filler threads


def _score_discards(cards12: List, cfg: dict, trump: Optional[str]):
    """For every 2-card discard, the calibrated net probability of the kept 10-hand.
    Returns {frozenset(discard_ids): p} plus the sorted (p, discard_ids) list, best-first."""
    prov = _provider()
    head, colored = cfg["head"], cfg["colored"]
    out: Dict[frozenset, float] = {}
    ranked = []
    with _net_lock:
        for i, j in itertools.combinations(range(12), 2):
            keep = [cards12[k] for k in range(12) if k != i and k != j]
            x = featurize(keep, trump if colored else None, colored)
            p = prov._p(head, x)
            key = frozenset((cards12[i].id, cards12[j].id))
            out[key] = p
            ranked.append((p, (cards12[i].id, cards12[j].id)))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return out, ranked


class Puzzle:
    __slots__ = ("id", "contract", "trump", "cards12", "pmap", "ranked",
                 "best_p", "worst_p", "optimal_ids", "difficulty")

    def __init__(self, contract, trump, cards12, pmap, ranked, difficulty):
        self.id = uuid.uuid4().hex[:8]
        self.contract = contract
        self.trump = trump
        self.cards12 = cards12
        self.pmap = pmap
        self.ranked = ranked
        self.best_p = ranked[0][0]
        self.worst_p = ranked[-1][0]
        self.optimal_ids = list(ranked[0][1])
        self.difficulty = difficulty

    def prompt(self) -> str:
        lbl = _CONTRACTS[self.contract]["label"]
        if self.trump is not None:
            return f"Legerősebb {lbl} — adu: {_SUIT_HU.get(self.trump, self.trump)}"
        return f"Legerősebb {lbl}"

    def public(self) -> dict:
        return {
            "puzzle_id": self.id,
            "contract": self.contract,
            "contract_label": _CONTRACTS[self.contract]["label"],
            "trump": self.trump,
            "prompt": self.prompt(),
            "difficulty": round(self.difficulty, 2),
            # Display only — all logic is card-id keyed, so order never affects scoring.
            # Same ordering rule as the play table (ulti.card.sort_hand), including the
            # 10-low reading for the colourless contracts (betli / színtelen duri).
            "cards": [card_to_dict(c) for c in
                      sort_hand(self.cards12, not _CONTRACTS[self.contract]["colored"])],
        }


def _generate(score: int, rng: random.Random) -> Puzzle:
    """Deal-until-good-enough at the current difficulty. Falls back to the best attempt so it
    never blocks forever (the quality gate is a preference, not a hard requirement)."""
    d = _difficulty(score)
    lo, hi = _target_band(d)
    contract = rng.choice(_contract_pool(d))   # fix the contract first → balanced variety
    cfg = _CONTRACTS[contract]
    best_attempt = None
    for _ in range(_MAX_GEN_TRIES):
        deal = cfg["dealer"](seed=rng.getrandbits(31), alpha=_alpha(contract, d))
        cards12 = list(deal.sol_hand) + list(deal.talon)
        pmap, ranked = _score_discards(cards12, cfg, deal.trump)
        pz = Puzzle(contract, deal.trump, cards12, pmap, ranked, d)
        spread = pz.best_p - pz.worst_p
        if lo <= pz.best_p <= hi and spread >= _MIN_SPREAD:
            return pz
        # keep the attempt with the widest spread in-band as a fallback
        score_key = spread + (0.0 if lo <= pz.best_p <= hi else -1.0)
        if best_attempt is None or score_key > best_attempt[0]:
            best_attempt = (score_key, pz)
    return best_attempt[1]


# ── Session ───────────────────────────────────────────────────────────────────────

COMBO_THRESH = 0.85        # quality at/above this extends the combo
COMBO_STEP = 0.25          # multiplier grows by this per combo step
COMBO_CAP = 8              # multiplier caps at 1 + STEP*CAP = 3.0×
BASE_POINTS = 100


class PuzzleSession:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.solved = 0
        self.start = time.monotonic()
        self.rng = random.Random()
        self.queue: deque = deque()
        self.current: Optional[Puzzle] = None
        self.last_result: Optional[dict] = None
        self.lock = RLock()
        self._running = True
        self._filler = Thread(target=self._fill_loop, daemon=True)
        self._filler.start()

    # -- timing --
    def time_left(self) -> float:
        return max(0.0, DURATION - (time.monotonic() - self.start))

    def expired(self) -> bool:
        return self.time_left() <= 0.0

    # -- background pre-generation --
    def _fill_loop(self):
        while self._running:
            try:
                if self.expired():
                    break
                if len(self.queue) < QUEUE_TARGET:
                    pz = _generate(self.score, self.rng)
                    with self.lock:
                        self.queue.append(pz)
                else:
                    time.sleep(0.02)
            except Exception:
                time.sleep(0.05)      # never let the filler die on a transient error

    def _next_puzzle(self) -> Optional[Puzzle]:
        with self.lock:
            if self.queue:
                return self.queue.popleft()
        # queue behind (rare) → generate synchronously so the player never stalls
        return _generate(self.score, self.rng)

    def start_first(self):
        self.current = self._next_puzzle()

    def stop(self):
        self._running = False

    # -- solve --
    def solve(self, discard_ids: List[int]) -> dict:
        with self.lock:
            pz = self.current
            if pz is None:
                raise HTTPException(status_code=400, detail="no active puzzle")
            key = frozenset(discard_ids)
            if key not in pz.pmap:
                raise HTTPException(status_code=400, detail="pick exactly 2 of the 12 cards")
            counted = not self.expired()
            p_player = pz.pmap[key]
            denom = max(pz.best_p - pz.worst_p, 1e-6)
            quality = min(1.0, max(0.0, (p_player - pz.worst_p) / denom))
            points = 0
            if counted:
                if quality >= COMBO_THRESH:
                    self.combo += 1
                else:
                    self.combo = 0
                self.best_combo = max(self.best_combo, self.combo)
                mult = 1.0 + COMBO_STEP * min(self.combo, COMBO_CAP)
                diff_mult = 1.0 + 0.5 * pz.difficulty
                points = round(BASE_POINTS * quality * mult * diff_mult)
                self.score += points
                self.solved += 1
            self.last_result = {
                "points": points,
                "quality": round(quality, 3),
                "your_p": round(p_player, 3),
                "best_p": round(pz.best_p, 3),
                "optimal_ids": pz.optimal_ids,
                "your_ids": list(discard_ids),
                "perfect": discard_ids and frozenset(discard_ids) == frozenset(pz.optimal_ids),
                "combo": self.combo,
                "counted": counted,
            }
            # advance
            self.current = None if self.expired() else self._next_puzzle()
        return self.snapshot()

    def snapshot(self) -> dict:
        done = self.expired() or self.current is None
        return {
            "game_id": self.id,
            "score": self.score,
            "combo": self.combo,
            "best_combo": self.best_combo,
            "solved": self.solved,
            "time_left": round(self.time_left(), 1),
            "duration": DURATION,
            "done": done,
            "last_result": self.last_result,
            "puzzle": None if done else self.current.public(),
        }


_sessions: Dict[str, PuzzleSession] = {}
_sessions_lock = RLock()


def _get(game_id: str) -> PuzzleSession:
    with _sessions_lock:
        s = _sessions.get(game_id)
    if s is None:
        raise HTTPException(status_code=404, detail=f"unknown puzzle game {game_id}")
    return s


# ── Endpoints ───────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    game_id: str
    discard_ids: List[int] = Field(..., min_length=2, max_length=2)


@router.post("/puzzle/new")
def puzzle_new(request: Request = None) -> dict:
    # prune finished/stale sessions (keep memory bounded), then cap: each puzzle
    # session owns a background filler THREAD, so unbounded sessions = unbounded threads.
    with _sessions_lock:
        for gid in [g for g, s in _sessions.items() if s.expired()]:
            _sessions[gid].stop(); _sessions.pop(gid, None)
        ip = guard_new_session(request, _sessions, lambda s: getattr(s, "owner_ip", None),
                               on_evict=lambda s: s.stop())
    sess = PuzzleSession()
    sess.owner_ip = ip
    sess.start_first()                 # blocks on first-ever net load; then the queue is warm
    with _sessions_lock:
        _sessions[sess.id] = sess
    return sess.snapshot()


@router.post("/puzzle/solve")
def puzzle_solve(req: SolveRequest) -> dict:
    return _get(req.game_id).solve(req.discard_ids)


@router.get("/puzzle/state/{game_id}")
def puzzle_state(game_id: str) -> dict:
    return _get(game_id).snapshot()


@router.post("/puzzle/end/{game_id}")
def puzzle_end(game_id: str) -> dict:
    s = _get(game_id)
    s.stop()
    snap = s.snapshot()
    with _sessions_lock:
        _sessions.pop(game_id, None)
    return snap
