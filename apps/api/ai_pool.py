"""The AI worker pool — how many tables can think at once.

The Cython solver holds the GIL for a whole search and keeps process-global state
(EV_MULTI weights + the transposition table), so ONE process can only ever run one
search, no matter how it is locked. Every AI decision therefore goes through here:

  AI_WORKERS > 0 (default)  a spawn-based ProcessPoolExecutor; each worker owns its
                            own solver globals → N tables solve on N cores, and one
                            session's weight vector can never trample another's.
  AI_WORKERS = 0            run inline in this process under a lock — the exact
                            pre-pool behaviour. Debug/CI escape hatch.

Determinism: a job carries everything (deal, history, weights, seed) and the worker
runs the same code either way, so pool and inline produce byte-identical decisions —
the golden harness passes in both modes.

The main process stays SOLVER-FREE for the play flow (its position objects are used
only for legality/serialization); the nets (bidder, betli defense) stay in the main
process — they are microsecond forward passes and workers must stay torch-free.

solver_lock: modules that still solve in-process (the stateless /pis/* explore
endpoints) must hold it around set_multi_weights + solve so two threadpool requests
cannot interleave weight-setting and searching. (That race predates the pool.)
"""
from __future__ import annotations

import atexit
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from threading import RLock
from typing import Any

from ulti.config import env_int

from . import ai_worker

# Worker count: leave headroom for the web process + the torch nets. 0 = inline.
AI_WORKERS = env_int("AI_WORKERS", max(1, min(4, (os.cpu_count() or 4) - 2)))

solver_lock = RLock()          # guards ALL in-process solver use (inline mode, /pis/*)

_pool: ProcessPoolExecutor | None = None
_pool_lock = RLock()


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(
                max_workers=AI_WORKERS,
                mp_context=get_context("spawn"),   # no fork: torch/openmp in the parent
            )
            atexit.register(_pool.shutdown, wait=False, cancel_futures=True)
        return _pool


def run(op: str, job: dict) -> Any:
    """Execute one AI job. Blocks until the result is back — callers are already
    request-scoped threads, and the whole point is that OTHER requests' jobs run
    concurrently on other workers."""
    if AI_WORKERS <= 0:
        with solver_lock:
            return ai_worker.dispatch(op, job)
    return _get_pool().submit(ai_worker.dispatch, op, job).result()
