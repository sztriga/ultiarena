"""Contract registry for the multi-contract V-net pipeline.

One ``ContractConfig`` per supported contract. Each entry plugs the contract
into the shared data_gen / features / train / alpha_sweep flow.

Adding a new contract:
    1. Implement its dealer in ``eval.dojo`` (returns ``BiasedDeal``).
    2. Confirm the trickster solver handles ``contract=<name>`` correctly.
    3. Add a ``_make_<name>()`` factory below and register it in ``REGISTRY``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ulti.card import Card


# ──────────────────────────────────────────────────────────────────────────────
# Strength functions (used for per-suit "top card" feature summaries)
# ──────────────────────────────────────────────────────────────────────────────
#
# These mirror the solver-side strength orders in
# ``ultisolver._solver_core._str_b`` (betli) and ``_str_n`` (everything else).
# Returning a 0..7 int so the feature normalisation (``/ 7.0``) is contract-
# independent.

_BETLI_STRENGTH = {
    "7": 0, "8": 1, "9": 2, "10": 3,
    "lower": 4, "upper": 5, "king": 6, "ace": 7,
}


def betli_strength(c: Card) -> int:
    """Betli order: 7 < 8 < 9 < 10 < J < Q < K < A (10 is weak)."""
    return _BETLI_STRENGTH[c.rank]


def parti_strength(c: Card) -> int:
    """Standard ulti order: 7 < 8 < 9 < J < Q < K < 10 < A (10 strong).
    Card.rank_index already follows this convention."""
    return c.rank_index


# ──────────────────────────────────────────────────────────────────────────────
# ContractConfig
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContractConfig:
    """All the contract-specific knobs the V-net pipeline needs.

    name
        Short identifier; appears in filenames, registry keys, log lines.
    solver_name
        Contract string passed to ``pis.solve_all(contract=...)``.
    has_trump
        Whether the contract has a trump suit. Drives the trump-tail feature
        bits (4 + 8 + 8 = 20 extra dims) and the ``trump`` kwarg into
        ``build_position``.
    deal_fn
        Dealer; ``deal_fn(seed=..., alpha=...)`` → ``BiasedDeal``.
    strength_fn
        ``Card → int 0..7``, used for ``my_top_per_suit`` and
        ``live_top_per_suit`` feature summaries. Must match the solver's
        per-contract strength order.
    target_kind
        ``"binary"`` (label in {-1, +1}, soft if K-averaged) or
        ``"regression"`` (label is raw scalar, e.g. parti card-points).
    label_fn
        ``(solver_max_value: float, ...) → float``. Converts the solver's
        best-value output into a single scalar label per world. For betli
        this thresholds against ``WIN_VAL``; for parti it's the identity.
    feature_dim
        Total feature vector length. Base layout = 132 dims; contracts with
        trump add 20 dims of trump-tail features → 152.
    """
    name:           str
    solver_name:    str
    has_trump:      bool
    deal_fn:        Callable
    strength_fn:    Callable
    target_kind:    str
    label_fn:       Callable
    feature_dim:    int

    @property
    def trump_tail_dims(self) -> int:
        return 20 if self.has_trump else 0

    @property
    def base_feature_dim(self) -> int:
        return self.feature_dim - self.trump_tail_dims


# ──────────────────────────────────────────────────────────────────────────────
# Label functions
# ──────────────────────────────────────────────────────────────────────────────
#
# Each label_fn takes the solver's *best* soloist-perspective value
# (``max(solve_all(pos).values())``) and returns a single scalar label.
# The data_gen pipeline averages these across K worlds for soft labels.

_BETLI_WIN_VAL = 1.0


def _betli_label(solver_max: float) -> float:
    """Defender-perspective binary label: +1 def wins, -1 sol wins."""
    return -1.0 if solver_max >= _BETLI_WIN_VAL - 1e-6 else +1.0


_PARTI_WIN_VAL = 1.0


def _parti_label(solver_max: float) -> float:
    """Defender-perspective binary label: +1 def wins, -1 sol wins.

    Parti now uses the uniform binary 0/10 solver scale (10 = sol won,
    0 = sol lost), same shape as betli and ulti.
    """
    return -1.0 if solver_max >= _PARTI_WIN_VAL - 1e-6 else +1.0


_ULTI_WIN_VAL = 1.0


def _ulti_label(solver_max: float) -> float:
    """Defender-perspective binary label: +1 def wins, -1 sol wins.

    Pure-ulti solver returns C_ULTI_WIN (1.0) iff the soloist took the
    last trick with the trump-7, C_ULTI_LOSE (0.0) otherwise. Same shape
    as the betli labeller: threshold against the win value, return the
    defender's POV as ±1.
    """
    return -1.0 if solver_max >= _ULTI_WIN_VAL - 1e-6 else +1.0


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_BASE_FEATURE_DIM = 132   # see vnet.features for the layout


def _make_betli() -> ContractConfig:
    from eval.dojo import deal_betli
    return ContractConfig(
        name         = "betli",
        solver_name  = "betli",
        has_trump    = False,
        deal_fn      = deal_betli,
        strength_fn  = betli_strength,
        target_kind  = "binary",
        label_fn     = _betli_label,
        feature_dim  = _BASE_FEATURE_DIM,
    )


def _make_parti() -> ContractConfig:
    from eval.dojo import deal_parti
    return ContractConfig(
        name         = "parti",
        solver_name  = "parti",
        has_trump    = True,
        deal_fn      = deal_parti,
        strength_fn  = parti_strength,
        target_kind  = "binary",
        label_fn     = _parti_label,
        feature_dim  = _BASE_FEATURE_DIM + 20,
    )


def _make_ulti() -> ContractConfig:
    """Announced Ulti: binary contract — soloist must take the last trick
    with the trump 7.

    Modular per-contract solver. ``solvers/pis.py`` sets ``has_ulti=True``
    for this solver name, which activates 7esre tartás in the Cython
    ``_legal()`` (soloist may not voluntarily play the trump 7 before the
    last trick). A later bidding-time evaluator composes V_ulti with
    V_parti and other fundamental contract values.
    """
    from eval.dojo import deal_ulti_biased
    return ContractConfig(
        name         = "ulti",
        solver_name  = "ulti",
        has_trump    = True,
        deal_fn      = deal_ulti_biased,
        strength_fn  = parti_strength,
        target_kind  = "binary",
        label_fn     = _ulti_label,
        feature_dim  = _BASE_FEATURE_DIM + 20,
    )


REGISTRY: Dict[str, ContractConfig] = {
    "betli": _make_betli(),
    "parti": _make_parti(),
    "ulti":  _make_ulti(),
}


def get(name: str) -> ContractConfig:
    if name not in REGISTRY:
        raise KeyError(f"unknown contract {name!r}; expected one of "
                       f"{sorted(REGISTRY)}")
    return REGISTRY[name]
