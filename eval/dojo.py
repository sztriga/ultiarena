"""
Biased deal generator + forced-contract match/arena runner.

Ports trickster's ``UltiDojo.deal`` to oldtawer's ``Card`` model and drives
``UltiGame`` into a forced contract using the dealt cards, so two specialists
can be compared head-to-head on the contract they were trained for.

Adding a new contract
---------------------
The generic engine below — ``deal_biased``, ``run_contract_match``,
``run_contract_arena`` — is parameterised by a ``ContractSpec``. To support
a new contract (say 40-100), define a new spec::

    HUNDRED_SPEC = ContractSpec(
        name='40100',
        bid_action_offset=4,                      # action layout: 0..3 parti
        trump_count_weights={3: 0.10, ...},       # tune to taste
        mandatory_trump_ranks=('K', '10'),        # adjust to engine rules
    )

then call ``run_contract_arena(spec=HUNDRED_SPEC, ...)``. The Ulti wrappers
at the bottom of this file are kept for backwards compatibility with the
existing FastAPI callers.

Differences from trickster's UltiDojo:
- Always seeds the contract-mandatory cards (e.g. trump 7 for Ulti) into
  the declarer's hand. oldtawer's ``Licit.ULTI`` requires it; trickster's
  training-time dojo sometimes generates trump-7-less hands that we'd have
  to throw away.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agents import make_agent
from ulti.card import Card, RANKS, SUITS, fresh_deck
from ulti.game import Phase, UltiGame


# ──────────────────────────────────────────────────────────────────────────────
# Contract spec — the only contract-specific knobs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContractSpec:
    """How to bias a deal toward, and force-bid, a specific Ulti-engine contract.

    bid_action_offset
        UltiGame's licit action layout is::
            0..3   PARTI                 (offset=0)
            4..7   40-100                (offset=4)
            8..11  ULTI                  (offset=8)
            12..15 40-100 + ULTI         (offset=12)
        ``bid_action = offset + trump_idx``.

    trump_count_weights
        Discrete distribution over how many trumps the declarer should hold.

    mandatory_trump_ranks
        Trump ranks that MUST land in the declarer's hand for the contract
        to be legal (e.g. ``('7',)`` for ULTI). The dealer always seeds
        these and excludes them from defenders / talon.
    """
    name:                  str
    bid_action_offset:     int
    trump_count_weights:   Mapping[int, float]
    mandatory_trump_ranks: Tuple[str, ...]


ULTI_SPEC = ContractSpec(
    name='ulti',
    bid_action_offset=8,
    # Same trump-count distribution as parti, but trump-7 is still forced into
    # the soloist's hand. This widens the structural difficulty spread so the
    # α sweep produces a real S-curve in win rate (with the prior 4–8 trumps
    # mode the soloist won ≥78% even at α=0 — too easy for useful datagen).
    trump_count_weights={2: 0.10, 3: 0.40, 4: 0.30, 5: 0.15, 6: 0.05},
    mandatory_trump_ranks=('7',),
)


PARTI_SPEC = ContractSpec(
    name='parti',
    bid_action_offset=0,
    trump_count_weights={2: 0.10, 3: 0.40, 4: 0.30, 5: 0.15, 6: 0.05},
    mandatory_trump_ranks=(),
)


# Durchmars (duri): soloist must win all 10 tricks. Two flavours:
#   * colored  — there's a trump, fat trump suit makes the contract feasible
#   * colorless — no trump, soloist needs an all-around dominating hand;
#     handled by ``deal_durchmars_colorless`` (separate dealer, no trump)
#
# bid_action_offset is set to -1 for now; UltiGame doesn't have a duri
# licit slot yet. These specs exist purely for the solver/dojo bench
# pipeline until the licit layer catches up.
DURI_COLORED_SPEC = ContractSpec(
    name='durchmars_colored',
    bid_action_offset=-1,
    # Many-trump distribution: duri usually requires the soloist to hold
    # a long trump suit. Skewed higher than ulti/parti.
    trump_count_weights={5: 0.15, 6: 0.30, 7: 0.30, 8: 0.15, 9: 0.07, 10: 0.03},
    mandatory_trump_ranks=(),
)

DURI_COLORLESS_SPEC = ContractSpec(
    name='durchmars_colorless',
    bid_action_offset=-1,
    trump_count_weights={},        # unused — colorless dealer ignores this
    mandatory_trump_ranks=(),
)


# ──────────────────────────────────────────────────────────────────────────────
# Deal
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BiasedDeal:
    sol_hand:  List[Card]              # 10 cards (excludes the 2 talon)
    def1_hand: List[Card]              # 10 cards
    def2_hand: List[Card]              # 10 cards
    talon:     List[Card]              # 2 cards
    trump:     Optional[str] = None    # one of SUITS, or None for trumpless contracts (betli)


# Backwards-compat alias for callers that imported the old name.
UltiDeal = BiasedDeal


def _weighted_sample(items, weights, k, rng: random.Random):
    """Efraimidis-Spirakis weighted sampling without replacement."""
    keys = [(rng.random() ** (1.0 / max(w, 1e-9)), i) for i, w in enumerate(weights)]
    keys.sort(reverse=True)
    chosen_idx    = [i for _, i in keys[:k]]
    remaining_idx = [i for _, i in keys[k:]]
    return [items[i] for i in chosen_idx], [items[i] for i in remaining_idx]


def deal_biased(
    *,
    spec:       ContractSpec,
    seed:       int,
    alpha:      float = 0.5,
    suit_sigma: float = 1.0,
) -> BiasedDeal:
    """Generate a contract-biased deal: declarer gets a fat trump suit
    including all ``spec.mandatory_trump_ranks``, weighted toward high cards;
    defenders share the remaining 22."""
    rng = random.Random(seed)
    deck = fresh_deck()
    alpha = rng.uniform(0.0, alpha)

    trump = rng.choice(SUITS)

    counts  = list(spec.trump_count_weights.keys())
    weights = list(spec.trump_count_weights.values())
    n_trump = rng.choices(counts, weights, k=1)[0]

    trump_cards = [c for c in deck if c.suit == trump]
    mandatory   = [c for c in trump_cards if c.rank in spec.mandatory_trump_ranks]
    others      = [c for c in trump_cards if c.rank not in spec.mandatory_trump_ranks]

    n_others      = max(0, min(n_trump - len(mandatory), len(others)))
    other_weights = [math.exp(alpha * c.rank_index) for c in others]
    sol_other_trumps, leftover_trumps = _weighted_sample(
        others, other_weights, n_others, rng,
    )
    sol_trumps = mandatory + sol_other_trumps

    n_rest = 10 - len(sol_trumps)
    non_trump = [c for c in deck if c.suit != trump]
    suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in SUITS}
    suit_mult[trump] = 0.0
    rest_weights = [
        math.exp(alpha * c.rank_index) * suit_mult[c.suit]
        for c in non_trump
    ]
    sol_rest, leftover_non_trump = _weighted_sample(
        non_trump, rest_weights, n_rest, rng,
    )
    sol_hand = sol_trumps + sol_rest

    leftover = leftover_trumps + leftover_non_trump
    rng.shuffle(leftover)
    def1, def2, talon = leftover[:10], leftover[10:20], leftover[20:22]

    for m in mandatory:
        assert m not in def1 and m not in def2 and m not in talon

    return BiasedDeal(
        sol_hand=sol_hand, def1_hand=def1, def2_hand=def2,
        talon=talon, trump=trump,
    )


def deal_ulti_biased(
    *, seed: int, alpha: float = 0.5, suit_sigma: float = 1.0,
) -> BiasedDeal:
    """Backwards-compat: Ulti-biased deal. Equivalent to
    ``deal_biased(spec=ULTI_SPEC, ...)``."""
    return deal_biased(spec=ULTI_SPEC, seed=seed, alpha=alpha, suit_sigma=suit_sigma)


def deal_parti(
    *, seed: int, alpha: float = 0.5, suit_sigma: float = 1.0,
) -> BiasedDeal:
    """Parti-biased deal: declarer gets a moderate trump suit (2–6 trumps)
    biased toward strong cards via standard rank-order. No mandatory ranks
    (parti has no required cards). Equivalent to
    ``deal_biased(spec=PARTI_SPEC, ...)``."""
    return deal_biased(spec=PARTI_SPEC, seed=seed, alpha=alpha, suit_sigma=suit_sigma)


def deal_durchmars_colored(
    *, seed: int, alpha: float = 0.5, suit_sigma: float = 1.0,
) -> BiasedDeal:
    """Colored-duri-biased deal: declarer gets a long, strong trump suit
    (5–10 trumps) plus strong side cards."""
    return deal_biased(spec=DURI_COLORED_SPEC, seed=seed, alpha=alpha, suit_sigma=suit_sigma)


# Colorless-duri rank weighting: 7,8,9,10,J,Q,K,A in *colorless* order
# (10 sits between 9 and J). We weight by ascending strength so the
# soloist's hand clusters around the top.
_DURI_COLORLESS_STRENGTH = {
    '7': 0, '8': 1, '9': 2, '10': 3,
    'lower': 4, 'upper': 5, 'king': 6, 'ace': 7,
}


def deal_durchmars_colorless(
    *,
    seed:       Optional[int] = None,
    alpha:      float = 1.5,
    suit_sigma: float = 0.0,
) -> BiasedDeal:
    """Generate a colorless-durchmars-biased deal.

    Soloist's 10 cards picked by weighted sampling where
    ``weight = exp(alpha * COLORLESS_STRENGTH) * suit_multiplier``.
    Higher alpha → stronger hands (the soloist needs near-dominance).
    No trump. Mirrors ``deal_betli`` but with positive alpha so high
    cards land on the soloist instead of low ones."""
    rng  = random.Random(seed)
    deck = fresh_deck()
    suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in SUITS}
    weights = [
        math.exp(alpha * _DURI_COLORLESS_STRENGTH[c.rank]) * suit_mult[c.suit]
        for c in deck
    ]
    sol_hand, rest = _weighted_sample(deck, weights, 10, rng)
    rng.shuffle(rest)
    return BiasedDeal(
        sol_hand  = sol_hand,
        def1_hand = rest[:10],
        def2_hand = rest[10:20],
        talon     = rest[20:22],
        trump     = None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Betli — different biasing rule (rank-strength, not trump-suit)
# ──────────────────────────────────────────────────────────────────────────────
# Betli has no trump and no mandatory ranks. The biasing weights cards by
# inverse betli-strength so the soloist gets a hand that's plausibly
# winnable for the "take zero tricks" goal.

_BETLI_STRENGTH = {'7': 0, '8': 1, '9': 2, '10': 3, 'lower': 4, 'upper': 5, 'king': 6, 'ace': 7}


def deal_betli(
    *,
    seed:       Optional[int] = None,
    alpha:      float = 1.5,
    suit_sigma: float = 0.3,
) -> BiasedDeal:
    """Generate a Betli-biased face-up deal.

    Soloist's 10 cards are picked by weighted sampling where
    ``weight = exp(-alpha * BETLI_STRENGTH) * suit_multiplier``. Higher
    alpha → weaker hands. ``suit_sigma > 0`` lets one suit dominate the
    soloist's hand (a key Betli pattern). The remaining 22 cards are
    dealt randomly: 10 to each defender, 2 to the talon. ``trump`` is
    ``None`` because betli has no trump.

    Mirrors the original ``solvers/betli/dojo.deal_betli``.
    """
    rng  = random.Random(seed)
    deck = fresh_deck()
    suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in SUITS}
    weights = [
        math.exp(-alpha * _BETLI_STRENGTH[c.rank]) * suit_mult[c.suit]
        for c in deck
    ]
    sol_hand, rest = _weighted_sample(deck, weights, 10, rng)
    rng.shuffle(rest)
    return BiasedDeal(
        sol_hand  = sol_hand,
        def1_hand = rest[:10],
        def2_hand = rest[10:20],
        talon     = rest[20:22],
        trump     = None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Match runner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    seats:               Tuple[str, str, str]   # (soloist, def1, def2) ids
    seed:                int
    declarer_wins:       bool
    payoffs:             Tuple[int, int, int]
    game_points:         Tuple[int, int, int]
    soloist_tricks:      int


# Backwards-compat alias.
UltiMatchResult = MatchResult


def run_contract_match(
    *,
    spec:       ContractSpec,
    soloist_id: str,
    def1_id:    str,
    def2_id:    str,
    seed:       int,
    alpha:      float = 0.5,
    suit_sigma: float = 1.0,
    agent_seed: Optional[int] = None,
) -> MatchResult:
    """Run a single forced-contract game with the given agents."""
    deal = deal_biased(spec=spec, seed=seed, alpha=alpha, suit_sigma=suit_sigma)

    actors = [
        make_agent(soloist_id, agent_seed),
        make_agent(def1_id,    agent_seed),
        make_agent(def2_id,    agent_seed),
    ]

    game = UltiGame()
    game.reset(seed=seed)
    game.hands = [
        list(deal.sol_hand) + list(deal.talon),  # 12 cards (10 + 2 talon)
        list(deal.def1_hand),
        list(deal.def2_hand),
    ]

    trump_idx  = SUITS.index(deal.trump)
    bid_action = spec.bid_action_offset + trump_idx
    if bid_action not in game.legal_actions():
        raise RuntimeError(
            f"Contract {spec.name!r} not legal on this dojo deal "
            f"(trump={deal.trump}); legal={game.legal_actions()}. "
            f"Likely missing a mandatory trump card."
        )
    game.step(bid_action)
    assert game.phase == Phase.DISCARD

    # Discard the dojo's two talon cards back into the talon. If a talon card
    # happens to be protected (mandatory rank — won't happen by construction;
    # trump K/U — possible), fall back to any legal discard.
    for talon_card in deal.talon:
        legal = game.legal_actions()
        action = talon_card.id if talon_card.id in legal else legal[0]
        game.step(action)
    assert game.phase == Phase.PLAY

    while not game.is_over():
        pid = game.current_player_id
        obs = game.get_observation(pid)
        obs["player_id"] = pid
        obs["_game"]     = game            # for MCTS-enabled adapters
        action = int(actors[pid](obs))
        game.step(action)

    payoffs    = tuple(game.payoffs())
    game_pts   = tuple(game.game_points())
    return MatchResult(
        seats=(soloist_id, def1_id, def2_id),
        seed=seed,
        declarer_wins=game.declarer_wins(),
        payoffs=payoffs,
        game_points=game_pts,
        soloist_tricks=sum(1 for w in game.last_trick_winner_per_trick if w == 0)
            if hasattr(game, "last_trick_winner_per_trick")
            else int(game.trick_scores[0] > 0),
    )


def run_ulti_match(
    *,
    soloist_id: str,
    def1_id:    str,
    def2_id:    str,
    seed:       int,
    alpha:      float = 0.5,
    suit_sigma: float = 1.0,
    agent_seed: Optional[int] = None,
) -> MatchResult:
    """Backwards-compat: forced-Ulti single match."""
    return run_contract_match(
        spec=ULTI_SPEC,
        soloist_id=soloist_id, def1_id=def1_id, def2_id=def2_id,
        seed=seed, alpha=alpha, suit_sigma=suit_sigma,
        agent_seed=agent_seed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Arena — forced-contract, three fixed agents, N games split evenly across the
# three soloist seats. The web UI's Arena tab calls this.
# ──────────────────────────────────────────────────────────────────────────────

def run_contract_arena(
    *,
    spec:       ContractSpec,
    agent_ids:  Tuple[str, str, str],
    n_games:    int   = 30,
    seed:       int   = 0,
    alpha:      float = 0.2,
    suit_sigma: float = 1.0,
    agent_seed: Optional[int] = None,
):
    """Three agents, N forced-contract games. Each agent takes the soloist
    seat N/3 times (rounded down); the remaining two sit as def1/def2 in a
    fixed rotation. Defender seating is randomised per game so neither
    agent systematically lands at engine-seat 1 (plays trick 1 second) or
    seat 2 (plays trick 1 last). Returns a dict with config, totals,
    per-soloist stats, and a per-game log."""
    if len(agent_ids) != 3:
        raise ValueError(f"arena needs exactly 3 agents, got {len(agent_ids)}")

    per_seat = max(1, n_games // 3)
    rotation = [
        (agent_ids[0], agent_ids[1], agent_ids[2]),
        (agent_ids[1], agent_ids[2], agent_ids[0]),
        (agent_ids[2], agent_ids[0], agent_ids[1]),
    ]
    seat_rng = random.Random(seed)

    per_soloist: Dict[str, Dict[str, Any]] = {
        aid: {"games": 0, "wins": 0, "payoff_sum": 0.0, "gp_sum": 0.0,
              "tricks_sum": 0}
        for aid in agent_ids
    }
    games_log: List[Dict[str, Any]] = []
    seed_counter = seed
    totals = {"games": 0, "declarer_wins": 0}

    for soloist, defA, defB in rotation:
        for _ in range(per_seat):
            if seat_rng.random() < 0.5:
                d1, d2 = defA, defB
            else:
                d1, d2 = defB, defA
            r = run_contract_match(
                spec=spec,
                soloist_id=soloist, def1_id=d1, def2_id=d2,
                seed=seed_counter, alpha=alpha, suit_sigma=suit_sigma,
                agent_seed=agent_seed,
            )
            seed_counter += 1

            s = per_soloist[soloist]
            s["games"]      += 1
            s["wins"]       += int(r.declarer_wins)
            s["payoff_sum"] += r.payoffs[0]
            s["gp_sum"]     += r.game_points[0]
            s["tricks_sum"] += r.soloist_tricks

            totals["games"]         += 1
            totals["declarer_wins"] += int(r.declarer_wins)

            games_log.append({
                "seed":           r.seed,
                "soloist":        soloist,
                "def1":           d1,
                "def2":           d2,
                "declarer_wins":  r.declarer_wins,
                "payoffs":        list(r.payoffs),
                "game_points":    list(r.game_points),
                "soloist_tricks": r.soloist_tricks,
            })

    for s in per_soloist.values():
        n = s["games"] or 1
        s["win_rate"]   = s["wins"]      / n
        s["avg_payoff"] = s["payoff_sum"] / n
        s["avg_gp"]     = s["gp_sum"]   / n
        s["avg_tricks"] = s["tricks_sum"] / n

    return {
        "config":      {"contract": spec.name, "agents": list(agent_ids),
                        "n_games": totals["games"], "seed": seed,
                        "alpha": alpha, "suit_sigma": suit_sigma},
        "totals":      totals,
        "per_soloist": per_soloist,
        "games":       games_log,
    }


def run_ulti_arena(
    agent_ids:  Tuple[str, str, str],
    *,
    n_games:    int   = 30,
    seed:       int   = 0,
    alpha:      float = 0.2,
    suit_sigma: float = 1.0,
    agent_seed: Optional[int] = None,
):
    """Backwards-compat: forced-Ulti arena. Equivalent to
    ``run_contract_arena(spec=ULTI_SPEC, ...)``."""
    return run_contract_arena(
        spec=ULTI_SPEC,
        agent_ids=agent_ids, n_games=n_games, seed=seed,
        alpha=alpha, suit_sigma=suit_sigma, agent_seed=agent_seed,
    )
