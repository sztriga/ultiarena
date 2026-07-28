"""Betli data generation: biased dealing + solver-labeled samples.

Generates (features, label) pairs for supervised training of BetliNet.
Each sample is a 10-card betli hand encoded via the betli encoder,
labeled with the solver-derived win probability from PIMC.

Also provides ``discard_candidates`` for smart 12→10 discard selection.
"""
from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from trickster.betli.encoder import encode_hand
from trickster.games.ulti.adapter import UltiGame, UltiNode
from trickster.games.ulti.cards import (
    ALL_SUITS,
    BETLI_STRENGTH,
    Card,
    Suit,
    make_deck,
)
from trickster.games.ulti.game import (
    GameState,
    declare_all_marriages,
    is_terminal,
    next_player,
    set_contract,
)

try:
    from trickster._solver_core import solve_root as _solve_root
except ImportError:
    from trickster.games.ulti.solver import solve_root as _solve_root


# ---------------------------------------------------------------------------
#  Discard candidate generation
# ---------------------------------------------------------------------------

def discard_candidates(hand12: list[Card]) -> list[tuple[Card, Card]]:
    """Return valid discard pairs: top card per suit, or top 2 from same suit.

    Candidates are pairs of the highest betli-strength cards. Either:
    - Top card from two different suits (cross-suit), or
    - Top two cards from the same suit (same-suit strip)

    Returns at most C(4,2) + 4 = 10 pairs.
    """
    # Top 2 cards per suit, sorted by betli strength descending
    by_suit: dict[Suit, list[Card]] = {s: [] for s in ALL_SUITS}
    for c in hand12:
        by_suit[c.suit].append(c)
    for s in ALL_SUITS:
        by_suit[s].sort(key=lambda c: BETLI_STRENGTH[c.rank], reverse=True)

    # Top card per suit (for cross-suit pairs)
    tops: list[Card] = [cards[0] for cards in by_suit.values() if cards]

    pairs: list[tuple[Card, Card]] = []

    # Cross-suit: top from suit A + top from suit B
    for c1, c2 in combinations(tops, 2):
        pairs.append((c1, c2))

    # Same-suit: top 2 from the same suit
    for cards in by_suit.values():
        if len(cards) >= 2:
            pairs.append((cards[0], cards[1]))

    return pairs


# ---------------------------------------------------------------------------
#  Biased dealing (reuses BetliDojo logic)
# ---------------------------------------------------------------------------

def _weighted_sample(
    deck: list[Card],
    weights: list[float],
    n: int,
    rng: random.Random,
) -> tuple[list[Card], list[Card]]:
    """Pick *n* cards from *deck* using *weights*, return (chosen, rest)."""
    remaining = list(range(len(deck)))
    remaining_weights = list(weights)
    chosen_indices: list[int] = []

    for _ in range(n):
        total = sum(remaining_weights)
        r = rng.random() * total
        cumulative = 0.0
        pick = 0
        for i, w in enumerate(remaining_weights):
            cumulative += w
            if cumulative >= r:
                pick = i
                break
        chosen_indices.append(remaining[pick])
        remaining.pop(pick)
        remaining_weights.pop(pick)

    chosen = [deck[i] for i in chosen_indices]
    rest = [deck[i] for i in remaining]
    return chosen, rest


def deal_betli(
    rng: random.Random,
    alpha: float = 1.5,
) -> tuple[list[list[Card]], list[Card]]:
    """Deal a betli-biased hand for soloist (index 0).

    Alpha controls bias strength: 0 = uniform, higher = more low cards.
    Alpha is randomized per deal: Uniform(0, alpha) for quality spectrum.

    Returns (hands[3], talon[2]).
    """
    import math

    deck = make_deck()
    a = rng.uniform(0, alpha)
    suit_sigma = 1.0
    suit_mult = {s: math.exp(rng.gauss(0, suit_sigma)) for s in ALL_SUITS}
    weights = [
        math.exp(-a * BETLI_STRENGTH[c.rank]) * suit_mult[c.suit]
        for c in deck
    ]
    sol_hand, rest = _weighted_sample(deck, weights, 10, rng)
    rng.shuffle(rest)
    def1 = rest[:10]
    def2 = rest[10:20]
    talon = rest[20:22]
    return [sol_hand, def1, def2], talon


# ---------------------------------------------------------------------------
#  Game state setup for solver
# ---------------------------------------------------------------------------

def _make_betli_state(
    hands: list[list[Card]],
    soloist: int = 0,
    dealer: int = 2,
) -> GameState:
    """Create a betli game state ready for play."""
    gs = GameState(
        hands=hands,
        trump=None,
        betli=False,
        soloist=soloist,
        dealer=dealer,
        captured=[[], [], []],
        scores=[0, 0, 0],
        leader=next_player(dealer),
        trick_no=0,
        trick_cards=[],
        last_trick=None,
    )
    set_contract(gs, soloist, trump=None, betli=True)
    declare_all_marriages(gs, soloist_marriage_restrict=None)
    return gs


def _make_betli_node(gs: GameState, dealer: int = 2) -> UltiNode:
    """Wrap a betli GameState into an UltiNode."""
    return UltiNode(
        gs=gs,
        known_voids=(frozenset(), frozenset(), frozenset()),
        bid_rank=0,
        is_red=False,
        contract_components=frozenset({"betli"}),
        dealer=dealer,
    )


# ---------------------------------------------------------------------------
#  Solver-based labeling
# ---------------------------------------------------------------------------

def solver_label(
    hands: list[list[Card]],
    game: UltiGame,
    n_dets: int = 30,
    rng: random.Random | None = None,
) -> float:
    """Label a betli hand using PIMC solver.

    Plays solver-vs-solver from multiple determinizations.
    Returns win fraction [0, 1] for the soloist.
    """
    if rng is None:
        rng = random.Random()

    soloist = 0
    dealer = 2
    gs = _make_betli_state([list(h) for h in hands], soloist, dealer)
    node = _make_betli_node(gs, dealer)

    wins = 0
    for _ in range(n_dets):
        # Determinize from soloist's view for fair sampling
        det = game.determinize(node, soloist, rng)
        # Play out using solver from root position
        det_gs = det.gs
        while not is_terminal(det_gs):
            vals = _solve_root(det_gs, contract="betli")
            if not vals:
                break
            player = det_gs.leader if not det_gs.trick_cards else (
                (det_gs.trick_cards[-1][0] + 1) % 3
                if len(det_gs.trick_cards) < 3
                else det_gs.leader
            )
            # Pick best move for current player
            from trickster.games.ulti.game import current_player as _cp, legal_actions as _la, play_card
            cp = _cp(det_gs)
            if cp == soloist:
                action = max(vals, key=vals.__getitem__)
            else:
                action = min(vals, key=vals.__getitem__)
            play_card(det_gs, action)

        # Check outcome
        from trickster.games.ulti.game import soloist_lost_betli
        if not soloist_lost_betli(det_gs):
            wins += 1

    return wins / max(n_dets, 1)


def solver_label_fast(
    hands: list[list[Card]],
    n_dets: int = 30,
    rng: random.Random | None = None,
    game: UltiGame | None = None,
) -> float:
    """Label a betli hand using single-position PIMC solver.

    Instead of playing out full games, evaluates the root position
    from multiple determinizations. Faster than full game simulation.
    Returns win fraction [0, 1] for the soloist.
    """
    if rng is None:
        rng = random.Random()
    if game is None:
        game = UltiGame(restrictions=[])

    soloist = 0
    dealer = 2
    gs = _make_betli_state([list(h) for h in hands], soloist, dealer)
    node = _make_betli_node(gs, dealer)

    wins = 0
    for _ in range(n_dets):
        det = game.determinize(node, soloist, rng)
        vals = _solve_root(det.gs, contract="betli")
        if vals:
            # Soloist picks max, get the best value
            best = max(vals.values())
            # Betli solver: 10 = soloist wins, 0 = soloist loses
            if best >= 5.0:
                wins += 1

    return wins / max(n_dets, 1)


# ---------------------------------------------------------------------------
#  Batch data generation
# ---------------------------------------------------------------------------

def generate_samples(
    n_samples: int = 50_000,
    n_dets: int = 30,
    alpha: float = 3.0,
    seed: int = 42,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate solver-labeled training data for BetliNet.

    For each sample:
    1. Deal biased 10-card hand + 2 defenders + talon
    2. Setup betli game state
    3. Run PIMC solver (n_dets determinizations)
    4. Label = win fraction → converted to [-1, 1] range
    5. Encode hand → feature vector

    Returns (features, labels) arrays.
    """
    import time as _time
    rng = random.Random(seed)
    game = UltiGame(restrictions=[])
    features_list: list[np.ndarray] = []
    labels_list: list[float] = []
    t0 = _time.perf_counter()

    for i in range(n_samples):
        hands, _talon = deal_betli(rng, alpha=alpha)
        win_frac = solver_label_fast(hands, n_dets=n_dets, rng=rng, game=game)

        # Encode soloist's 10-card hand
        feats = encode_hand(hands[0])
        # Convert [0, 1] → [-1, 1] for MSE training (consistent with hand_evaluator)
        label = 2.0 * win_frac - 1.0

        features_list.append(feats)
        labels_list.append(label)

        if progress and (i + 1) % 100 == 0:
            elapsed = _time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n_samples - i - 1) / max(rate, 0.01)
            print(f"\r  Generated {i + 1}/{n_samples} ({rate:.0f}/s, ETA {eta:.0f}s)", end="", flush=True)

    if progress:
        print()

    return np.stack(features_list), np.array(labels_list, dtype=np.float32)
