"""CFR (Counterfactual Regret Minimization) bidding strategy for Ulti.

Computes an approximate Nash equilibrium bidding strategy offline using
external-sampling MCCFR.  The pre-trained value heads serve as terminal
evaluators — when the auction resolves, ``evaluate_all_contracts`` scores
the winner's hand and returns game_pts as the payoff.

The strategy table maps ``(hand_bucket, auction_info_key) → action_probs``
and can be loaded at tournament time as a drop-in replacement for the
Kermit-style greedy bidding.

Hand bucketing uses a coarse discretisation:
  (trump_count, strength_tier, has_marriage, best_side_suit_len)
yielding ~200 buckets, balancing precision with convergence speed.
"""
from __future__ import annotations

import copy
import itertools
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trickster.bidding.constants import PASS_PENALTY
from trickster.bidding.evaluator import evaluate_contract
from trickster.bidding.registry import (
    BID_TO_CONTRACT,
    CONTRACT_DEFS,
    CONTRACT_TO_BID_RANK,
    SUPPORTED_BID_RANKS,
)
from trickster.games.ulti.auction import (
    AuctionState,
    BID_BY_RANK,
    BID_PASSZ,
    Bid,
    can_pickup,
    create_auction,
    legal_bids,
    submit_bid,
    submit_pass,
    submit_pickup,
)
from trickster.games.ulti.cards import Card, Rank, Suit, make_deck
from trickster.games.ulti.game import GameState, deal, next_player
from trickster.model import UltiNetWrapper
from trickster.train_utils import _GAME_PTS_MAX


# ---------------------------------------------------------------------------
#  Hand bucketing
# ---------------------------------------------------------------------------

def _suit_counts(hand: list[Card]) -> dict[Suit, int]:
    counts: dict[Suit, int] = {}
    for c in hand:
        counts[c.suit] = counts.get(c.suit, 0) + 1
    return counts


def _best_trump_suit(hand: list[Card]) -> tuple[Suit, int]:
    """Return the suit with the most cards and its count."""
    counts = _suit_counts(hand)
    best_suit = max(counts, key=counts.get)
    return best_suit, counts[best_suit]


def _total_strength(hand: list[Card]) -> int:
    """Sum of normal-order strength values (0-7 per card)."""
    return sum(int(c.rank) for c in hand)


def _has_marriage(hand: list[Card]) -> bool:
    """Whether hand contains a King+Queen in the same suit."""
    hand_set = set(hand)
    for suit in Suit:
        if Card(suit, Rank.KING) in hand_set and Card(suit, Rank.QUEEN) in hand_set:
            return True
    return False


def _best_side_suit_length(hand: list[Card], trump_suit: Suit) -> int:
    """Length of the longest non-trump suit."""
    counts = _suit_counts(hand)
    best = 0
    for suit, cnt in counts.items():
        if suit != trump_suit and cnt > best:
            best = cnt
    return best


def bucket_hand(hand: list[Card]) -> int:
    """Map a hand to a bucket ID for CFR.

    Features:
      - trump_count (0-8, capped at 8): 9 bins
      - strength_tier (0-3): 4 bins  (total strength / 20, capped)
      - has_marriage (0-1): 2 bins
      - best_side_suit_len (0-3, capped): 4 bins

    Total: 9 × 4 × 2 × 4 = 288 buckets.
    """
    trump_suit, trump_count = _best_trump_suit(hand)
    trump_count = min(trump_count, 8)

    strength = _total_strength(hand)
    strength_tier = min(strength // 20, 3)

    marriage = 1 if _has_marriage(hand) else 0

    side_len = min(_best_side_suit_length(hand, trump_suit), 3)

    return (trump_count * 4 * 2 * 4
            + strength_tier * 2 * 4
            + marriage * 4
            + side_len)


NUM_BUCKETS = 9 * 4 * 2 * 4  # 288


# ---------------------------------------------------------------------------
#  Auction state abstraction
# ---------------------------------------------------------------------------

def auction_info_key(auction: AuctionState, player: int) -> str:
    """Compact information-set key for the auction state.

    Uses a coarse abstraction that keeps only strategically relevant
    information, avoiding history-dependent explosion:

      - Relative position to first bidder (0, 1, 2)
      - Current bid rank (0 if none)
      - Whether this player is the holder
      - Whether we're awaiting a bid (i.e. just picked up)
      - Number of pickups so far (0, 1, 2 — auction depth)

    This yields a manageable number of info sets per bucket while
    capturing the essential strategic situation.
    """
    rel_pos = (player - auction.first_bidder) % 3
    bid_rank = auction.current_bid.rank if auction.current_bid else 0
    is_holder = 1 if auction.holder == player else 0
    awaiting = 1 if auction.awaiting_bid else 0

    # Count pickups as a rough measure of auction depth
    n_pickups = sum(1 for _, action, _ in auction.history if action == "pickup")

    return f"{rel_pos}|{bid_rank}|{is_holder}|{awaiting}|{n_pickups}"


# ---------------------------------------------------------------------------
#  Action space
# ---------------------------------------------------------------------------

# Actions at each decision point, indexed by integer:
#   Pickup decision: 0=pass, 1=pickup
#   Bid decision: actions correspond to supported bid ranks + passz

# Supported bid ranks in order (for indexing)
_BID_ACTIONS: list[int] = sorted(SUPPORTED_BID_RANKS)  # [1, 2, 3, 4, 5, 8, 10, 11]


def _legal_bid_action_mask(auction: AuctionState) -> list[bool]:
    """Boolean mask over _BID_ACTIONS for legal bids."""
    legal = set(b.rank for b in legal_bids(auction))
    return [rank in legal for rank in _BID_ACTIONS]


# ---------------------------------------------------------------------------
#  CFR node
# ---------------------------------------------------------------------------

@dataclass
class CFRNode:
    """Information-set node for CFR."""

    info_key: str
    num_actions: int
    regret_sum: np.ndarray = field(default=None)
    strategy_sum: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.regret_sum is None:
            self.regret_sum = np.zeros(self.num_actions, dtype=np.float64)
        if self.strategy_sum is None:
            self.strategy_sum = np.zeros(self.num_actions, dtype=np.float64)

    def current_strategy(self) -> np.ndarray:
        """Regret-matching: positive regrets normalised to probabilities."""
        pos = np.maximum(self.regret_sum, 0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.ones(self.num_actions, dtype=np.float64) / self.num_actions

    def average_strategy(self) -> np.ndarray:
        """Time-averaged strategy (converges to Nash equilibrium)."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(self.num_actions, dtype=np.float64) / self.num_actions


# ---------------------------------------------------------------------------
#  CFR Strategy (loaded at tournament time)
# ---------------------------------------------------------------------------

@dataclass
class CFRStrategy:
    """Loaded CFR strategy table for one player."""

    # (bucket, auction_key) → action_probs (numpy array)
    pickup_table: dict[str, np.ndarray] = field(default_factory=dict)
    bid_table: dict[str, np.ndarray] = field(default_factory=dict)

    def get_pickup_probs(self, bucket: int, auction_key: str) -> np.ndarray | None:
        """Get pickup probabilities [p_pass, p_pickup]."""
        key = f"pickup|{bucket}|{auction_key}"
        return self.pickup_table.get(key)

    def get_bid_probs(self, bucket: int, auction_key: str) -> np.ndarray | None:
        """Get bid probabilities over _BID_ACTIONS."""
        key = f"bid|{bucket}|{auction_key}"
        return self.bid_table.get(key)


# ---------------------------------------------------------------------------
#  Terminal evaluation
# ---------------------------------------------------------------------------

def _evaluate_terminal(
    gs: GameState,
    soloist: int,
    dealer: int,
    bid_rank: int,
    wrappers: dict[str, UltiNetWrapper],
) -> float:
    """Evaluate a resolved auction: return game_pts for the soloist.

    Evaluates only the specific contract that was bid (looked up via
    ``BID_TO_CONTRACT``), not all contracts.  This is ~15× faster and
    reflects the actual committed bid rather than the best alternative.
    """
    hand = gs.hands[soloist]
    if len(hand) != 12:
        return 0.0

    contract_info = BID_TO_CONTRACT.get(bid_rank)
    if contract_info is None:
        return 0.0

    contract_key, is_piros = contract_info
    cdef = CONTRACT_DEFS.get(contract_key)
    if cdef is None:
        return 0.0

    wrapper = wrappers.get(contract_key)
    if wrapper is None:
        return 0.0

    if cdef.is_betli:
        trump = None
    elif is_piros:
        trump = Suit.HEARTS
    else:
        counts: dict[Suit, int] = {}
        for c in hand:
            counts[c.suit] = counts.get(c.suit, 0) + 1
        trump = max(counts, key=counts.get)

    ev = evaluate_contract(
        gs, soloist, dealer, cdef,
        trump=trump, is_piros=is_piros,
        wrapper=wrapper,
    )
    return ev.game_pts if ev is not None else 0.0


# ---------------------------------------------------------------------------
#  Parallel worker helpers (module-level for pickling)
# ---------------------------------------------------------------------------

_CFR_WORKER_WRAPPERS: dict[str, UltiNetWrapper] = {}


def _cfr_worker_init(wrappers: dict[str, UltiNetWrapper]) -> None:
    global _CFR_WORKER_WRAPPERS
    _CFR_WORKER_WRAPPERS = wrappers


def _cfr_worker_fn(
    args: tuple[int, int, int, int, int],
) -> tuple[dict[str, "CFRNode"], int]:
    """Run CFR iterations in a worker process. Returns (nodes, n_iterations)."""
    n_iterations, n_hands_per_iter, seed, progress_every, worker_id = args
    solver = CFRSolver(wrappers=_CFR_WORKER_WRAPPERS)
    solver.train(
        n_iterations=n_iterations,
        n_hands_per_iter=n_hands_per_iter,
        rng=random.Random(seed),
        progress_every=progress_every,
        workers=1,
    )
    return solver.nodes, solver.iterations


# ---------------------------------------------------------------------------
#  CFR Solver
# ---------------------------------------------------------------------------

_ALL_CARDS_SET = frozenset(make_deck())


class CFRSolver:
    """External-sampling MCCFR for the Ulti auction.

    Each iteration: deal a random hand, compute a bucket, and traverse
    the auction tree for the traversing player.  Opponent actions are
    sampled from their current strategy; the traverser explores all
    actions and updates regrets.
    """

    def __init__(self, wrappers: dict[str, UltiNetWrapper]):
        self.wrappers = wrappers
        self.nodes: dict[str, CFRNode] = {}
        self.iterations = 0

    def _get_node(self, info_key: str, num_actions: int) -> CFRNode:
        if info_key not in self.nodes:
            self.nodes[info_key] = CFRNode(info_key=info_key, num_actions=num_actions)
        return self.nodes[info_key]

    def train(
        self,
        n_iterations: int,
        n_hands_per_iter: int = 100,
        rng: random.Random | None = None,
        progress_every: int = 100,
        workers: int = 1,
    ) -> None:
        """Run external-sampling MCCFR iterations.

        When *workers* > 1, splits iterations across independent solver
        processes and merges their node tables (regret/strategy sums are
        additive).
        """
        if workers > 1:
            self._train_parallel(
                n_iterations, n_hands_per_iter, rng, progress_every, workers,
            )
            return

        rng = rng or random.Random()

        for it in range(n_iterations):
            for _ in range(n_hands_per_iter):
                seed = rng.randint(0, 2**31)
                dealer = rng.randint(0, 2)
                gs, talon = deal(seed=seed, dealer=dealer)

                # Pre-compute bucket for each player's 10-card hand
                buckets = [bucket_hand(gs.hands[p]) for p in range(3)]

                # Traverse for each player
                for traverser in range(3):
                    gs_copy = copy.deepcopy(gs)
                    talon_copy = list(talon)
                    first_bidder = next_player(dealer)

                    # Set up auction: first bidder picks up talon
                    gs_copy.hands[first_bidder].extend(talon_copy)
                    auction = create_auction(first_bidder, talon_copy)

                    self._cfr_traverse(
                        gs_copy, auction, dealer, traverser, buckets,
                        rng, reach_prob=1.0,
                    )

            self.iterations += 1

            if progress_every > 0 and (it + 1) % progress_every == 0:
                n_nodes = len(self.nodes)
                # Compute strategy entropy as a diagnostic
                entropies = []
                for node in self.nodes.values():
                    strat = node.average_strategy()
                    ent = -np.sum(strat * np.log(strat + 1e-10))
                    entropies.append(ent)
                avg_entropy = np.mean(entropies) if entropies else 0.0
                print(f"  Iteration {it + 1}/{n_iterations}: "
                      f"{n_nodes} info sets, avg entropy={avg_entropy:.3f}")

    def _train_parallel(
        self,
        n_iterations: int,
        n_hands_per_iter: int,
        rng: random.Random | None,
        progress_every: int,
        workers: int,
    ) -> None:
        """Run CFR training across multiple processes, then merge."""
        from concurrent.futures import ProcessPoolExecutor

        rng = rng or random.Random()

        # Split iterations across workers with unique seeds
        worker_args = []
        iters_per_worker = n_iterations // workers
        remainder = n_iterations % workers
        for w in range(workers):
            w_iters = iters_per_worker + (1 if w < remainder else 0)
            w_seed = rng.randint(0, 2**31)
            worker_args.append((w_iters, n_hands_per_iter, w_seed, progress_every, w))

        print(f"  Spawning {workers} workers "
              f"({iters_per_worker}-{iters_per_worker + 1} iters each)...",
              flush=True)

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_cfr_worker_init,
            initargs=(self.wrappers,),
        ) as pool:
            results = list(pool.map(_cfr_worker_fn, worker_args))

        # Merge node tables from all workers
        for worker_nodes, worker_iters in results:
            self.iterations += worker_iters
            for key, wnode in worker_nodes.items():
                if key in self.nodes:
                    self.nodes[key].regret_sum += wnode.regret_sum
                    self.nodes[key].strategy_sum += wnode.strategy_sum
                else:
                    self.nodes[key] = wnode

        print(f"  Merged: {len(self.nodes)} info sets, "
              f"{self.iterations} total iterations")

    def _cfr_traverse(
        self,
        gs: GameState,
        auction: AuctionState,
        dealer: int,
        traverser: int,
        buckets: list[int],
        rng: random.Random,
        reach_prob: float,
    ) -> float:
        """Recursive CFR traversal through the auction tree.

        Returns the expected payoff for the traverser.
        """
        if auction.done:
            return self._terminal_payoff(gs, auction, dealer, traverser)

        player = auction.turn

        # Auto-pass when bid is at or above max supported rank
        from trickster.bidding.registry import MAX_SUPPORTED_RANK
        if (
            auction.current_bid is not None
            and auction.current_bid.rank >= MAX_SUPPORTED_RANK
            and not auction.awaiting_bid
        ):
            submit_pass(auction, player)
            return self._cfr_traverse(
                gs, auction, dealer, traverser, buckets, rng, reach_prob,
            )

        # Holder's turn came back — stand
        if not auction.awaiting_bid and player == auction.holder:
            submit_pass(auction, player)
            return self._cfr_traverse(
                gs, auction, dealer, traverser, buckets, rng, reach_prob,
            )

        bucket = buckets[player]
        akey = auction_info_key(auction, player)

        if auction.awaiting_bid:
            # Bid decision: player has 12 cards, must discard + bid
            return self._traverse_bid(
                gs, auction, dealer, player, traverser, bucket, akey,
                buckets, rng, reach_prob,
            )
        else:
            # Pickup decision: pass or pickup
            return self._traverse_pickup(
                gs, auction, dealer, player, traverser, bucket, akey,
                buckets, rng, reach_prob,
            )

    def _traverse_pickup(
        self,
        gs: GameState,
        auction: AuctionState,
        dealer: int,
        player: int,
        traverser: int,
        bucket: int,
        akey: str,
        buckets: list[int],
        rng: random.Random,
        reach_prob: float,
    ) -> float:
        """Traverse a pickup decision node (pass/pickup)."""
        can_pu = can_pickup(auction)
        if not can_pu:
            # Must pass
            submit_pass(auction, player)
            return self._cfr_traverse(
                gs, auction, dealer, traverser, buckets, rng, reach_prob,
            )

        info_key = f"pickup|{bucket}|{akey}"
        node = self._get_node(info_key, 2)
        strategy = node.current_strategy()

        if player == traverser:
            # Explore all actions
            values = np.zeros(2, dtype=np.float64)

            for action_idx in range(2):
                gs_c = copy.deepcopy(gs)
                a_c = copy.deepcopy(auction)

                if action_idx == 0:
                    # Pass
                    submit_pass(a_c, player)
                else:
                    # Pickup
                    gs_c.hands[player].extend(a_c.talon)
                    submit_pickup(a_c, player)
                    # Update bucket for post-pickup 12-card hand
                    buckets_c = list(buckets)
                    buckets_c[player] = bucket_hand(gs_c.hands[player])
                    values[action_idx] = self._cfr_traverse(
                        gs_c, a_c, dealer, traverser, buckets_c, rng, reach_prob,
                    )
                    continue

                values[action_idx] = self._cfr_traverse(
                    gs_c, a_c, dealer, traverser, list(buckets), rng, reach_prob,
                )

            # Update regrets
            ev = np.dot(strategy, values)
            node.regret_sum += values - ev
            node.strategy_sum += reach_prob * strategy
            return ev
        else:
            # Sample from strategy
            action_idx = rng.choices([0, 1], weights=strategy.tolist(), k=1)[0]

            if action_idx == 0:
                submit_pass(auction, player)
            else:
                gs.hands[player].extend(auction.talon)
                submit_pickup(auction, player)
                buckets[player] = bucket_hand(gs.hands[player])

            node.strategy_sum += reach_prob * strategy
            return self._cfr_traverse(
                gs, auction, dealer, traverser, buckets, rng,
                reach_prob * strategy[action_idx],
            )

    def _traverse_bid(
        self,
        gs: GameState,
        auction: AuctionState,
        dealer: int,
        player: int,
        traverser: int,
        bucket: int,
        akey: str,
        buckets: list[int],
        rng: random.Random,
        reach_prob: float,
    ) -> float:
        """Traverse a bid decision node."""
        legal_mask = _legal_bid_action_mask(auction)
        legal_indices = [i for i, m in enumerate(legal_mask) if m]

        if not legal_indices:
            # No legal bids (shouldn't happen in normal play)
            return 0.0

        num_legal = len(legal_indices)
        info_key = f"bid|{bucket}|{akey}"
        node = self._get_node(info_key, len(_BID_ACTIONS))
        strategy = node.current_strategy()

        # Zero out illegal actions and renormalise
        masked_strategy = np.zeros_like(strategy)
        for i in legal_indices:
            masked_strategy[i] = strategy[i]
        total = masked_strategy.sum()
        if total > 0:
            masked_strategy /= total
        else:
            for i in legal_indices:
                masked_strategy[i] = 1.0 / num_legal

        if player == traverser:
            values = np.full(len(_BID_ACTIONS), 0.0, dtype=np.float64)

            for action_idx in legal_indices:
                gs_c = copy.deepcopy(gs)
                a_c = copy.deepcopy(auction)
                bid_rank = _BID_ACTIONS[action_idx]
                bid_obj = BID_BY_RANK[bid_rank]

                # Heuristic discards for the auction state — but keep the
                # 12-card hand intact so terminal evaluation can use it.
                hand = gs_c.hands[player]
                discards = sorted(hand, key=lambda c: (c.points(), c.strength()))[:2]
                submit_bid(a_c, player, bid_obj, discards)
                # NOTE: hand stays at 12 cards for NN evaluation at terminal

                values[action_idx] = self._cfr_traverse(
                    gs_c, a_c, dealer, traverser, list(buckets), rng, reach_prob,
                )

            ev = np.dot(masked_strategy, values)
            # Update regrets only for legal actions
            for i in legal_indices:
                node.regret_sum[i] += values[i] - ev
            node.strategy_sum += reach_prob * masked_strategy
            return ev
        else:
            # Sample from masked strategy
            action_idx = rng.choices(
                range(len(_BID_ACTIONS)),
                weights=masked_strategy.tolist(),
                k=1,
            )[0]
            bid_rank = _BID_ACTIONS[action_idx]
            bid_obj = BID_BY_RANK[bid_rank]

            hand = gs.hands[player]
            discards = sorted(hand, key=lambda c: (c.points(), c.strength()))[:2]
            submit_bid(auction, player, bid_obj, discards)
            # NOTE: hand stays at 12 cards for NN evaluation at terminal

            node.strategy_sum += reach_prob * masked_strategy
            return self._cfr_traverse(
                gs, auction, dealer, traverser, buckets, rng,
                reach_prob * masked_strategy[action_idx],
            )

    def _terminal_payoff(
        self,
        gs: GameState,
        auction: AuctionState,
        dealer: int,
        traverser: int,
    ) -> float:
        """Compute payoff at a terminal auction node."""
        soloist = auction.winner
        bid = auction.current_bid

        # Passz: soloist pays penalty
        if bid is not None and bid.rank <= BID_PASSZ.rank:
            if traverser == soloist:
                return -PASS_PENALTY * 2
            return PASS_PENALTY

        # Real game: evaluate with value heads
        hand = gs.hands[soloist]
        if len(hand) == 12:
            game_pts = _evaluate_terminal(
                gs, soloist, dealer, bid.rank if bid else 0, self.wrappers,
            )
        else:
            game_pts = 0.0

        if traverser == soloist:
            return game_pts * 2  # soloist earns 2× (from each defender)
        return -game_pts  # each defender loses game_pts

    # ------------------------------------------------------------------
    #  Strategy extraction
    # ------------------------------------------------------------------

    def build_strategy(self) -> CFRStrategy:
        """Extract the average strategy into a CFRStrategy object."""
        strat = CFRStrategy()
        for key, node in self.nodes.items():
            avg = node.average_strategy()
            if key.startswith("pickup|"):
                strat.pickup_table[key] = avg
            elif key.startswith("bid|"):
                strat.bid_table[key] = avg
        return strat

    def save(self, path: str | Path) -> None:
        """Save strategy table + solver state to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "strategy": self.build_strategy(),
            "nodes": self.nodes,
            "iterations": self.iterations,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved CFR strategy to {path} "
              f"({len(self.nodes)} info sets, {self.iterations} iterations)")

    @classmethod
    def load(cls, path: str | Path, wrappers: dict[str, UltiNetWrapper] | None = None) -> "CFRSolver":
        """Load solver state from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        solver = cls(wrappers=wrappers or {})
        solver.nodes = data["nodes"]
        solver.iterations = data.get("iterations", 0)
        return solver


def load_cfr_strategy(path: str | Path) -> CFRStrategy:
    """Load just the strategy table (lightweight, no solver state needed)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, CFRStrategy):
        return data
    return data["strategy"]


# ---------------------------------------------------------------------------
#  CFR-based decision functions (drop-in for Kermit)
# ---------------------------------------------------------------------------

def cfr_decide_pickup(
    hand: list[Card],
    auction: AuctionState,
    player: int,
    strategy: CFRStrategy,
    *,
    rng: random.Random | None = None,
    deterministic: bool = False,
) -> bool:
    """Decide whether to pick up using CFR strategy.

    Returns True if the player should pick up.
    """
    if not can_pickup(auction):
        return False

    bucket = bucket_hand(hand)
    akey = auction_info_key(auction, player)
    probs = strategy.get_pickup_probs(bucket, akey)

    if probs is None:
        # Unknown info set — default to not picking up
        return False

    if deterministic:
        return bool(probs[1] > probs[0])

    rng = rng or random.Random()
    return rng.random() < probs[1]


def cfr_decide_bid(
    hand: list[Card],
    auction: AuctionState,
    player: int,
    strategy: CFRStrategy,
    *,
    rng: random.Random | None = None,
    deterministic: bool = False,
) -> Bid:
    """Decide what to bid using CFR strategy.

    Returns the selected Bid object.
    """
    bucket = bucket_hand(hand)
    akey = auction_info_key(auction, player)
    probs = strategy.get_bid_probs(bucket, akey)

    legal_mask = _legal_bid_action_mask(auction)
    legal_indices = [i for i, m in enumerate(legal_mask) if m]

    if not legal_indices:
        return BID_PASSZ

    if probs is not None:
        # Mask illegal actions and renormalise
        masked = np.zeros_like(probs)
        for i in legal_indices:
            masked[i] = probs[i]
        total = masked.sum()
        if total > 0:
            masked /= total
        else:
            for i in legal_indices:
                masked[i] = 1.0 / len(legal_indices)

        if deterministic:
            action_idx = int(np.argmax(masked))
        else:
            rng = rng or random.Random()
            action_idx = rng.choices(
                range(len(_BID_ACTIONS)), weights=masked.tolist(), k=1,
            )[0]
    else:
        # Unknown info set — pick lowest legal bid
        action_idx = legal_indices[0]

    bid_rank = _BID_ACTIONS[action_idx]
    return BID_BY_RANK[bid_rank]
