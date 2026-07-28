"""
Core game engine for RabloUlti.

No gymnasium dependency. This module is the single source of truth for
all game rules: dealing, über, scoring, and win detection.

Player roles
------------
    Player 0  — declarer  (gets 12 cards, discards 2, chooses trump)
    Player 1  — defender
    Player 2  — defender

Licits (bids)
-------------
    PARTI                  — standard game; 1 GP/defender (declarer nets ±2)
    PIROS_PARTI            — hearts trump; 2 GP/defender (declarer nets ±4)
    NEGYVENSZAZ            — 40-100; requires trump king+upper in hand; 4 GP/defender
    PIROS_NEGYVENSZAZ      — Piros 40-100 (hearts trump); 8 GP/defender
    ULTI                   — composite: Ulti (4 GP/def) + Parti (1 GP/def); requires trump 7
    PIROS_ULTI             — composite: Piros Ulti (8 GP/def) + Piros Parti (2 GP/def)
    NEGYVENSZAZ_ULTI       — composite: Ulti (4 GP/def) + 40-100 (4 GP/def);
                             requires both trump 7 and trump king+upper
    PIROS_NEGYVENSZAZ_ULTI — Piros 40-100+Ulti; 8+8 GP/defender each component

Silent bids
-----------
    A silent 40-100 occurs when the declarer bids Parti/Piros Parti, wins the
    game, AND also satisfies the 40-100 win condition (trick_scores[0] >= 60
    while holding the trump king+upper pair after discard).
    Reward: 50% of the explicit 40-100 GP value as a bonus on top of Parti GPs.
        PARTI + silent:       bonus = 2 GP/defender  → declarer nets +6 (±2 Parti + ±4 bonus)
        PIROS_PARTI + silent: bonus = 4 GP/defender  → declarer nets +12 (±4 Piros + ±8 bonus)

Silent Ulti
-----------
    A silent Ulti occurs when the declarer wins the game AND wins the last trick
    by playing the trump 7 (rank_index == 0), regardless of which bid was declared.
    The win conditions do not interfere, so it applies to Parti AND 40-100 games.
    The explicit Ulti bid is worth 4 GP/defender; silent = 50% → 2 GP/defender bonus.
    Silent Ulti stacks with silent 40-100 when both conditions hold simultaneously.

40-100 rules
------------
    - Declarer must hold king+upper of the trump suit after discarding
      (the trump pair is protected: it cannot be discarded)
    - The 40 (trump pair bonus) is awarded automatically
    - Win condition: trick_scores[0] >= 60 (aces + tens + last-trick bonus)
    - Additional 20-point sets are NOT counted; defenders do not declare sets

40-100+Ulti rules
-----------------
    - Requires trump 7 AND trump king+upper; both are protected from discard
    - Composite bid: each component (Ulti and 40-100) is evaluated independently
    - Ulti component (4 or 8 GP/def): won iff _declarer_ulti
    - 40-100 component (4 or 8 GP/def): won iff trick_scores[0] >= 60
    - Full win (both): +8 GP/def; partial (one each): 0 GP/def; loss: −8 GP/def

Phase order
-----------
    TRUMP → DISCARD → PLAY → DONE

Action encoding (shared Discrete(32) action space)
---------------------------------------------------
    Phase.TRUMP   →  0..15
        0=acorns Parti    1=leaves Parti    2=hearts Piros Parti   3=bells Parti
        4=acorns 40-100   5=leaves 40-100   6=hearts Piros 40-100  7=bells 40-100
        8=acorns Ulti     9=leaves Ulti    10=hearts Piros Ulti   11=bells Ulti
       12=acorns 40+Ulti 13=leaves 40+Ulti 14=hearts Piros 40+Ulti 15=bells 40+Ulti
        Options 4-7 only legal when declarer holds king+upper of that suit.
        Options 8-11 only legal when declarer holds the 7 of that suit.
        Options 12-15 only legal when declarer holds BOTH the 7 and king+upper of that suit.
    Phase.DISCARD →  card_id  0..31
        For 40-100 licits, king and upper of the trump suit are masked out.
        For Ulti licits, the trump 7 is masked out.
        For 40-100+Ulti licits, the trump 7, king, and upper are all masked out.
    Phase.PLAY    →  card_id  0..31  (masked to legal plays)

Licit is set in _do_trump() and visible in the `licit` observation field.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .card import Card, SUITS, card_from_id, fresh_deck

HEARTS_ID = SUITS.index('hearts')   # = 2


class Phase(Enum):
    LICIT   = 'licit'    # Legacy — no longer visited; kept for observation index stability
    DISCARD = 'discard'  # Declarer discards 2 cards to the talon
    TRUMP   = 'trump'    # Declarer declares the trump suit (first phase of each episode)
    PLAY    = 'play'     # 10 tricks of card play
    DONE    = 'done'     # Game over; call payoffs() / game_points()


class Licit(Enum):
    PARTI                  = 'parti'                   # Standard; 1 GP/defender
    PIROS_PARTI            = 'piros_parti'             # Hearts trump; 2 GP/defender
    NEGYVENSZAZ            = 'negyvenszaz'             # 40-100; 4 GP/defender
    PIROS_NEGYVENSZAZ      = 'piros_negyvenszaz'       # Piros 40-100; 8 GP/defender
    ULTI                   = 'ulti'                    # Ulti (4) + Parti (1) GP/defender
    PIROS_ULTI             = 'piros_ulti'              # Piros Ulti (8) + Piros Parti (2) GP/defender
    NEGYVENSZAZ_ULTI       = 'negyvenszaz_ulti'        # Ulti (4) + 40-100 (4) GP/defender each
    PIROS_NEGYVENSZAZ_ULTI = 'piros_negyvenszaz_ulti'  # Piros Ulti (8) + Piros 40-100 (8) GP/defender each

# Licits that use the 40-100 win condition (trick_scores[0] >= 60)
_NEGYVENSZAZ_LICITS = frozenset({Licit.NEGYVENSZAZ, Licit.PIROS_NEGYVENSZAZ})
# Composite Ulti licits (Ulti win + Parti win, evaluated independently)
_ULTI_LICITS = frozenset({Licit.ULTI, Licit.PIROS_ULTI})
# Composite 40-100+Ulti licits (Ulti win + 40-100 win, evaluated independently)
_NEGYVENSZAZ_ULTI_LICITS = frozenset({Licit.NEGYVENSZAZ_ULTI, Licit.PIROS_NEGYVENSZAZ_ULTI})
# All licits in the 40-100 family (only declarer announces; no silent 40-100)
_NEGYVENSZAZ_FAMILY = _NEGYVENSZAZ_LICITS | _NEGYVENSZAZ_ULTI_LICITS
# All licits that include the Ulti win condition (declarer must win last trick with trump 7)
_ALL_ULTI_LICITS = _ULTI_LICITS | _NEGYVENSZAZ_ULTI_LICITS


class UltiGame:
    """
    RabloUlti for 3 players, Parti / Piros Parti licits.

    Typical usage
    -------------
    >>> game = UltiGame()
    >>> obs = game.reset()
    >>> while not game.is_over():
    ...     action = my_agent(obs, game.legal_actions())
    ...     obs, done = game.step(action)
    >>> print(game.game_points())    # (declarer, def1, def2) GP — zero-sum
    >>> print(game.declarer_wins())  # True / False
    """

    NUM_PLAYERS  = 3
    DECLARER_ID  = 0
    NUM_TRICKS   = 10

    def __init__(self) -> None:
        self._rng = np.random.default_rng()
        # Typed attributes — populated properly in reset()
        self.phase: Phase = Phase.DONE
        self.licit: Optional[Licit] = None
        self.hands: List[List[Card]] = [[], [], []]
        self.talon: List[Card] = []
        self.trump_suit: Optional[str] = None
        self.king_upper_points: List[int] = [0, 0, 0]
        self.current_player_id: int = 0
        self.trick_leader: int = 0
        self.current_trick: List[Tuple[int, Card]] = []
        self.trick_number: int = 0
        self.trick_scores: List[int] = [0, 0, 0]
        self._talon_points: int = 0
        self._played_cards: np.ndarray = np.zeros(32, dtype=np.int8)
        self._player_voids: List[set] = [set(), set(), set()]
        # Über upper bound: max rank_index player can hold per suit (default 7 = ace).
        # Updated when a player follows suit but cannot beat the current trick best,
        # proving they hold no card of that suit with higher rank.
        self._player_suit_uppers: np.ndarray = np.full((3, 4), 7, dtype=np.int8)
        self._declarer_has_trump_pair: bool = False
        self._declarer_ulti: bool = False
        self._defender_has_trump_pair: bool = False
        self._defender_ulti: bool = False
        self._announced_pairs: np.ndarray = np.zeros((3, 4), dtype=np.int8)  # [player][suit]
        # Reward-shaping helpers — populated during play
        self.last_trick_plays: List[Tuple[int, int]] = []  # [(pid, card_id)] play order
        self.last_trick_winner: int = -1                   # winner of last completed trick
        self.play_start_hands: List[np.ndarray] = []       # 3 × 32-bit arrays at PLAY start
        self.talon_card_ids: List[int] = []                # card IDs of the 2 discards
        # Per-player captured cards + per-trick leader/winner history. Used by
        # external adapters (e.g. trickster ONNX model) that need card-counting
        # context. Always present; ignored by the play_flatten() vector that
        # SB3 models consume.
        self.captured_cards: List[List[Card]] = [[], [], []]
        self.trick_history_leaders: List[int] = []
        self.trick_history_winners: List[int] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Dict:
        """
        Shuffle, deal, and return the declarer's initial observation.

        Hands
        -----
        Player 0 (declarer) : 12 cards
        Player 1 (defender) : 10 cards
        Player 2 (defender) : 10 cards
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        deck = fresh_deck()
        self._rng.shuffle(deck)

        self.phase             = Phase.TRUMP
        self.licit             = None
        self.hands             = [list(deck[:12]), list(deck[12:22]), list(deck[22:32])]
        self.talon             = []
        self.trump_suit        = None
        self.king_upper_points = [0, 0, 0]
        self.current_player_id = 0
        self.trick_leader      = 0
        self.current_trick     = []
        self.trick_number      = 0
        self.trick_scores      = [0, 0, 0]
        self._talon_points     = 0
        self._played_cards     = np.zeros(32, dtype=np.int8)
        self._player_voids        = [set(), set(), set()]
        self._player_suit_uppers  = np.full((3, 4), 7, dtype=np.int8)
        self._declarer_has_trump_pair = False
        self._declarer_ulti           = False
        self._defender_has_trump_pair = False
        self._defender_ulti           = False
        self._announced_pairs         = np.zeros((3, 4), dtype=np.int8)
        self.last_trick_plays  = []
        self.last_trick_winner = -1
        self.play_start_hands  = []
        self.talon_card_ids    = []
        self.captured_cards         = [[], [], []]
        self.trick_history_leaders  = []
        self.trick_history_winners  = []

        return self.get_observation(0)

    def legal_actions(self) -> List[int]:
        """Return legal action IDs for the current player."""
        if self.phase == Phase.TRUMP:
            actions = list(range(4))   # Parti / Piros Parti always available
            # 40-100 options (4-7) legal only when declarer holds trump pair
            for suit_id, suit in enumerate(SUITS):
                if self._has_pair(0, suit):
                    actions.append(4 + suit_id)
            # Ulti options (8-11) legal only when declarer holds the trump 7
            for suit_id, suit in enumerate(SUITS):
                if self._has_seven(0, suit):
                    actions.append(8 + suit_id)
            # 40-100+Ulti options (12-15) legal when declarer holds BOTH the 7 AND the pair
            for suit_id, suit in enumerate(SUITS):
                if self._has_seven(0, suit) and self._has_pair(0, suit):
                    actions.append(12 + suit_id)
            return actions
        if self.phase == Phase.DISCARD:
            hand = self.hands[0]
            if self.licit in _NEGYVENSZAZ_ULTI_LICITS:
                # Both the trump 7 and the trump king+upper must survive
                protected = frozenset(
                    c.id for c in hand
                    if c.suit == self.trump_suit
                    and (c.rank_index == 0 or c.rank in ('king', 'upper'))
                )
                return [c.id for c in hand if c.id not in protected]
            if self.licit in _NEGYVENSZAZ_LICITS:
                # Trump king+upper must survive to the play phase
                protected = frozenset(
                    c.id for c in hand
                    if c.suit == self.trump_suit and c.rank in ('king', 'upper')
                )
                return [c.id for c in hand if c.id not in protected]
            if self.licit in _ULTI_LICITS:
                # Trump 7 must survive to the play phase
                protected = frozenset(
                    c.id for c in hand
                    if c.suit == self.trump_suit and c.rank_index == 0
                )
                return [c.id for c in hand if c.id not in protected]
            return [c.id for c in hand]
        if self.phase == Phase.PLAY:
            return self._legal_play_actions()
        return []   # Phase.DONE (and legacy Phase.LICIT)

    def step(self, action: int) -> Tuple[Dict, bool]:
        """
        Execute `action` for the current player.

        Returns
        -------
        obs  : observation dict for the *next* acting player (empty dict if done)
        done : True when the game has ended
        """
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(
                f'Illegal action {action} in {self.phase}. Legal: {legal}'
            )

        if self.phase == Phase.TRUMP:
            self._do_trump(action)
        elif self.phase == Phase.DISCARD:
            self._do_discard(action)
        elif self.phase == Phase.PLAY:
            self._do_play(action)

        done = self.phase == Phase.DONE
        obs  = {} if done else self.get_observation(self.current_player_id)
        return obs, done

    def get_observation(self, player_id: int) -> Dict:
        """
        Observation for `player_id`.

        Keys
        ----
        hand              (32,) int8    – one-hot: cards in this player's hand
        trump_suit         (5,) int8    – one-hot: 0-3 = suit, 4 = not yet declared
        phase              (4,) int8    – one-hot: LICIT=0 DISCARD=1 TRUMP=2 PLAY=3
        licit              (4,) int8    – one-hot: PARTI=0 PIROS_PARTI=1 NEGYVENSZAZ=2 PIROS_NEGYVENSZAZ=3
                                          (all zeros while in TRUMP phase, before licit is decided)
        current_trick     (96,) int8    – 3 trick-positions × 32 cards (play order)
        trick_number       (1,) float32 – trick_number / 10 (range 0.0–1.0)
        played_cards      (32,) int8    – one-hot: cards seen in completed previous tricks
        player_voids      (12,) int8    – P0: void_ac/le/he/be  P1: void_ac/le/he/be  P2: void_ac/le/he/be
        player_suit_uppers(12,) float32 – P0[4]  P1[4]  P2[4]: max holdable rank/7 per suit (1.0=unconstrained)
        announced_pairs   (12,) int8    – P0[4] P1[4] P2[4]: 1 if player holds king+upper of that suit
                                          (all zeros during TRUMP/DISCARD; filled at start of PLAY)
        trick_scores       (3,) float32 – P0/P1/P2 accumulated card trick points / 120
        king_upper_points  (3,) float32 – P0/P1/P2 announced pair bonus points / 120
        action_mask       (32,) int8    – 1 where action is legal (zeros if not this player's turn)
        """
        hand_rep = np.zeros(32, dtype=np.int8)
        for c in self.hands[player_id]:
            hand_rep[c.id] = 1

        trump_rep = np.zeros(5, dtype=np.int8)
        if self.trump_suit is not None:
            trump_rep[SUITS.index(self.trump_suit)] = 1
        else:
            trump_rep[4] = 1

        phase_rep = np.zeros(4, dtype=np.int8)
        if self.phase == Phase.TRUMP:
            phase_rep[2] = 1
        elif self.phase == Phase.DISCARD:
            phase_rep[1] = 1
        elif self.phase == Phase.PLAY:
            phase_rep[3] = 1
        # phase_rep[0] (LICIT) is never set — kept for observation index stability

        licit_rep = np.zeros(4, dtype=np.int8)
        if self.licit in (Licit.PARTI, Licit.ULTI):
            licit_rep[0] = 1
        elif self.licit in (Licit.PIROS_PARTI, Licit.PIROS_ULTI):
            licit_rep[1] = 1
        elif self.licit in (Licit.NEGYVENSZAZ, Licit.NEGYVENSZAZ_ULTI):
            licit_rep[2] = 1
        elif self.licit in (Licit.PIROS_NEGYVENSZAZ, Licit.PIROS_NEGYVENSZAZ_ULTI):
            licit_rep[3] = 1

        trick_rep = np.zeros(3 * 32, dtype=np.int8)
        for pos, (_, card) in enumerate(self.current_trick):
            trick_rep[pos * 32 + card.id] = 1

        trick_num = np.array([self.trick_number / 10.0], dtype=np.float32)

        void_rep = np.zeros(12, dtype=np.int8)
        for p_idx in range(3):
            for s_idx, suit in enumerate(SUITS):
                if suit in self._player_voids[p_idx]:
                    void_rep[p_idx * 4 + s_idx] = 1

        suit_upper_rep = (self._player_suit_uppers / 7.0).flatten().astype(np.float32)

        mask = np.zeros(32, dtype=np.int8)
        if self.current_player_id == player_id and self.phase != Phase.DONE:
            for a in self.legal_actions():
                mask[a] = 1

        pairs_rep       = self._announced_pairs.flatten().copy()   # (12,) int8
        trick_scores    = np.array(self.trick_scores,      dtype=np.float32) / 120.0
        ku_points       = np.array(self.king_upper_points, dtype=np.float32) / 120.0

        # starting_hand: each player's hand at the start of the PLAY phase
        if self.play_start_hands:
            starting_hand = self.play_start_hands[player_id].copy()
        else:
            starting_hand = np.zeros(32, dtype=np.float32)

        # talon: the 2 discarded cards — visible to the declarer, zeros for defenders
        talon_rep = np.zeros(32, dtype=np.float32)
        if player_id == 0:
            for cid in self.talon_card_ids:
                talon_rep[cid] = 1.0

        # Per-player captured cards (32-bit each). Adapters that need card-
        # counting context (e.g. trickster) read these; play_flatten() ignores
        # them so SB3 models trained on the 264-dim layout are unaffected.
        captured = [np.zeros(32, dtype=np.int8) for _ in range(3)]
        for pid in range(3):
            for c in self.captured_cards[pid]:
                captured[pid][c.id] = 1

        return {
            'hand':              hand_rep,
            'trump_suit':        trump_rep,
            'phase':             phase_rep,
            'licit':             licit_rep,
            'current_trick':     trick_rep,
            'trick_number':      trick_num,
            'played_cards':      self._played_cards.copy(),
            'player_voids':        void_rep,
            'player_suit_uppers':  suit_upper_rep,
            'announced_pairs':   pairs_rep,
            'trick_scores':      trick_scores,
            'king_upper_points': ku_points,
            'starting_hand':     starting_hand,
            'talon':             talon_rep,
            'action_mask':       mask,
            # Extension fields for adapters; ignored by play_flatten().
            'captured_p0':       captured[0],
            'captured_p1':       captured[1],
            'captured_p2':       captured[2],
            'trick_history_leaders': list(self.trick_history_leaders),
            'trick_history_winners': list(self.trick_history_winners),
        }

    def payoffs(self) -> Tuple[int, int, int]:
        """
        Total trick points per player: trick_scores + king_upper_points.

        Talon points are not included — they count toward the defenders'
        combined total. Use declarer_wins() for the win/loss result, and
        game_points() for the zero-sum GP outcome.
        """
        return tuple(self.trick_scores[i] + self.king_upper_points[i] for i in range(3))

    def game_points(self) -> Tuple[int, int, int]:
        """
        Zero-sum game points per player.

        Returns (0, 0, 0) if the game is not yet over.

        PARTI             : ±1 GP/defender  (declarer nets ±2)
        PIROS_PARTI       : ±2 GP/defender  (declarer nets ±4)
        NEGYVENSZAZ       : ±4 GP/defender  (declarer nets ±8)
        PIROS_NEGYVENSZAZ : ±8 GP/defender  (declarer nets ±16)

        Silent bid bonuses:
        Each bonus is 50% of the equivalent explicit bid's GP value and stacks.
        Both DECLARER and DEFENDERS can earn silent bonuses independently.

        Declarer silent 40-100: Parti/Piros Parti licits only (circular for explicit 40-100).
            Requires: declarer holds trump pair + trick_scores[0] >= 60 + declarer wins.
            PARTI + silent 40-100:       +2 GP/defender  → declarer nets +6
            PIROS_PARTI + silent 40-100: +4 GP/defender  → declarer nets +12

        Declarer silent Ulti: applies to Parti AND 40-100 licits — win conditions
            do not interfere with each other.
            Requires: declarer wins the last trick by playing the trump 7 + declarer wins.
            Non-hearts: +2 GP/defender bonus (50% of explicit Ulti 4 GP/def).
            Hearts:     +4 GP/defender bonus (50% of explicit Piros Ulti 8 GP/def).

        Defender silent Ulti: PARTI and 40-100 licits (not ULTI/NEGYVENSZAZ_ULTI where
            Ulti is an explicit component — defender winning there already negates it).
            Requires: either defender wins the last trick by playing the trump 7.
            Non-hearts: −2 GP/defender off declarer's reward (defenders earn +2 GP each).
            Hearts:     −4 GP/defender off declarer's reward (defenders earn +4 GP each).

        Defender silent 40-100: PARTI family licits (PARTI, PIROS_PARTI, ULTI, PIROS_ULTI).
            Circular for explicit 40-100 licits (defenders don't announce there).
            Requires: either defender announces trump pair + defenders' combined
            trick_scores[1]+trick_scores[2] >= 60.
            Non-hearts: −2 GP/defender off declarer's reward.
            Hearts:     −4 GP/defender off declarer's reward.

        All bonuses stack when their conditions hold simultaneously.

        Ulti (composite bid, evaluated independently per component):
            Ulti component  (4 or 8 GP/defender): won iff _declarer_ulti.
            Parti component (1 or 2 GP/defender): won iff conventional point majority.
            Declarer silent 40-100 bonus applies when the Parti component is won.

        40-100+Ulti (composite bid, evaluated independently per component):
            Ulti component  (4 or 8 GP/defender): won iff _declarer_ulti.
            40-100 component (4 or 8 GP/defender): won iff trick_scores[0] >= 60.
            Full win: +8 (or +16) GP/def; partial (one won): 0; full loss: −8 (or −16).
        """
        if self.phase != Phase.DONE:
            return (0, 0, 0)

        if self.licit in _NEGYVENSZAZ_ULTI_LICITS:
            mul = 2 if self.trump_suit == 'hearts' else 1
            # Each component worth 4*mul GP/defender, evaluated independently
            ulti_gp = 4 * mul if self._declarer_ulti        else -4 * mul
            negy_gp = 4 * mul if self.trick_scores[0] >= 60 else -4 * mul
            total = ulti_gp + negy_gp
            return (2 * total, -total, -total)

        if self.licit in _ULTI_LICITS:
            parti_mul = 2 if self.trump_suit == 'hearts' else 1
            ulti_mul  = 8 if self.trump_suit == 'hearts' else 4
            pay        = self.payoffs()
            parti_wins = pay[0] > pay[1] + pay[2] + self._talon_points
            ulti_gp    = ulti_mul  if self._declarer_ulti else -ulti_mul
            parti_gp   = parti_mul if parti_wins           else -parti_mul
            # Declarer silent 40-100: Parti subgame won + trump pair + ≥60 trick pts
            decl_bonus = 0
            if parti_wins and self._declarer_has_trump_pair and self.trick_scores[0] >= 60:
                decl_bonus = 2 * parti_mul
            # Defender silent 40-100: Parti subgame → both sides can earn it
            def_bonus = 0
            if self._defender_has_trump_pair and self.trick_scores[1] + self.trick_scores[2] >= 60:
                def_bonus = 2 * parti_mul
            gp_per_def = ulti_gp + parti_gp + decl_bonus - def_bonus
            return (2 * gp_per_def, -gp_per_def, -gp_per_def)

        mul = {
            Licit.PARTI:             1,
            Licit.PIROS_PARTI:       2,
            Licit.NEGYVENSZAZ:       4,
            Licit.PIROS_NEGYVENSZAZ: 8,
        }.get(self.licit, 1)
        # Silent Ulti value: 50% of explicit Ulti (4 GP/def non-hearts, 8 GP/def hearts)
        ulti_silent = 2 * (2 if self.trump_suit == 'hearts' else 1)

        decl_wins = self.declarer_wins()

        decl_bonus = 0
        if decl_wins:
            if self.licit not in _NEGYVENSZAZ_LICITS:
                # Declarer silent 40-100 only for Parti (circular for explicit 40-100)
                if self._declarer_has_trump_pair and self.trick_scores[0] >= 60:
                    decl_bonus += 2 * mul
            if self._declarer_ulti:
                decl_bonus += ulti_silent   # Piros-adjusted silent Ulti

        def_bonus = 0
        # Defender silent Ulti: applies to Parti and 40-100 licits.
        # For explicit Ulti/40-100+Ulti licits, the outcome is already embedded in
        # _declarer_ulti so no additional adjustment is needed.
        if self._defender_ulti:
            def_bonus += ulti_silent
        # Defender silent 40-100: Parti licits only (circular for 40-100 licits)
        if self.licit not in _NEGYVENSZAZ_LICITS:
            if (self._defender_has_trump_pair
                    and self.trick_scores[1] + self.trick_scores[2] >= 60):
                def_bonus += 2 * mul

        gp_per_def = (mul if decl_wins else -mul) + decl_bonus - def_bonus
        return (2 * gp_per_def, -gp_per_def, -gp_per_def)

    def talon_points(self) -> int:
        """Points in the talon (count for defenders). Meaningful only after DONE."""
        return self._talon_points

    def declarer_wins(self) -> Optional[bool]:
        """
        None if not over.

        40-100: True if trick_scores[0] >= 60 (aces + tens + last-trick bonus).
        Others: True if declarer's total points strictly exceed defenders' total
                (defenders' total includes the talon).
        """
        if self.phase != Phase.DONE:
            return None
        if self.licit in _NEGYVENSZAZ_ULTI_LICITS:
            return self._declarer_ulti and self.trick_scores[0] >= 60
        if self.licit in _NEGYVENSZAZ_LICITS:
            return self.trick_scores[0] >= 60
        if self.licit in _ULTI_LICITS:
            return self._declarer_ulti   # primary win condition for Ulti
        pay = self.payoffs()
        return pay[0] > pay[1] + pay[2] + self._talon_points

    def is_over(self) -> bool:
        return self.phase == Phase.DONE

    @property
    def current_player(self) -> int:
        return self.current_player_id

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _has_pair(self, player_id: int, suit: str) -> bool:
        """Return True if player_id holds both king and upper of the given suit."""
        ranks = {c.rank for c in self.hands[player_id] if c.suit == suit}
        return 'king' in ranks and 'upper' in ranks

    def _has_seven(self, player_id: int, suit: str) -> bool:
        """Return True if player_id holds the 7 (rank_index==0) of the given suit."""
        return any(c.suit == suit and c.rank_index == 0
                   for c in self.hands[player_id])

    def _do_trump(self, suit_id: int) -> None:
        """
        Declare trump suit and licit (first action of every episode).

        Action encoding:
            0-3  → Parti / Piros Parti (suit = suit_id % 4)
            4-7  → 40-100 / Piros 40-100 (suit = suit_id - 4)

        Transitions to DISCARD; king-upper bonuses computed after discard so
        only cards still in hand are counted.
        """
        actual_suit_id = suit_id % 4
        self.trump_suit = SUITS[actual_suit_id]
        if suit_id >= 12:  # 40-100+Ulti family
            self.licit = (Licit.PIROS_NEGYVENSZAZ_ULTI
                          if actual_suit_id == HEARTS_ID
                          else Licit.NEGYVENSZAZ_ULTI)
        elif suit_id >= 8:  # Ulti family
            self.licit = (Licit.PIROS_ULTI
                          if actual_suit_id == HEARTS_ID
                          else Licit.ULTI)
        elif suit_id >= 4:  # 40-100 family
            self.licit = (Licit.PIROS_NEGYVENSZAZ
                          if actual_suit_id == HEARTS_ID
                          else Licit.NEGYVENSZAZ)
        else:               # Parti family
            self.licit = (Licit.PIROS_PARTI
                          if actual_suit_id == HEARTS_ID
                          else Licit.PARTI)
        self.phase = Phase.DISCARD

    def _hand_array(self, pid: int) -> np.ndarray:
        arr = np.zeros(32, dtype=np.float32)
        for c in self.hands[pid]:
            arr[c.id] = 1.0
        return arr

    def pop_last_trick(self) -> List[Tuple[int, int]]:
        """Return and clear the most recently completed trick's play log."""
        plays = self.last_trick_plays
        self.last_trick_plays = []
        return plays

    def _do_discard(self, card_id: int) -> None:
        card = card_from_id(card_id)
        self.hands[0].remove(card)
        self.talon.append(card)
        self.talon_card_ids.append(card_id)
        if len(self.talon) == 2:
            self._compute_king_upper_points()
            self.phase = Phase.PLAY
            self.current_player_id = 0
            self.trick_leader = 0
            self.play_start_hands = [self._hand_array(pid) for pid in range(3)]

    def _compute_king_upper_points(self) -> None:
        """
        Award king-upper bonuses for the DECLARER and populate _announced_pairs[0].
        Called at the end of DISCARD, before trick 1.

        40-100 family (NEGYVENSZAZ / PIROS_NEGYVENSZAZ / NEGYVENSZAZ_ULTI / PIROS_NEGYVENSZAZ_ULTI):
          - Only the declarer announces, and only the trump pair counts (40 pts).
          - Defenders do not announce or score pairs in 40-100 games.
          - _defender_has_trump_pair stays False throughout the game.

        Parti family (PARTI / PIROS_PARTI / ULTI / PIROS_ULTI):
          - Declarer announces ALL their pairs here (start of PLAY, before leading trick 1).
          - Defender pairs are announced progressively during trick 1 via _announce_pairs().
          - _defender_has_trump_pair starts False; updated when defenders announce.
        """
        trump_suit_idx = SUITS.index(self.trump_suit)

        if self.licit in _NEGYVENSZAZ_FAMILY:
            # 40-100 family: declarer announces trump pair only
            if self._has_pair(0, self.trump_suit):
                self._announced_pairs[0][trump_suit_idx] = 1
                self.king_upper_points[0] = 40
            self._declarer_has_trump_pair = bool(self._announced_pairs[0][trump_suit_idx])
            self._defender_has_trump_pair = False
        else:
            # Parti family: declarer announces all their pairs
            ranks_by_suit: Dict[str, set] = {s: set() for s in SUITS}
            for c in self.hands[0]:
                ranks_by_suit[c.suit].add(c.rank)
            for suit_idx, suit in enumerate(SUITS):
                if 'king' in ranks_by_suit[suit] and 'upper' in ranks_by_suit[suit]:
                    self._announced_pairs[0][suit_idx] = 1
                    pts = 40 if suit_idx == trump_suit_idx else 20
                    self.king_upper_points[0] += pts
            self._declarer_has_trump_pair = bool(self._announced_pairs[0][trump_suit_idx])
            self._defender_has_trump_pair = False  # set later in _announce_pairs()

    def _announce_pairs(self, pid: int) -> None:
        """
        Announce all king-upper pairs for a defender at the start of their first
        turn in trick 1 (Parti-family games only).

        Updates _announced_pairs[pid], king_upper_points[pid], and sets
        _defender_has_trump_pair if the defender holds the trump pair.
        """
        trump_suit_idx = SUITS.index(self.trump_suit)
        ranks_by_suit: Dict[str, set] = {s: set() for s in SUITS}
        for c in self.hands[pid]:
            ranks_by_suit[c.suit].add(c.rank)
        for suit_idx, suit in enumerate(SUITS):
            if 'king' in ranks_by_suit[suit] and 'upper' in ranks_by_suit[suit]:
                self._announced_pairs[pid][suit_idx] = 1
                pts = 40 if suit_idx == trump_suit_idx else 20
                self.king_upper_points[pid] += pts
        if self._announced_pairs[pid][trump_suit_idx]:
            self._defender_has_trump_pair = True

    def _legal_play_actions(self) -> List[int]:
        """
        Über rule:
          Follow suit if possible:
            - Can you beat the overall trick winner (highest trump, or highest of led suit)?
              → must play higher; otherwise play any same-suit card.
            A same-suit card can never beat a trump (unless led suit IS trump).
          No same suit → must play trump if held:
            - Trump already in trick? → must über the highest trump if possible.
            - No trump in trick yet? → any trump wins; play any.
          No same suit, no trump → play any card.

        Hetesre tartás: when the licit includes Ulti, the declarer may not play
        the trump 7 unless it is the only legal card.
        """
        hand = self.hands[self.current_player_id]

        if not self.current_trick:
            candidates = [c.id for c in hand]
        else:
            led_suit = self.current_trick[0][1].suit
            same_suit = [c for c in hand if c.suit == led_suit]

            trump_plays = [c for _, c in self.current_trick if c.suit == self.trump_suit]
            if trump_plays:
                winning_rank = max(c.rank_index for c in trump_plays)
                winning_is_trump = True
            else:
                winning_rank = max(
                    c.rank_index for _, c in self.current_trick if c.suit == led_suit
                )
                winning_is_trump = False

            if same_suit:
                # Must follow suit. A non-trump same-suit card can never beat a trump.
                if winning_is_trump and led_suit != self.trump_suit:
                    candidates = [c.id for c in same_suit]
                else:
                    higher = [c for c in same_suit if c.rank_index > winning_rank]
                    candidates = [c.id for c in (higher if higher else same_suit)]
            else:
                trump_cards = [c for c in hand if c.suit == self.trump_suit]
                if trump_cards:
                    if winning_is_trump:
                        higher_trumps = [c for c in trump_cards if c.rank_index > winning_rank]
                        candidates = [c.id for c in (higher_trumps if higher_trumps else trump_cards)]
                    else:
                        candidates = [c.id for c in trump_cards]
                else:
                    candidates = [c.id for c in hand]

        # Hetesre tartás: declarer cannot play trump 7 in an Ulti licit
        # unless it is the only legal card.
        if (self.current_player_id == self.DECLARER_ID
                and self.licit in _ALL_ULTI_LICITS
                and self.trump_suit is not None
                and len(candidates) > 1):
            trump_7_id = SUITS.index(self.trump_suit) * 8  # rank_index 0 = '7'
            candidates = [c for c in candidates if c != trump_7_id]

        return candidates

    def _do_play(self, card_id: int) -> None:
        card = card_from_id(card_id)
        pid  = self.current_player_id
        self.hands[pid].remove(card)
        self.current_trick.append((pid, card))

        if len(self.current_trick) > 1:
            led_suit = self.current_trick[0][1].suit
            if card.suit == led_suit:
                # Über constraint: if the player couldn't beat the current best,
                # they provably hold no higher card of this suit.
                prev_max = max(
                    c.rank_index for _, c in self.current_trick[:-1]
                    if c.suit == led_suit
                )
                if card.rank_index <= prev_max:
                    s_idx = SUITS.index(led_suit)
                    self._player_suit_uppers[pid][s_idx] = min(
                        self._player_suit_uppers[pid][s_idx], prev_max
                    )
            else:
                self._player_voids[pid].add(led_suit)
                if self.trump_suit and card.suit != self.trump_suit:
                    self._player_voids[pid].add(self.trump_suit)

        if len(self.current_trick) == 3:
            self._resolve_trick()
        else:
            next_pid = (self.current_player_id + 1) % 3
            # In trick 1, Parti family: each defender announces their pairs
            # before playing their first card.
            if (self.trick_number == 0
                    and self.licit not in _NEGYVENSZAZ_FAMILY
                    and next_pid in (1, 2)):
                self._announce_pairs(next_pid)
            self.current_player_id = next_pid

    def _resolve_trick(self) -> None:
        """
        Determine the trick winner, award points, and set up the next trick.
        """
        # Snapshot completed trick for reward-shaping bonuses BEFORE clearing
        self.last_trick_plays = [(pid, card.id) for pid, card in self.current_trick]
        trump_plays = [
            (pid, c)
            for pid, c in self.current_trick
            if c.suit == self.trump_suit
        ]

        if trump_plays:
            winner_pid = max(trump_plays, key=lambda x: x[1].rank_index)[0]
        else:
            led_suit = self.current_trick[0][1].suit
            led_plays = [
                (pid, c)
                for pid, c in self.current_trick
                if c.suit == led_suit
            ]
            winner_pid = max(led_plays, key=lambda x: x[1].rank_index)[0]

        for _, card in self.current_trick:
            self._played_cards[card.id] = 1

        for _, card in self.current_trick:
            self.trick_scores[winner_pid] += card.points

        # Per-player captured cards + per-trick leader/winner history.
        leader_pid = self.current_trick[0][0]
        for _, card in self.current_trick:
            self.captured_cards[winner_pid].append(card)
        self.trick_history_leaders.append(leader_pid)
        self.trick_history_winners.append(winner_pid)

        self.trick_number += 1

        self.last_trick_winner = winner_pid

        if self.trick_number == self.NUM_TRICKS:
            # Ulti check: who wins the last trick with the trump 7?
            winner_card = next(c for pid, c in self.current_trick if pid == winner_pid)
            if winner_card.suit == self.trump_suit and winner_card.rank_index == 0:
                if winner_pid == 0:
                    self._declarer_ulti = True
                else:
                    self._defender_ulti = True
            self.trick_scores[winner_pid] += 10
            self._talon_points = sum(c.points for c in self.talon)
            self.phase = Phase.DONE
        else:
            self.current_player_id = winner_pid
            self.trick_leader = winner_pid

        self.current_trick = []
