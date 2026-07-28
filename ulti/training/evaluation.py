"""
Evaluation helpers for play and pre-game agents, plus the Evaluator class.

evaluate_play / evaluate_pregame
    Episode-level evaluation helpers; both use game_points()[0] > 0 as win
    criterion so the result is correct across all licits including silent bids.

print_play_results / print_pregame_results
    Human-readable summary printers used at the end of training.

Evaluator, MatchupStats, AgentStats
    Structured round-robin evaluation harness (originally in root evaluation.py).
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from ulti.game import UltiGame, Licit
from ulti.card import SUITS
from .registry import DEFENDERS, DEFENDER_POOL, make_pregame_selector


# ──────────────────────────────────────────────────────────────────────────────
# Episode-level evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_play(
    play_model,
    play_env_cls,
    heuristic_selector: Callable,
    defender_cls,
    n_episodes: int = 500,
    trump_selector: Optional[Callable] = None,
    pregame_agent=None,
) -> dict:
    """Run the play agent against one defender type; return metrics dict.

    trump_selector defaults to heuristic_selector when None.  Pass the
    pre-game agent's selector to evaluate on pre-game-approved hands only.

    Win criterion: game_points()[0] > 0 — correct across all licits including
    composite ones and games decided by silent bid bonuses.
    """
    env = play_env_cls(
        trump_selector=trump_selector or heuristic_selector,
        pregame_agent=pregame_agent,
        defender1=defender_cls().act,
        defender2=defender_cls().act,
    )
    wins, total_pts, total_gp = 0, 0.0, 0.0
    decl_ulti, def_ulti = 0, 0
    decl_s40100, def_s40100 = 0, 0
    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            mask = info['action_mask'].astype(bool)
            action, _ = play_model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        g = env.game
        wins       += int(g.game_points()[0] > 0)
        total_pts  += g.payoffs()[0]
        total_gp   += g.game_points()[0]
        decl_ulti  += int(g._declarer_ulti)
        def_ulti   += int(g._defender_ulti)
        decl_s40100 += int(g._declarer_has_trump_pair and g.trick_scores[0] >= 60)
        def_s40100  += int(g._defender_has_trump_pair and
                           g.trick_scores[1] + g.trick_scores[2] >= 60)
    return {
        'win_rate':          wins        / n_episodes,
        'avg_pts':           total_pts   / n_episodes,
        'avg_gp':            total_gp    / n_episodes,
        'decl_silent_ulti':  decl_ulti   / n_episodes,
        'def_silent_ulti':   def_ulti    / n_episodes,
        'decl_silent_40100': decl_s40100 / n_episodes,
        'def_silent_40100':  def_s40100  / n_episodes,
    }


def evaluate_pregame(
    pregame_model,
    play_model,
    pregame_env_cls,
    defender_cls,
    trump_lo: int,
    n_episodes: int = 500,
) -> dict:
    """Run the pre-game agent (deterministic) against one defender type.

    Drives all 3 pre-game steps (TRUMP, DISCARD1, DISCARD2) then reads the
    reward returned by the last step (which includes the internally-run PLAY).
    """
    env = pregame_env_cls(
        play_agent=play_model,
        defender1=defender_cls().act,
        defender2=defender_cls().act,
    )
    passz_count  = 0
    suit_counts  = [0] * 4
    wins         = 0
    total_reward = 0.0

    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done             = False
        ep_reward        = 0.0
        trump_action_raw = None
        step             = 0
        while not done:
            mask = info['action_mask'].astype(bool)
            action, _ = pregame_model.predict(obs, action_masks=mask, deterministic=True)
            if step == 0:
                trump_action_raw = int(action)
            obs, reward, done, _, info = env.step(int(action))
            ep_reward += reward
            step      += 1
        total_reward += ep_reward
        if env._passz_chosen:
            passz_count += 1
        else:
            suit = trump_action_raw - trump_lo   # 0-3
            suit_counts[suit] += 1
            if env.game.game_points()[0] > 0:
                wins += 1

    play_count = n_episodes - passz_count
    return {
        'passz_rate': passz_count / n_episodes,
        'avg_reward': total_reward / n_episodes,
        'win_rate':   wins / play_count if play_count > 0 else 0.0,
        'suit_pct':   {SUITS[i]: suit_counts[i] / n_episodes for i in range(4)},
        'piros_pct':  suit_counts[2] / n_episodes,
    }


def _mean(results: dict, key: str) -> float:
    return sum(r[key] for r in results.values()) / len(results)


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_play_results(
    play_model,
    play_env_cls,
    heuristic_selector: Callable,
    n_episodes: int = 500,
    pregame_model=None,
    trump_lo: int = 0,
    trump_hi: int = 3,
    adversarial_defender_cls=None,
    silent_ulti: bool = False,
    silent_40100: bool = False,
) -> None:
    selector = (
        make_pregame_selector(pregame_model, trump_lo, trump_hi)
        if pregame_model else None
    )
    if adversarial_defender_cls is not None:
        r = evaluate_play(play_model, play_env_cls, heuristic_selector,
                          adversarial_defender_cls, n_episodes,
                          trump_selector=selector, pregame_agent=pregame_model)
        mean_gp     = r['avg_gp']
        mean_wr     = r['win_rate']
        mean_pts    = r['avg_pts']
        decl_s_ulti = r['decl_silent_ulti']
        def_s_ulti  = r['def_silent_ulti']
        decl_s40100 = r['decl_silent_40100']
        def_s40100  = r['def_silent_40100']
    else:
        results = {
            name: evaluate_play(play_model, play_env_cls, heuristic_selector,
                                cls, n_episodes,
                                trump_selector=selector, pregame_agent=pregame_model)
            for name, cls in DEFENDERS.items()
        }
        mean_gp     = _mean(results, 'avg_gp')
        mean_wr     = _mean(results, 'win_rate')
        mean_pts    = _mean(results, 'avg_pts')
        decl_s_ulti = _mean(results, 'decl_silent_ulti')
        def_s_ulti  = _mean(results, 'def_silent_ulti')
        decl_s40100 = _mean(results, 'decl_silent_40100')
        def_s40100  = _mean(results, 'def_silent_40100')
    ulti_tag     = 'sulti' if silent_ulti else 'ulti'
    def_ulti_str = f'  def_{ulti_tag}={def_s_ulti:.1%}' if silent_ulti else ''
    s40100_str   = (f'  decl_s40100={decl_s40100:.1%}  def_s40100={def_s40100:.1%}'
                    if silent_40100 else '')
    print(f"  mean_gp={mean_gp:+.2f}  win%={mean_wr*100:.1f}  avg_pts={mean_pts:.1f}"
          f"  decl_{ulti_tag}={decl_s_ulti:.1%}{def_ulti_str}{s40100_str}")


def print_pregame_results(
    pregame_model,
    play_model,
    pregame_env_cls,
    trump_lo: int,
    n_episodes: int = 500,
    adversarial_defender_cls=None,
) -> None:
    if adversarial_defender_cls is not None:
        r = evaluate_pregame(pregame_model, play_model, pregame_env_cls,
                             adversarial_defender_cls, trump_lo, n_episodes)
        mean_rew   = r['avg_reward']
        mean_passz = r['passz_rate']
        mean_wr    = r['win_rate']
        mean_piros = r['piros_pct']
        suits_str  = '  '.join(f"{s}:{r['suit_pct'][s]:.1%}" for s in SUITS)
    else:
        results = {
            name: evaluate_pregame(pregame_model, play_model, pregame_env_cls,
                                   cls, trump_lo, n_episodes)
            for name, cls in DEFENDERS.items()
        }
        mean_rew   = _mean(results, 'avg_reward')
        mean_passz = _mean(results, 'passz_rate')
        mean_wr    = _mean(results, 'win_rate')
        mean_piros = _mean(results, 'piros_pct')
        suit_means = {
            s: sum(r['suit_pct'][s] for r in results.values()) / len(results)
            for s in SUITS
        }
        suits_str = '  '.join(f"{s}:{suit_means[s]:.1%}" for s in SUITS)
    print(f"  passz={mean_passz:.1%}  win%={mean_wr*100:.1f}  "
          f"piros={mean_piros:.1%}  mean_rew={mean_rew:+.1f}  [{suits_str}]")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator — round-robin tournament harness
# ──────────────────────────────────────────────────────────────────────────────

# Type for agent factory (class or zero-arg callable returning an agent).
AgentFactory = Callable[[], object]


@dataclass
class MatchupStats:
    """Accumulated statistics for one (declarer_name, def1_name, def2_name) triple."""
    declarer: str
    defender1: str
    defender2: str
    games: int = 0
    declarer_wins: int = 0
    declarer_points: float = 0.0        # sum of trick_scores[0] + king_upper_points[0]
    trump_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sanity_errors: int = 0              # games where trick_scores sum != 90
    king_upper_declared: int = 0        # games where any player had a king-upper pair
    licit_games: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    licit_wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class AgentStats:
    """Aggregated statistics for one agent across all roles it played."""
    name: str
    games_as_declarer: int = 0
    wins_as_declarer: int = 0
    total_declarer_points: float = 0.0
    games_as_defender: int = 0         # combined defender1 + defender2
    trump_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    king_upper_declared: int = 0        # games (as declarer) where any player had a pair
    licit_games: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    licit_wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def win_rate(self) -> Optional[float]:
        if self.games_as_declarer == 0:
            return None
        return self.wins_as_declarer / self.games_as_declarer

    @property
    def avg_declarer_points(self) -> Optional[float]:
        if self.games_as_declarer == 0:
            return None
        return self.total_declarer_points / self.games_as_declarer

    @property
    def king_upper_rate(self) -> Optional[float]:
        if self.games_as_declarer == 0:
            return None
        return self.king_upper_declared / self.games_as_declarer


class Evaluator:
    """
    Round-robin evaluator for Ulti agents.

    Parameters
    ----------
    agents : dict[str, type | callable]
        Mapping of name → agent class (or zero-arg factory).
        Each factory is called fresh for every game.
    games_per_matchup : int
        Number of games per (declarer, def1, def2) ordered triple.
    seed : int, optional
        Base RNG seed. Each game gets a derived seed for reproducibility.
    """

    def __init__(
        self,
        agents: Dict[str, AgentFactory],
        games_per_matchup: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        self.agents = agents
        self.games_per_matchup = games_per_matchup
        self.seed = seed
        self._matchup_stats: List[MatchupStats] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, AgentStats]:
        """
        Run all matchups and return per-agent aggregated stats.

        Returns
        -------
        dict[str, AgentStats]
            One entry per agent name.
        """
        names = list(self.agents.keys())
        agent_stats: Dict[str, AgentStats] = {n: AgentStats(name=n) for n in names}
        self._matchup_stats = []

        global_game_id = 0
        for dec_name, def1_name, def2_name in itertools.product(names, repeat=3):
            ms = MatchupStats(declarer=dec_name, defender1=def1_name, defender2=def2_name)
            for _ in range(self.games_per_matchup):
                seed = None if self.seed is None else (self.seed + global_game_id)
                self._play_game(dec_name, def1_name, def2_name, ms, seed)
                global_game_id += 1
            self._matchup_stats.append(ms)

            # Aggregate into per-agent stats
            as_dec = agent_stats[dec_name]
            as_dec.games_as_declarer += ms.games
            as_dec.wins_as_declarer  += ms.declarer_wins
            as_dec.total_declarer_points += ms.declarer_points
            for suit, cnt in ms.trump_counts.items():
                as_dec.trump_counts[suit] += cnt
            as_dec.king_upper_declared += ms.king_upper_declared
            for licit_name, cnt in ms.licit_games.items():
                as_dec.licit_games[licit_name] += cnt
            for licit_name, cnt in ms.licit_wins.items():
                as_dec.licit_wins[licit_name] += cnt

            for def_name in (def1_name, def2_name):
                agent_stats[def_name].games_as_defender += ms.games

        return agent_stats

    def print_report(self, agent_stats: Dict[str, AgentStats]) -> None:
        """Print a human-readable summary to stdout."""
        sep = "─" * 70
        print(sep)
        print(f"{'Agent':<20} {'Dec.games':>10} {'Win%':>8} {'Avg pts':>9} {'Def.games':>10}")
        print(sep)
        for name, st in agent_stats.items():
            wr  = f"{st.win_rate * 100:.1f}%" if st.win_rate is not None else "  N/A"
            avg = f"{st.avg_declarer_points:.1f}" if st.avg_declarer_points is not None else " N/A"
            print(f"{name:<20} {st.games_as_declarer:>10} {wr:>8} {avg:>9} {st.games_as_defender:>10}")
        print(sep)

        # Trump distribution
        print("\nTrump suit distribution (as declarer):")
        for name, st in agent_stats.items():
            total = sum(st.trump_counts.values())
            if total == 0:
                continue
            dist = "  ".join(
                f"{s}: {st.trump_counts.get(s, 0) / total * 100:.0f}%"
                for s in SUITS
            )
            print(f"  {name:<18} {dist}")

        # King-upper declaration rate
        print("\nKing-upper declaration rate (as declarer):")
        for name, st in agent_stats.items():
            if st.games_as_declarer == 0:
                continue
            rate = st.king_upper_declared / st.games_as_declarer * 100
            print(f"  {name:<18} {rate:.1f}%  ({st.king_upper_declared}/{st.games_as_declarer} games)")

        # Licit distribution and win rates
        print("\nLicit distribution and win rate (as declarer):")
        licit_order = [
            'PARTI', 'PIROS_PARTI',
            'NEGYVENSZAZ', 'PIROS_NEGYVENSZAZ',
            'ULTI', 'PIROS_ULTI',
            'NEGYVENSZAZ_ULTI', 'PIROS_NEGYVENSZAZ_ULTI',
        ]
        for name, st in agent_stats.items():
            if st.games_as_declarer == 0:
                continue
            parts = []
            for licit_name in licit_order:
                g = st.licit_games.get(licit_name, 0)
                w = st.licit_wins.get(licit_name, 0)
                if g == 0:
                    parts.append(f"{licit_name}: 0%")
                else:
                    pct = g / st.games_as_declarer * 100
                    wr  = w / g * 100
                    parts.append(f"{licit_name}: {pct:.0f}% played, {wr:.1f}% win")
            print(f"  {name:<18} {'  |  '.join(parts)}")

        # Sanity errors
        total_errors = sum(ms.sanity_errors for ms in self._matchup_stats)
        if total_errors:
            print(f"\nWARNING: {total_errors} sanity error(s) detected (trick_scores sum != 90)!")
        else:
            print("\nSanity check passed: all games scored correctly.")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _make_agent(self, name: str, seed: Optional[int]):
        """Instantiate an agent, passing a seed if the factory accepts one."""
        try:
            return self.agents[name](seed=seed)
        except TypeError:
            return self.agents[name]()

    def _play_game(
        self,
        dec_name: str,
        def1_name: str,
        def2_name: str,
        ms: MatchupStats,
        seed: Optional[int],
    ) -> None:
        """Play one game; update matchup stats in place."""
        game = UltiGame()
        game.reset(seed=seed)

        # Instantiate agents fresh for each game; pass derived seeds when available
        agents = [
            self._make_agent(dec_name,  None if seed is None else seed * 3),
            self._make_agent(def1_name, None if seed is None else seed * 3 + 1),
            self._make_agent(def2_name, None if seed is None else seed * 3 + 2),
        ]

        done = False
        while not done:
            pid = game.current_player_id
            obs = game.get_observation(pid)
            obs["player_id"] = pid    # for adapters that need perspective info
            action = agents[pid].act(obs)
            _, done = game.step(action)

        # Collect stats
        ms.games += 1
        won = game.game_points()[0] > 0
        if won:
            ms.declarer_wins += 1
        ms.declarer_points += game.payoffs()[0]
        if game.trump_suit is not None:
            ms.trump_counts[game.trump_suit] += 1

        # King-upper: any player declared a pair this game
        if any(p > 0 for p in game.king_upper_points):
            ms.king_upper_declared += 1

        # Licit distribution and win rate
        if game.licit is not None:
            licit_name = game.licit.name  # 'PARTI' or 'PIROS_PARTI'
            ms.licit_games[licit_name] += 1
            if won:
                ms.licit_wins[licit_name] += 1

        # Sanity: total trick points (including talon) must equal 90
        total = sum(game.trick_scores) + game.talon_points()
        if total != 90:
            ms.sanity_errors += 1
# run_round_robin (registry-aware entry point) lives in arena/__init__.py.
