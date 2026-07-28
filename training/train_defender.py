"""
Alternating adversarial training for a defender and declarer in any subgame.

Both agents co-evolve in B→A turns (freeze one, train the other, swap):

  Phase B — Defender agent
    One agent plays both defenders against a frozen declarer.
    Obs: 200-dim (same layout as declarer).  Action: Discrete(32).
    Reward: −(delta of declarer's trick payoff) − (game_points()[0]×10).

  Phase A — Declarer / play agent fine-tuning
    The play agent trains against a blended defender pool (75% adversarial +
    25% heuristic) to prevent catastrophic forgetting of general play while
    maintaining adversarial pressure.  TRUMP + DISCARD are driven by the
    pre-game agent (composite) or the play agent restricted to [trump_lo,
    trump_hi] (Parti).

Both phases use DummyVecEnv (in-process) because the opposing agent is not
picklable for SubprocVecEnv.

Subgame    --subgame   TRUMP range   --load-pregame required?
---------  ----------  ------------  ----------------------
Parti      parti       0–3           No
40-100     40100       4–7           Yes
Ulti       ulti        8–11          Yes
40-100+U   ulti40100   12–15         Yes

Typical usage
-------------
    # Parti (first run)
    python train_defender.py --subgame parti --load-declarer models/parti_play_best --save models/defender_parti

    # 40-100 (requires pre-game agent)
    python train_defender.py --subgame 40100 --load-declarer models/40100_play_best --load-pregame models/40100_pregame_best --save models/defender_40100

    # Continue from checkpoints
    python train_defender.py --subgame ulti --load-declarer models/defender_ulti_declarer_best --load-pregame models/ulti_pregame_best --load-defender models/defender_ulti_agent_best --save models/defender_ulti

    # Evaluate only
    python train_defender.py --subgame ulti40100 --eval-only --load-declarer models/defender_ulti40100_declarer_best --load-pregame models/ulti40100_pregame_best --load-defender models/defender_ulti40100_agent_best
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base import UltiDeclarerEnv
from envs.defender import DefenderEnv
from ulti.game import Phase, _NEGYVENSZAZ_LICITS
from agents.heuristic import RandomAgent, GreedyAgent, ConservativeAgent, SmartAgent
from training.registry import make_pregame_selector, DEFENDERS, DEFENDER_POOL, make_adversarial_cls
from training.vectors import build_pregame_vec
from training.callbacks import PreGameCallback


# ──────────────────────────────────────────────────────────────────────────────
# Subgame registry
# ──────────────────────────────────────────────────────────────────────────────

def _get_subgame(name: str) -> dict:
    """
    Return subgame-specific config and lazily-imported classes/functions.

    Keys
    ----
    trump_lo, trump_hi    Inclusive TRUMP bid range.
    pregame_env_cls       PreGameEnv for Phase C recalibration; None for Parti.
    play_env_cls          PlayEnv for Phase A; None → use UltiDeclarerEnv.
    heuristic_selector    Fallback trump selector; None for Parti.
    """
    if name == 'parti':
        return dict(
            trump_lo=0, trump_hi=3,
            pregame_env_cls=None, play_env_cls=None, heuristic_selector=None,
        )
    if name == '40100':
        from envs.subgames import (NegyvenszazPlayEnv, NegyvenszazPreGameEnv,
                                    heuristic_trump_selector)
        return dict(
            trump_lo=4, trump_hi=7,
            pregame_env_cls=NegyvenszazPreGameEnv,
            play_env_cls=NegyvenszazPlayEnv,
            heuristic_selector=heuristic_trump_selector,
        )
    if name == 'ulti':
        from envs.subgames import (UltiPlayEnv, UltiPreGameEnv,
                                    heuristic_ulti_selector)
        return dict(
            trump_lo=8, trump_hi=11,
            pregame_env_cls=UltiPreGameEnv,
            play_env_cls=UltiPlayEnv,
            heuristic_selector=heuristic_ulti_selector,
        )
    if name == 'ulti40100':
        from envs.subgames import (UltiNegyvenszazPlayEnv, UltiNegyvenszazPreGameEnv,
                                    heuristic_combo_selector)
        return dict(
            trump_lo=12, trump_hi=15,
            pregame_env_cls=UltiNegyvenszazPreGameEnv,
            play_env_cls=UltiNegyvenszazPlayEnv,
            heuristic_selector=heuristic_combo_selector,
        )
    raise ValueError(f'Unknown subgame: {name!r}')


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_defender(
    defender_model: MaskablePPO,
    play_model: MaskablePPO,
    n_episodes: int = 500,
    trump_selector=None,
    trump_lo: int = 0,
    trump_hi: int = 3,
    pregame_agent=None,
) -> dict:
    """
    Run the defender (both seats) against the frozen declarer.

    Metrics are from the declarer's perspective: lower declarer_win_rate /
    lower avg_declarer_gp means the defender is doing better.
    """
    env = DefenderEnv(play_model, trump_selector=trump_selector,
                      pregame_agent=pregame_agent,
                      trump_lo=trump_lo, trump_hi=trump_hi)
    wins, total_gp, total_pts = 0, 0.0, 0.0
    def_silent_ulti_count   = 0
    def_silent_40100_count  = 0

    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            mask = info['action_mask'].astype(bool)
            action, _ = defender_model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        gp         = env.game.game_points()[0]
        wins      += int(gp > 0)
        total_gp  += gp
        total_pts += env.game.payoffs()[0]
        if env.game._defender_ulti:
            def_silent_ulti_count += 1
        if (env.game.licit not in _NEGYVENSZAZ_LICITS
                and env.game._defender_has_trump_pair
                and env.game.trick_scores[1] + env.game.trick_scores[2] >= 60):
            def_silent_40100_count += 1

    return {
        'declarer_win_rate': wins / n_episodes,
        'avg_declarer_gp':   total_gp  / n_episodes,
        'avg_declarer_pts':  total_pts / n_episodes,
        'def_silent_ulti':   def_silent_ulti_count  / n_episodes,
        'def_silent_40100':  def_silent_40100_count / n_episodes,
    }


def evaluate_declarer(
    play_model: MaskablePPO,
    defender_model: MaskablePPO,
    n_episodes: int = 500,
    play_env_cls=None,
    trump_selector=None,
    trump_lo: int = 0,
    trump_hi: int = 3,
    pregame_agent=None,
) -> dict:
    """
    Run the play agent against the adversarial defender (both seats).

    Parti (play_env_cls=None): uses UltiDeclarerEnv; play_model picks trump
    restricted to [trump_lo, trump_hi].
    Composite: uses play_env_cls with trump_selector driving TRUMP and
    pregame_agent driving DISCARD.
    """
    AdversarialCls = make_adversarial_cls(defender_model)
    if play_env_cls is None:
        env = UltiDeclarerEnv(
            defender1=AdversarialCls().act,
            defender2=AdversarialCls().act,
        )
    else:
        env = play_env_cls(
            trump_selector=trump_selector,
            pregame_agent=pregame_agent,
            defender1=AdversarialCls().act,
            defender2=AdversarialCls().act,
        )

    wins, total_gp, total_pts = 0, 0.0, 0.0
    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            mask = info['action_mask'].copy().astype(bool)
            if play_env_cls is None and env.game.phase == Phase.TRUMP:
                mask[:trump_lo]      = False
                mask[trump_hi + 1:]  = False
            action, _ = play_model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        gp         = env.game.game_points()[0]
        wins      += int(gp > 0)
        total_gp  += gp
        total_pts += env.game.payoffs()[0]

    return {
        'win_rate': wins    / n_episodes,
        'avg_gp':   total_gp  / n_episodes,
        'avg_pts':  total_pts / n_episodes,
    }


def evaluate_vs_heuristics(
    play_model: MaskablePPO,
    n_episodes: int = 200,
    play_env_cls=None,
    trump_selector=None,
    trump_lo: int = 0,
    trump_hi: int = 3,
    pregame_agent=None,
) -> float:
    """
    Average declarer GP (pregame_agent + play_model) vs the heuristic pool.

    Uses DefenderEnv so the declarer is always driven by the correct combination
    (pregame_agent for trump/discard, play_model for play).  The heuristic
    defender acts via game.get_observation() → cls.act(), avoiding any
    obs-format mismatch between UltiDeclarerEnv and the trained play agent.
    """
    per_cls  = max(n_episodes // len(DEFENDER_POOL), 1)
    total_gp = 0.0
    for cls in DEFENDER_POOL:
        env      = DefenderEnv(play_model, trump_selector=trump_selector,
                               pregame_agent=pregame_agent,
                               trump_lo=trump_lo, trump_hi=trump_hi)
        heuristic = cls()
        for seed in range(per_cls):
            env.reset(seed=seed)
            done = False
            while not done:
                pid      = env.game.current_player
                dict_obs = env.game.get_observation(pid)
                action   = heuristic.act(dict_obs)
                _, _, done, _, _ = env.step(action)
            total_gp += env.game.game_points()[0]
    return total_gp / (per_cls * len(DEFENDER_POOL))


# ──────────────────────────────────────────────────────────────────────────────
# VecEnv builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_defender_vec(
    play_agent,
    n_envs: int,
    trump_selector=None,
    trump_lo: int = 0,
    trump_hi: int = 3,
    pregame_agent=None,
    vecnorm_path: Optional[str] = None,
) -> VecNormalize:
    """DummyVecEnv of DefenderEnv (play_agent not picklable)."""
    def _make():
        env = DefenderEnv(play_agent, trump_selector=trump_selector,
                          pregame_agent=pregame_agent,
                          trump_lo=trump_lo, trump_hi=trump_hi)
        return ActionMasker(env, lambda e: e.action_masks())

    raw = DummyVecEnv([_make] * n_envs)
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, raw)
        vec.training = True
    else:
        vec = VecNormalize(raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return vec


def _build_declarer_vec(
    pool: list,
    n_envs: int,
    play_env_cls=None,
    trump_selector=None,
    trump_lo: int = 0,
    trump_hi: int = 3,
    pregame_agent=None,
    vecnorm_path: Optional[str] = None,
) -> VecNormalize:
    """
    Parti: DummyVecEnv of UltiDeclarerEnv with TRUMP masked to [lo, hi].
    Composite: DummyVecEnv of play_env_cls with pre-game-driven trump + discard.
    """
    def _make():
        if play_env_cls is None:
            env = UltiDeclarerEnv(defender_pool=pool)
            def _masked(e: UltiDeclarerEnv) -> np.ndarray:
                mask = e.action_masks()
                if e.game.phase == Phase.TRUMP:
                    mask[:trump_lo]     = False
                    mask[trump_hi + 1:] = False
                return mask
            return ActionMasker(env, _masked)
        else:
            env = play_env_cls(trump_selector=trump_selector,
                               pregame_agent=pregame_agent,
                               defender_pool=pool)
            return ActionMasker(env, lambda e: e.action_masks())

    raw = DummyVecEnv([_make] * n_envs)
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, raw)
        vec.training = True
    else:
        vec = VecNormalize(raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class DefenderCallback(BaseCallback):
    """Evaluates and checkpoints the defender agent (Phase B)."""

    def __init__(
        self,
        play_model: MaskablePPO,
        trump_selector=None,
        trump_lo: int = 0,
        trump_hi: int = 3,
        pregame_agent=None,
        play_env_cls=None,
        eval_freq: int = 20_000,
        n_eval: int = 200,
        save_path: Optional[str] = None,
        patience: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._play_model      = play_model
        self._trump_selector  = trump_selector
        self._trump_lo        = trump_lo
        self._trump_hi        = trump_hi
        self._pregame_agent   = pregame_agent
        self._play_env_cls    = play_env_cls
        self._eval_freq       = eval_freq
        self._n_eval          = n_eval
        self._save_path       = save_path
        self._patience        = patience
        self._best_gp         = float('inf')   # best = lowest avg declarer GP
        self._no_imp          = 0
        self._heuristic_gp    = None           # lazily computed on first eval
        self.surpassed        = False          # True once adversarial > heuristics

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        # Compute heuristic baseline once per Phase B (play_model is frozen here)
        if self._heuristic_gp is None:
            self._heuristic_gp = evaluate_vs_heuristics(
                self._play_model, self._n_eval,
                play_env_cls=self._play_env_cls,
                trump_selector=self._trump_selector,
                trump_lo=self._trump_lo, trump_hi=self._trump_hi,
                pregame_agent=self._pregame_agent,
            )

        r = evaluate_defender(
            self.model, self._play_model, self._n_eval,
            trump_selector=self._trump_selector,
            trump_lo=self._trump_lo, trump_hi=self._trump_hi,
            pregame_agent=self._pregame_agent,
        )
        avg_gp  = r['avg_declarer_gp']
        decl_wr = r['declarer_win_rate']
        ds_ulti = r['def_silent_ulti']
        ds40100 = r['def_silent_40100']
        margin  = self._heuristic_gp - avg_gp   # positive = adversarial is stronger

        self.surpassed = margin > 0

        self.logger.record('eval_def/declarer_win_rate', decl_wr)
        self.logger.record('eval_def/avg_declarer_gp',   avg_gp)
        self.logger.record('eval_def/heuristic_gp',      self._heuristic_gp)
        self.logger.record('eval_def/defender_margin',   margin)
        self.logger.record('eval_def/def_silent_ulti',   ds_ulti)
        self.logger.record('eval_def/def_silent_40100',  ds40100)
        self.logger.dump(self.num_timesteps)

        surp_tag = '  *** SURPASSED heuristics ***' if self.surpassed else ''
        line = (f"  [defender] step {self.num_timesteps:>9,}  "
                f"decl_wr={decl_wr:.1%}  avg_decl_gp={avg_gp:+.2f}  "
                f"margin={margin:+.2f}  heur={self._heuristic_gp:+.2f}"
                f"{surp_tag}")

        if avg_gp < self._best_gp:
            self._best_gp = avg_gp
            self._no_imp  = 0
            if self._save_path:
                self.model.save(f"{self._save_path}_agent_best")
                line += "  ← best saved"
        else:
            self._no_imp += 1
            if self._patience > 0:
                line += f"  (no imp {self._no_imp}/{self._patience})"

        if self.verbose:
            print(line)

        return not (self._patience > 0 and self._no_imp >= self._patience)


class DeclarerCallback(BaseCallback):
    """Evaluates and checkpoints the play agent during Phase A."""

    def __init__(
        self,
        defender_model: MaskablePPO,
        play_env_cls=None,
        trump_selector=None,
        trump_lo: int = 0,
        trump_hi: int = 3,
        pregame_agent=None,
        eval_freq: int = 20_000,
        n_eval: int = 200,
        save_path: Optional[str] = None,
        patience: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._defender_model = defender_model
        self._play_env_cls   = play_env_cls
        self._trump_selector = trump_selector
        self._trump_lo       = trump_lo
        self._trump_hi       = trump_hi
        self._pregame_agent  = pregame_agent
        self._eval_freq      = eval_freq
        self._n_eval         = n_eval
        self._save_path      = save_path
        self._patience       = patience
        self._best_gp        = -float('inf')
        self._no_imp         = 0

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        r = evaluate_declarer(
            self.model, self._defender_model, self._n_eval,
            play_env_cls=self._play_env_cls,
            trump_selector=self._trump_selector,
            trump_lo=self._trump_lo, trump_hi=self._trump_hi,
            pregame_agent=self._pregame_agent,
        )
        avg_gp = r['avg_gp']
        wr     = r['win_rate']

        self.logger.record('eval_def/declarer_win_rate_vs_adv', wr)
        self.logger.record('eval_def/declarer_avg_gp_vs_adv',   avg_gp)
        self.logger.dump(self.num_timesteps)

        line = (f"  [declarer] step {self.num_timesteps:>9,}  "
                f"wr={wr:.1%}  avg_gp={avg_gp:+.2f}")

        if avg_gp > self._best_gp:
            self._best_gp = avg_gp
            self._no_imp  = 0
            if self._save_path:
                self.model.save(f"{self._save_path}_declarer_best")
                line += "  ← best saved"
        else:
            self._no_imp += 1
            if self._patience > 0:
                line += f"  (no imp {self._no_imp}/{self._patience})"

        if self.verbose:
            print(line)

        return not (self._patience > 0 and self._no_imp >= self._patience)


# ──────────────────────────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────────────────────────

def _print_defender_results(defender_model, play_model, trump_selector=None,
                             trump_lo=0, trump_hi=3, pregame_agent=None,
                             n_episodes=500) -> None:
    r = evaluate_defender(defender_model, play_model, n_episodes,
                          trump_selector=trump_selector,
                          trump_lo=trump_lo, trump_hi=trump_hi,
                          pregame_agent=pregame_agent)
    print(f"  decl_wr={r['declarer_win_rate']:.1%}  "
          f"avg_decl_gp={r['avg_declarer_gp']:+.2f}  "
          f"avg_decl_pts={r['avg_declarer_pts']:.1f}  "
          f"def_sulti={r['def_silent_ulti']:.1%}  "
          f"def_s40100={r['def_silent_40100']:.1%}")


def _print_declarer_results(play_model, defender_model, play_env_cls=None,
                             trump_selector=None, trump_lo=0, trump_hi=3,
                             pregame_agent=None, n_episodes=500) -> None:
    r = evaluate_declarer(play_model, defender_model, n_episodes,
                          play_env_cls=play_env_cls,
                          trump_selector=trump_selector,
                          trump_lo=trump_lo, trump_hi=trump_hi,
                          pregame_agent=pregame_agent)
    print(f"  wr={r['win_rate']:.1%}  avg_gp={r['avg_gp']:+.2f}  "
          f"avg_pts={r['avg_pts']:.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Alternating adversarial defender / declarer training'
    )
    p.add_argument('--subgame', choices=['parti', '40100', 'ulti', 'ulti40100'],
                   default='parti',
                   help='Subgame to train defenders for (default: parti)')
    p.add_argument('--load-declarer', required=True,
                   help='Play/declarer checkpoint to start from')
    p.add_argument('--load-pregame',  default=None,
                   help='Pre-game agent checkpoint (required for composite subgames)')
    p.add_argument('--load-defender', default=None,
                   help='Defender checkpoint to continue from')
    p.add_argument('--save',          default=None,
                   help='Save prefix (default: models/defender_{subgame})')
    p.add_argument('--n-envs',        type=int, default=4,
                   help='Parallel envs per phase (default: 4)')
    p.add_argument('--turns',         type=int, default=5,
                   help='Alternating B/A turns (default: 5)')
    p.add_argument('--defender-steps-first', type=int, default=500_000,
                   help='Phase B steps for turn 0 (default: 500_000)')
    p.add_argument('--defender-steps', type=int, default=150_000,
                   help='Phase B steps for turns 1+ (default: 150_000)')
    p.add_argument('--declarer-steps', type=int, default=300_000,
                   help='Phase A steps per turn (default: 300_000)')
    p.add_argument('--patience',      type=int, default=5,
                   help='Early-stop patience (0 = disabled, default: 5)')
    p.add_argument('--pregame-recal-steps', type=int, default=50_000,
                   help='Pre-game recalibration steps after B/A loop (composite only, default: 50_000)')
    p.add_argument('--seed',          type=int, default=0)
    p.add_argument('--tb-suffix',     default='',
                   help='Suffix appended to TensorBoard run name (used by train_all.py)')
    p.add_argument('--eval-only',     action='store_true',
                   help='Skip training; evaluate --load-declarer and --load-defender')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = _get_subgame(args.subgame)
    save   = args.save or f'models/defender_{args.subgame}'

    # Composite subgames require a pre-game agent
    if cfg['pregame_env_cls'] is not None and not args.load_pregame:
        print(f'--subgame {args.subgame} requires --load-pregame', file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
    def_vecnorm_path  = f"{save}_agent_vecnorm.pkl"
    decl_vecnorm_path = f"{save}_declarer_vecnorm.pkl"

    # ── Build trump_selector from frozen pre-game agent ───────────────────────
    pregame_agent  = None
    trump_selector = None
    if args.load_pregame:
        pregame_agent  = MaskablePPO.load(args.load_pregame)
        trump_selector = make_pregame_selector(
            pregame_agent, cfg['trump_lo'], cfg['trump_hi']
        )
    elif cfg['heuristic_selector'] is not None:
        trump_selector = cfg['heuristic_selector']

    trump_lo = cfg['trump_lo']
    trump_hi = cfg['trump_hi']

    # ── Eval-only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.load_defender:
            print('--eval-only requires --load-defender', file=sys.stderr)
            sys.exit(1)
        play_model     = MaskablePPO.load(args.load_declarer)
        defender_model = MaskablePPO.load(args.load_defender)
        print(f'\nDefender vs {args.subgame} declarer — 500 episodes:')
        _print_defender_results(defender_model, play_model,
                                trump_selector=trump_selector,
                                trump_lo=trump_lo, trump_hi=trump_hi,
                                pregame_agent=pregame_agent)
        print(f'\nDeclarer vs adversarial defender — 500 episodes:')
        _print_declarer_results(play_model, defender_model,
                                play_env_cls=cfg['play_env_cls'],
                                trump_selector=trump_selector,
                                trump_lo=trump_lo, trump_hi=trump_hi,
                                pregame_agent=pregame_agent)
        return

    # ── Load play/declarer agent ───────────────────────────────────────────────
    print(f'Loading play agent from {args.load_declarer} …')
    decl_vec   = _build_declarer_vec(
        DEFENDER_POOL, args.n_envs,
        play_env_cls=cfg['play_env_cls'],
        trump_selector=trump_selector,
        trump_lo=trump_lo, trump_hi=trump_hi,
        pregame_agent=pregame_agent,
        vecnorm_path=decl_vecnorm_path,
    )
    play_agent = MaskablePPO.load(args.load_declarer, env=decl_vec)

    # ── Load defender if resuming ─────────────────────────────────────────────
    defender_agent = None
    if args.load_defender:
        def_vec = _build_defender_vec(
            play_agent, args.n_envs,
            trump_selector=trump_selector,
            trump_lo=trump_lo, trump_hi=trump_hi,
            pregame_agent=pregame_agent,
            vecnorm_path=def_vecnorm_path,
        )
        print(f'Loading defender from {args.load_defender} …')
        defender_agent = MaskablePPO.load(args.load_defender, env=def_vec)

    # ── Alternating B → A loop ────────────────────────────────────────────────
    tb_prefix = f'defender_{args.subgame}'
    print(f'\nAlternating {args.subgame} defender / play agent — '
          f'{args.turns} turns (B→A)\n')

    for turn in range(args.turns):
        print(f'\n{"=" * 64}\n  Turn {turn + 1} / {args.turns}\n{"=" * 64}')

        # ── Phase B: train defender ───────────────────────────────────────────
        def_vec = _build_defender_vec(
            play_agent, args.n_envs,
            trump_selector=trump_selector,
            trump_lo=trump_lo, trump_hi=trump_hi,
            pregame_agent=pregame_agent,
            vecnorm_path=def_vecnorm_path,
        )
        if defender_agent is None:
            defender_agent = MaskablePPO(
                policy='MlpPolicy', env=def_vec,
                learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log='runs/', verbose=0, seed=args.seed,
            )
        else:
            defender_agent.set_env(def_vec)

        d_steps = args.defender_steps_first if turn == 0 else args.defender_steps
        print(f'\n  Phase B — defender ({d_steps:,} steps)\n')
        def_cb = DefenderCallback(
            play_model=play_agent,
            trump_selector=trump_selector,
            trump_lo=trump_lo, trump_hi=trump_hi,
            pregame_agent=pregame_agent,
            play_env_cls=cfg['play_env_cls'],
            eval_freq=max(20_000 // args.n_envs, 1_000),
            n_eval=200, save_path=save, patience=args.patience,
        )
        defender_agent.learn(
            total_timesteps=d_steps,
            callback=def_cb,
            reset_num_timesteps=(turn == 0 and not args.load_defender),
            tb_log_name='_'.join(filter(None, [f'{tb_prefix}_agent', args.tb_suffix])),
        )
        def_vec.save(def_vecnorm_path)

        # ── Phase A: fine-tune play agent ─────────────────────────────────────
        AdversarialCls = make_adversarial_cls(defender_agent)
        pool = [AdversarialCls, AdversarialCls, AdversarialCls] + DEFENDER_POOL
        pool_desc = '75% adversarial + 25% heuristic'

        decl_vec = _build_declarer_vec(
            pool, args.n_envs,
            play_env_cls=cfg['play_env_cls'],
            trump_selector=trump_selector,
            trump_lo=trump_lo, trump_hi=trump_hi,
            pregame_agent=pregame_agent,
            vecnorm_path=decl_vecnorm_path,
        )
        play_agent.set_env(decl_vec)

        print(f'\n  Phase A — play agent ({args.declarer_steps:,} steps, '
              f'{pool_desc})\n')
        play_agent.learn(
            total_timesteps=args.declarer_steps,
            callback=DeclarerCallback(
                defender_model=defender_agent,
                play_env_cls=cfg['play_env_cls'],
                trump_selector=trump_selector,
                trump_lo=trump_lo, trump_hi=trump_hi,
                pregame_agent=pregame_agent,
                eval_freq=max(20_000 // args.n_envs, 1_000),
                n_eval=200, save_path=save, patience=args.patience,
            ),
            reset_num_timesteps=False,
            tb_log_name='_'.join(filter(None, [f'{tb_prefix}_play', args.tb_suffix])),
        )
        decl_vec.save(decl_vecnorm_path)

    # ── Phase C: pre-game recalibration (composite subgames only) ────────────
    if cfg['pregame_env_cls'] is not None and args.load_pregame:
        print(f'\n{"=" * 64}\n  Phase C — pre-game recalibration\n{"=" * 64}')

        recal_play_path = f'{save}_declarer_best'
        recal_play = (MaskablePPO.load(recal_play_path)
                      if os.path.exists(recal_play_path + '.zip') else play_agent)

        best_def_path  = f'{save}_agent_best'
        best_def_model = (MaskablePPO.load(best_def_path)
                          if os.path.exists(best_def_path + '.zip') else defender_agent)

        AdversarialCls    = make_adversarial_cls(best_def_model)
        pregame_pool      = [AdversarialCls, AdversarialCls, AdversarialCls] + DEFENDER_POOL
        pregame_vecnorm   = f'{save}_pregame_vecnorm.pkl'

        def _pregame_init():
            env = cfg['pregame_env_cls'](play_agent=recal_play, defender_pool=pregame_pool)
            return ActionMasker(env, lambda e: e.action_masks())

        pregame_vec   = build_pregame_vec(_pregame_init, args.n_envs, pregame_vecnorm)
        print(f'Loading pre-game agent from {args.load_pregame} …')
        pregame_recal = MaskablePPO.load(args.load_pregame, env=pregame_vec)

        print(f'\n  Pre-game recalibration ({args.pregame_recal_steps:,} steps)\n')
        pregame_recal.learn(
            total_timesteps=args.pregame_recal_steps,
            callback=PreGameCallback(
                pregame_env_cls=cfg['pregame_env_cls'],
                play_model=recal_play,
                trump_lo=cfg['trump_lo'],
                tb_prefix=f'defender_{args.subgame}',
                adversarial_defender_cls=AdversarialCls,
                eval_freq=max(10_000 // args.n_envs, 500),
                n_eval=200, save_path=save, patience=args.patience,
            ),
            reset_num_timesteps=False,
            tb_log_name='_'.join(filter(None,
                [f'defender_{args.subgame}_pregame', args.tb_suffix])),
        )
        pregame_vec.save(pregame_vecnorm)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print('\n\nFinal evaluation — best checkpoints (500 episodes, deterministic):')
    best_play     = MaskablePPO.load(f'{save}_declarer_best')
    best_defender = MaskablePPO.load(f'{save}_agent_best')

    print('\nDefender (stops declarer):')
    _print_defender_results(best_defender, best_play,
                            trump_selector=trump_selector,
                            trump_lo=trump_lo, trump_hi=trump_hi,
                            pregame_agent=pregame_agent)
    print('\nPlay agent (vs adversarial defender):')
    _print_declarer_results(best_play, best_defender,
                            play_env_cls=cfg['play_env_cls'],
                            trump_selector=trump_selector,
                            trump_lo=trump_lo, trump_hi=trump_hi,
                            pregame_agent=pregame_agent)


if __name__ == '__main__':
    main()
