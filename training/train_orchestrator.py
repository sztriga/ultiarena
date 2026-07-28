"""
Train the Ulti Orchestrator: a 1-step agent that selects a licit × trump suit
from the 16 TRUMP actions and receives game_points()[0] as reward.

    Observation (48-dim float32):
        hand[32] + trump_legal_mask[16]

    Action: Discrete(16) → TRUMP action 0–15
        Subgame routing: action // 4
            0 = Parti      (actions 0–3)
            1 = 40-100     (actions 4–7)
            2 = Ulti       (actions 8–11)
            3 = 40-100+Ulti (actions 12–15)

The four specialist play agents are required.  Adversarial defenders are
optional; heuristic defenders are used for any subgame without one.

Primary metric: average game_points()[0] per hand.

Usage
-----
    python train_orchestrator.py \\
        --load-parti         models/parti_play_best \\
        --load-40100         models/defender_40100_declarer_best \\
        --load-ulti          models/defender_ulti_declarer_best \\
        --load-ulti40100     models/defender_ulti40100_declarer_best \\
        --load-def-parti     models/defender_parti_agent_best \\
        --load-def-40100     models/defender_40100_agent_best \\
        --load-def-ulti      models/defender_ulti_agent_best \\
        --load-def-ulti40100 models/defender_ulti40100_agent_best

    # Continue from checkpoint
    python train_orchestrator.py ... --load-orch models/orchestrator_best

    # Evaluate only
    python train_orchestrator.py \\
        --load-parti ... --load-40100 ... --load-ulti ... --load-ulti40100 ... \\
        --eval-only --load-orch models/orchestrator_best
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

from envs.orchestrator import OrchestratorEnv
from envs.obs import play_flatten as _def_flatten
from agents.heuristic import RandomAgent, GreedyAgent, ConservativeAgent, SmartAgent
from training.registry import DEFENDERS, DEFENDER_POOL

SUBGAME_NAMES = ['Parti', '40-100', 'Ulti', '40-100+Ulti']


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial defender wrapper
# ──────────────────────────────────────────────────────────────────────────────

def make_adversarial_cls(model: MaskablePPO):
    """Return a defender class (with .act method) wrapping a trained model."""
    class AdversarialDefender:
        def act(self, obs_dict: dict) -> int:
            flat   = _def_flatten(obs_dict)
            mask   = obs_dict['action_mask'].astype(bool)
            action, _ = model.predict(flat, action_masks=mask, deterministic=True)
            return int(action)
    return AdversarialDefender


def _build_defender_pools(
    def_parti=None,
    def_40100=None,
    def_ulti=None,
    def_ulti40100=None,
) -> list:
    """
    Build 4 defender pools, one per subgame.

    When an adversarial model is provided for a subgame, the pool is blended
    50/50 adversarial + heuristic (2 each).  Subgames without an adversarial
    model use the standard 4-agent heuristic pool.
    """
    def _pool(model):
        if model is None:
            return DEFENDER_POOL[:]
        AdversarialCls = make_adversarial_cls(model)
        return DEFENDER_POOL[:2] + [AdversarialCls, AdversarialCls]

    return [
        _pool(def_parti),
        _pool(def_40100),
        _pool(def_ulti),
        _pool(def_ulti40100),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_orchestrator(
    orch_model: MaskablePPO,
    play_agents: list,
    defender_pools: list,
    n_episodes: int = 500,
) -> dict:
    """
    Run the orchestrator deterministically; return rich metrics dict.

    Keys
    ----
    avg_gp          Average game_points()[0] per hand  ← primary metric
    win_rate        Fraction of hands where gp > 0
    bid_share       {subgame_name: fraction of hands where bid chosen}
    win_rate_by_sg  {subgame_name: win rate conditioned on that subgame}
    piros_pct       Fraction of hands where hearts trump (action % 4 == 2)
    silent_ulti     Fraction where declarer won last trick with trump 7
                    (only counted on non-Ulti licits)
    silent_40100    Fraction where declarer had trump pair + ≥ 60 trick pts
                    (only counted on non-40-100 licits)
    """
    env = OrchestratorEnv(play_agents, defender_pools)

    total_gp = 0.0
    wins     = 0
    sg_counts = [0] * 4
    sg_wins   = [0] * 4
    piros_count   = 0
    silent_ulti   = 0
    silent_40100  = 0

    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        mask = info['action_mask'].astype(bool)
        action, _ = orch_model.predict(obs, action_masks=mask, deterministic=True)
        _, _, _, _, _ = env.step(int(action))

        gp  = env.game.game_points()[0]
        sg  = int(action) // 4
        total_gp += gp
        if gp > 0:
            wins += 1
            sg_wins[sg] += 1
        sg_counts[sg] += 1
        if int(action) % 4 == 2:
            piros_count += 1
        # Silent Ulti: won last trick with trump 7 on a non-Ulti licit
        if sg not in (2, 3) and env.game._declarer_ulti:
            silent_ulti += 1
        # Silent 40-100: trump pair + ≥ 60 trick pts on a non-40-100 licit
        if sg not in (1, 3) and env.game._declarer_has_trump_pair and env.game.trick_scores[0] >= 60:
            silent_40100 += 1

    return {
        'avg_gp':    total_gp / n_episodes,
        'win_rate':  wins / n_episodes,
        'bid_share': {SUBGAME_NAMES[i]: sg_counts[i] / n_episodes for i in range(4)},
        'win_rate_by_sg': {
            SUBGAME_NAMES[i]: sg_wins[i] / sg_counts[i] if sg_counts[i] > 0 else 0.0
            for i in range(4)
        },
        'piros_pct':   piros_count  / n_episodes,
        'silent_ulti': silent_ulti  / n_episodes,
        'silent_40100': silent_40100 / n_episodes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VecEnv builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_orch_vec(
    play_agents: list,
    defender_pools: list,
    n_envs: int,
    vecnorm_path: Optional[str] = None,
) -> VecNormalize:
    """DummyVecEnv — play_agents are not picklable for SubprocVecEnv."""
    def _make():
        env = OrchestratorEnv(play_agents, defender_pools)
        return ActionMasker(env, lambda e: e.action_masks())

    raw = DummyVecEnv([_make] * n_envs)
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, raw)
        vec.training = True
    else:
        # clip_reward=20 covers ~1 SD of the GP distribution
        vec = VecNormalize(raw, norm_obs=False, norm_reward=True, clip_reward=20.0)
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────────────────────

class OrchestratorCallback(BaseCallback):
    def __init__(
        self,
        play_agents: list,
        defender_pools: list,
        eval_freq: int = 20_000,
        n_eval: int = 500,
        save_path: Optional[str] = None,
        patience: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._play_agents    = play_agents
        self._defender_pools = defender_pools
        self._eval_freq      = eval_freq
        self._n_eval         = n_eval
        self._save_path      = save_path
        self._patience       = patience
        self._best_gp        = -float('inf')
        self._no_imp         = 0

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        r      = evaluate_orchestrator(
            self.model, self._play_agents, self._defender_pools, self._n_eval,
        )
        avg_gp = r['avg_gp']
        wr     = r['win_rate']

        self.logger.record('eval_orch/avg_gp',      avg_gp)
        self.logger.record('eval_orch/win_rate',     wr)
        self.logger.record('eval_orch/piros_pct',    r['piros_pct'])
        self.logger.record('eval_orch/silent_ulti',  r['silent_ulti'])
        self.logger.record('eval_orch/silent_40100', r['silent_40100'])
        for i, name in enumerate(SUBGAME_NAMES):
            key = name.replace('-', '').replace('+', '_')
            self.logger.record(f'eval_orch/bid_share_{key}', r['bid_share'][name])
            self.logger.record(f'eval_orch/wr_{key}',        r['win_rate_by_sg'][name])
        self.logger.dump(self.num_timesteps)

        shares = '  '.join(
            f"{SUBGAME_NAMES[i]}={r['bid_share'][SUBGAME_NAMES[i]]:.1%}"
            for i in range(4)
        )
        line = (f"  [orch] step {self.num_timesteps:>9,}  "
                f"avg_gp={avg_gp:+.3f}  wr={wr:.1%}  [{shares}]")

        if avg_gp > self._best_gp:
            self._best_gp = avg_gp
            self._no_imp  = 0
            if self._save_path:
                self.model.save(f"{self._save_path}_best")
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

def _print_orch_results(
    orch_model: MaskablePPO,
    play_agents: list,
    defender_pools: list,
    n_episodes: int = 1000,
) -> None:
    r = evaluate_orchestrator(orch_model, play_agents, defender_pools, n_episodes)
    print(f"  avg_gp={r['avg_gp']:+.3f}  wr={r['win_rate']:.1%}  "
          f"piros={r['piros_pct']:.1%}  "
          f"s_ulti={r['silent_ulti']:.1%}  s_40100={r['silent_40100']:.1%}")
    print("  Bid share:  " + "  ".join(
        f"{n}={r['bid_share'][n]:.1%}" for n in SUBGAME_NAMES))
    print("  Win rates:  " + "  ".join(
        f"{n}={r['win_rate_by_sg'][n]:.1%}" for n in SUBGAME_NAMES))


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train the Ulti orchestrator (bid selection meta-agent)',
    )
    # Required: 4 play agents
    p.add_argument('--load-parti',      required=True,
                   help='Parti play agent checkpoint')
    p.add_argument('--load-40100',      required=True,
                   help='40-100 play agent checkpoint')
    p.add_argument('--load-ulti',       required=True,
                   help='Ulti play agent checkpoint')
    p.add_argument('--load-ulti40100',  required=True,
                   help='40-100+Ulti play agent checkpoint')
    # Required: adversarial defenders (one per subgame)
    p.add_argument('--load-def-parti',      required=True,
                   help='Adversarial defender checkpoint for Parti')
    p.add_argument('--load-def-40100',      required=True,
                   help='Adversarial defender checkpoint for 40-100')
    p.add_argument('--load-def-ulti',       required=True,
                   help='Adversarial defender checkpoint for Ulti')
    p.add_argument('--load-def-ulti40100',  required=True,
                   help='Adversarial defender checkpoint for 40-100+Ulti')
    # Optional: resume orchestrator
    p.add_argument('--load-orch', default=None,
                   help='Orchestrator checkpoint to continue from')
    p.add_argument('--save',      default='models/orchestrator',
                   help='Save prefix (default: models/orchestrator)')
    p.add_argument('--n-envs',    type=int, default=4,
                   help='Parallel envs (default: 4)')
    p.add_argument('--steps',     type=int, default=500_000,
                   help='Total training timesteps (default: 500_000)')
    p.add_argument('--patience',  type=int, default=10,
                   help='Early-stop patience in eval intervals (0=disabled, default: 10)')
    p.add_argument('--seed',      type=int, default=0)
    p.add_argument('--tb-suffix', default='',
                   help='Suffix appended to TensorBoard run name (used by train_all.py)')
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training; evaluate --load-orch')
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load play agents ───────────────────────────────────────────────────────
    print('Loading play agents …')
    play_agents = [
        MaskablePPO.load(args.load_parti),
        MaskablePPO.load(args.load_40100),
        MaskablePPO.load(args.load_ulti),
        MaskablePPO.load(args.load_ulti40100),
    ]

    # ── Load adversarial defenders (optional) ─────────────────────────────────
    def _load_opt(path):
        return MaskablePPO.load(path) if path else None

    def_models = [
        _load_opt(args.load_def_parti),
        _load_opt(args.load_def_40100),
        _load_opt(args.load_def_ulti),
        _load_opt(args.load_def_ulti40100),
    ]
    defender_pools = _build_defender_pools(*def_models)
    print(f'Adversarial defenders loaded for: {", ".join(SUBGAME_NAMES)}')

    # ── Eval-only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.load_orch:
            print('--eval-only requires --load-orch', file=sys.stderr)
            sys.exit(1)
        orch_model = MaskablePPO.load(args.load_orch)
        print('\nOrchestrator evaluation — 1,000 episodes (deterministic):')
        _print_orch_results(orch_model, play_agents, defender_pools)
        return

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    vecnorm_path = f'{args.save}_vecnorm.pkl'

    # ── Build training env ────────────────────────────────────────────────────
    vec = _build_orch_vec(play_agents, defender_pools, args.n_envs, vecnorm_path)

    # ── Load or create orchestrator ───────────────────────────────────────────
    if args.load_orch:
        print(f'Loading orchestrator from {args.load_orch} …')
        orch_agent = MaskablePPO.load(args.load_orch, env=vec)
    else:
        orch_agent = MaskablePPO(
            policy='MlpPolicy', env=vec,
            learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
            gamma=1.0,       # 1-step episodes — no discounting
            gae_lambda=1.0,
            ent_coef=0.02,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log='runs/', verbose=0, seed=args.seed,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f'\nOrchestrator training — {args.steps:,} steps\n')
    orch_agent.learn(
        total_timesteps=args.steps,
        callback=OrchestratorCallback(
            play_agents=play_agents,
            defender_pools=defender_pools,
            eval_freq=max(20_000 // args.n_envs, 1_000),
            n_eval=500,
            save_path=args.save,
            patience=args.patience,
        ),
        reset_num_timesteps=not bool(args.load_orch),
        tb_log_name='_'.join(filter(None, ['orchestrator', args.tb_suffix])),
    )
    vec.save(vecnorm_path)

    # ── Final evaluation ───────────────────────────────────────────────────────
    print('\n\nFinal evaluation — best checkpoint (1,000 episodes, deterministic):')
    best = MaskablePPO.load(f'{args.save}_best')
    _print_orch_results(best, play_agents, defender_pools)


if __name__ == '__main__':
    main()
