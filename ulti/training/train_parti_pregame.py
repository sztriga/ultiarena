"""
Train the Parti pre-game agent (TRUMP selection + DISCARD) — Phase B of Parti training.

Using a frozen play agent to simulate the PLAY phase internally, the pre-game
agent learns which trump suit to pick and which two cards to discard.

Since Parti always bids (no PASSZ alternative), the episode is always 3 steps:
  step 0 — pick trump suit (actions 0-3)
  step 1 — discard card 1
  step 2 — discard card 2 → PLAY runs internally → reward = GP × 10

The resulting checkpoint warm-starts all composite subgame pre-game agents
(40-100, Ulti, 40-100+Ulti) because obs/action spaces are identical.  The
discard strategy (steps 1-2) transfers directly; step-0 trump weights provide
a useful feature-extraction warm start even though the action range differs.

Typical usage
-------------
    python train_parti_pregame.py --load-play models/parti_play_best --save models/parti

    # Continue from checkpoint
    python train_parti_pregame.py \\
        --load-play    models/parti_play_best \\
        --load-pregame models/parti_pregame_best \\
        --save         models/parti

    # Evaluate only
    python train_parti_pregame.py --eval-only \\
        --load-play    models/parti_play_best \\
        --load-pregame models/parti_pregame_best
"""
from __future__ import annotations

import argparse
import os
import sys

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.parti import PartiPreGameEnv
from ulti.training.registry import DEFENDERS, DEFENDER_POOL, make_adversarial_cls
from ulti.training.vectors import build_pregame_vec
from ulti.training.callbacks import PreGameCallback
from ulti.training.evaluation import evaluate_pregame, print_pregame_results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train Parti pre-game agent (trump + discard)'
    )
    p.add_argument('--load-play',    required=True,
                   help='Frozen play agent checkpoint')
    p.add_argument('--load-pregame', default=None,
                   help='Pre-game checkpoint to continue from')
    p.add_argument('--save',         default='models/parti',
                   help='Save prefix (default: models/parti)')
    p.add_argument('--steps',        type=int, default=200_000,
                   help='Training timesteps (default: 200_000)')
    p.add_argument('--n-envs',       type=int, default=4,
                   help='Parallel envs (default: 4)')
    p.add_argument('--patience',     type=int, default=5)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--load-def',     default=None,
                   help='Adversarial defender model (replaces heuristics in pool)')
    p.add_argument('--tb-suffix',    default='')
    p.add_argument('--eval-only',    action='store_true',
                   help='Skip training; evaluate --load-play and --load-pregame')
    return p.parse_args()


def main():
    args = parse_args()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    # ── Adversarial defender ───────────────────────────────────────────────────
    if args.load_def:
        print(f'Loading adversarial defender from {args.load_def} …')
        _adv_model      = MaskablePPO.load(args.load_def)
        adversarial_cls = make_adversarial_cls(_adv_model)
        defender_pool   = [adversarial_cls] * 3 + DEFENDER_POOL
        pool_label      = f'75% adversarial + 25% heuristic'
    else:
        adversarial_cls = None
        defender_pool   = DEFENDER_POOL
        pool_label      = 'heuristics'
    print(f'Defender pool: {pool_label}')

    # ── Load play agent ────────────────────────────────────────────────────────
    print(f'Loading play agent from {args.load_play} …')
    play_agent = MaskablePPO.load(args.load_play)

    # ── Eval-only path ─────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.load_pregame:
            print('--eval-only requires --load-pregame', file=sys.stderr)
            sys.exit(1)
        pregame_agent = MaskablePPO.load(args.load_pregame)
        print(f'\nPre-game agent ({args.load_pregame})  — 500 episodes each:')
        print_pregame_results(pregame_agent, play_agent, PartiPreGameEnv,
                              trump_lo=0, n_episodes=500)
        return

    # ── Build pregame vec ──────────────────────────────────────────────────────
    pregame_vecnorm_path = f"{args.save}_pregame_vecnorm.pkl" if args.save else None

    def _pregame_init():
        env = PartiPreGameEnv(play_agent=play_agent, defender_pool=defender_pool)
        return ActionMasker(env, lambda e: e.action_masks())

    pregame_vec = build_pregame_vec(_pregame_init, args.n_envs, pregame_vecnorm_path)

    # ── Initialise / load pregame agent ───────────────────────────────────────
    if args.load_pregame:
        print(f'Loading pre-game agent from {args.load_pregame} …')
        pregame_agent = MaskablePPO.load(args.load_pregame, env=pregame_vec)
    else:
        pregame_agent = MaskablePPO(
            policy='MlpPolicy', env=pregame_vec,
            learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.02,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log='runs/', verbose=0, seed=args.seed,
        )

    # ── Train ──────────────────────────────────────────────────────────────────
    tb_name = '_'.join(filter(None, ['parti_pregame', args.tb_suffix]))
    print(f'\nTraining Parti pre-game agent for {args.steps:,} steps …\n')

    pregame_agent.learn(
        total_timesteps=args.steps,
        callback=PreGameCallback(
            pregame_env_cls=PartiPreGameEnv,
            play_model=play_agent,
            trump_lo=0,
            tb_prefix='parti',
            adversarial_defender_cls=adversarial_cls,
            eval_freq=max(10_000 // args.n_envs, 500),
            n_eval=200,
            save_path=args.save,
            patience=args.patience,
        ),
        reset_num_timesteps=args.load_pregame is None,
        tb_log_name=tb_name,
    )

    if pregame_vecnorm_path:
        pregame_vec.save(pregame_vecnorm_path)

    # ── Final evaluation ───────────────────────────────────────────────────────
    best_path = f'{args.save}_pregame_best'
    best = MaskablePPO.load(best_path) if (args.save and os.path.exists(best_path + '.zip')) else pregame_agent
    print('\nFinal evaluation — best checkpoint (500 eps, deterministic):')
    print_pregame_results(best, play_agent, PartiPreGameEnv, trump_lo=0)


if __name__ == '__main__':
    main()
