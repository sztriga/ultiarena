"""
Shared argument parser and hierarchical training loop.

make_hierarchical_parser
    Shared argparse setup for all three hierarchical training scripts.

train_hierarchical
    The alternating B/A training loop (Phase B = pre-game, Phase A = play).
    Called by train_40100.py, train_ulti.py, train_ulti40100.py.
"""
from __future__ import annotations

import argparse
import os
from typing import Callable, Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from .registry import DEFENDER_POOL, make_adversarial_cls, make_pregame_selector, DEFENDERS
from .vectors import build_play_vec, build_pregame_vec
from .callbacks import PlayCallback, PreGameCallback
from .evaluation import print_play_results, print_pregame_results


# ──────────────────────────────────────────────────────────────────────────────
# Shared argument parser
# ──────────────────────────────────────────────────────────────────────────────

def make_hierarchical_parser(description: str, default_save: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser with the standard hierarchical training args."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--load-play',      required=True,
                   help='Play checkpoint to warm-start from')
    p.add_argument('--load-pregame',   default=None,
                   help='Pre-game checkpoint to continue from')
    p.add_argument('--save',           default=default_save,
                   help=f'Save prefix (default: {default_save})')
    p.add_argument('--n-envs',         type=int, default=4,
                   help='Parallel envs for play training (default: 4)')
    p.add_argument('--n-pregame-envs', type=int, default=4,
                   help='Parallel envs for pre-game training (default: 4)')
    p.add_argument('--turns',          type=int, default=5,
                   help='Alternating B/A turns (default: 5)')
    p.add_argument('--play-steps',     type=int, default=300_000,
                   help='Phase A steps per turn (default: 300_000)')
    p.add_argument('--pregame-steps',  type=int, default=50_000,
                   help='Phase B steps per turn (default: 50_000)')
    p.add_argument('--patience',       type=int, default=5,
                   help='Early-stop patience; 0 = disabled (default: 5)')
    p.add_argument('--seed',           type=int, default=0)
    p.add_argument('--load-def',       default=None,
                   help='Trained adversarial defender model (replaces heuristics)')
    p.add_argument('--tb-suffix',      default='',
                   help='Suffix appended to TensorBoard run name (used by train_all.py)')
    p.add_argument('--eval-only',      action='store_true',
                   help='Skip training; evaluate --load-play and --load-pregame')
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_hierarchical(
    args,
    play_env_cls,
    pregame_env_cls,
    heuristic_selector: Callable,
    trump_lo: int,
    trump_hi: int,
    tb_prefix: str,
    description: str,
    silent_ulti: bool = False,
    silent_40100: bool = False,
) -> None:
    """
    Alternating B/A hierarchical training loop shared by all composite subgames.

    Phase B: pre-game agent learns TRUMP + DISCARD decisions using a frozen
             play agent to simulate the PLAY phase internally.
    Phase A: play agent trains on pre-game-approved hands with the pre-game
             agent driving TRUMP selection and DISCARD steps.

    Parameters
    ----------
    args               Namespace from make_hierarchical_parser().parse_args()
    play_env_cls       PlayEnv class (e.g. UltiPlayEnv, NegyvenszazPlayEnv)
    pregame_env_cls    PreGameEnv class (e.g. UltiPreGameEnv, NegyvenszazPreGameEnv)
    heuristic_selector Fallback trump selector used before pre-game agent exists
    trump_lo           Inclusive lower bound of trump action range for this subgame
    trump_hi           Inclusive upper bound of trump action range for this subgame
    tb_prefix          TensorBoard key prefix (e.g. '40100', 'ulti', 'ulti40100')
    description        Human-readable label printed at training start
    """
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    play_vecnorm_path    = f"{args.save}_play_vecnorm.pkl"    if args.save else None
    pregame_vecnorm_path = f"{args.save}_pregame_vecnorm.pkl" if args.save else None

    # ── Adversarial defender ───────────────────────────────────────────────────
    load_def = getattr(args, 'load_def', None)
    if load_def:
        print(f'Loading adversarial defender from {load_def} …')
        _adv_model      = MaskablePPO.load(load_def)
        adversarial_cls = make_adversarial_cls(_adv_model)
        defender_pool   = [adversarial_cls, adversarial_cls, adversarial_cls] + DEFENDER_POOL
        pool_label      = f'75% adversarial + 25% heuristic ({load_def})'
    else:
        adversarial_cls = None
        defender_pool   = DEFENDER_POOL
        pool_label      = 'heuristics'
    print(f'Defender pool: {pool_label}')

    # ── Env factories ──────────────────────────────────────────────────────────
    def play_factory(selector, pregame_agent=None):
        def _init():
            env = play_env_cls(
                trump_selector=selector,
                pregame_agent=pregame_agent,
                defender_pool=defender_pool,
            )
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    def pregame_factory(play_agent):
        def _init():
            env = pregame_env_cls(play_agent=play_agent, defender_pool=defender_pool)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    # ── Eval-only path ─────────────────────────────────────────────────────────
    if args.eval_only:
        play_model    = MaskablePPO.load(args.load_play)
        pregame_model = MaskablePPO.load(args.load_pregame) if args.load_pregame else None
        print(f'\nPlay agent  ({args.load_play})  — 500 episodes each:')
        print_play_results(
            play_model, play_env_cls, heuristic_selector,
            pregame_model=pregame_model, trump_lo=trump_lo, trump_hi=trump_hi,
        )
        if pregame_model:
            print(f'\nPre-game agent ({args.load_pregame})  — 500 episodes each:')
            print_pregame_results(pregame_model, play_model, pregame_env_cls, trump_lo)
        return

    # ── Initialise play agent ──────────────────────────────────────────────────
    play_vec = build_play_vec(
        play_factory(heuristic_selector), args.n_envs, args.seed, play_vecnorm_path
    )
    print(f'Loading play agent from {args.load_play} …')
    play_agent = MaskablePPO.load(args.load_play, env=play_vec)

    # ── Initialise pre-game agent if resuming ──────────────────────────────────
    pregame_agent = None
    if args.load_pregame:
        pregame_vec = build_pregame_vec(
            pregame_factory(play_agent), args.n_pregame_envs, pregame_vecnorm_path
        )
        print(f'Loading pre-game agent from {args.load_pregame} …')
        pregame_agent = MaskablePPO.load(args.load_pregame, env=pregame_vec)

    # ── Alternating B/A loop ───────────────────────────────────────────────────
    print(f'\n{description} — {args.turns} turns  (B→A order)\n')

    for turn in range(args.turns):
        print(f'\n{"=" * 64}\n  Turn {turn + 1} / {args.turns}\n{"=" * 64}')

        # Phase B — pre-game agent
        pregame_vec = build_pregame_vec(
            pregame_factory(play_agent), args.n_pregame_envs, pregame_vecnorm_path
        )
        if pregame_agent is None:
            pregame_agent = MaskablePPO(
                policy='MlpPolicy', env=pregame_vec,
                learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.02,
                policy_kwargs=dict(net_arch=[128, 128]),
                tensorboard_log='runs/', verbose=0, seed=args.seed,
            )
        else:
            pregame_agent.set_env(pregame_vec)

        print(f'\n  Phase B — pre-game agent ({args.pregame_steps:,} steps)\n')
        pregame_agent.learn(
            total_timesteps=args.pregame_steps,
            callback=PreGameCallback(
                pregame_env_cls=pregame_env_cls, play_model=play_agent,
                trump_lo=trump_lo, tb_prefix=tb_prefix,
                adversarial_defender_cls=adversarial_cls,
                eval_freq=max(10_000 // args.n_pregame_envs, 500),
                n_eval=200, save_path=args.save, patience=args.patience,
            ),
            reset_num_timesteps=(turn == 0 and not args.load_pregame),
            tb_log_name='_'.join(filter(None, [f'{tb_prefix}_pregame', getattr(args, 'tb_suffix', '')])),
        )
        if pregame_vecnorm_path:
            pregame_vec.save(pregame_vecnorm_path)

        # Phase A — play agent (pre-game agent drives TRUMP + DISCARD)
        selector = make_pregame_selector(pregame_agent, trump_lo, trump_hi)
        play_vec = build_play_vec(
            play_factory(selector, pregame_agent=pregame_agent),
            args.n_envs, args.seed, play_vecnorm_path,
        )
        play_agent.set_env(play_vec)

        print(f'\n  Phase A — play agent  ({args.play_steps:,} steps, pre-game-driven trump)\n')
        play_agent.learn(
            total_timesteps=args.play_steps,
            callback=PlayCallback(
                play_env_cls=play_env_cls, heuristic_selector=heuristic_selector,
                trump_lo=trump_lo, trump_hi=trump_hi, tb_prefix=tb_prefix,
                pregame_agent=pregame_agent,
                adversarial_defender_cls=adversarial_cls,
                silent_ulti=silent_ulti,
                silent_40100=silent_40100,
                eval_freq=max(20_000 // args.n_envs, 1_000),
                n_eval=200, save_path=args.save, patience=args.patience,
            ),
            reset_num_timesteps=False,
            tb_log_name='_'.join(filter(None, [f'{tb_prefix}_play', getattr(args, 'tb_suffix', '')])),
        )
        if play_vecnorm_path:
            play_vec.save(play_vecnorm_path)

    # ── Final evaluation ───────────────────────────────────────────────────────
    print('\n\nFinal evaluation — best checkpoints (500 eps each, deterministic):')
    best_play    = MaskablePPO.load(f'{args.save}_play_best')    if args.save else play_agent
    best_pregame = MaskablePPO.load(f'{args.save}_pregame_best') if args.save else pregame_agent

    print('\nPlay agent:')
    print_play_results(
        best_play, play_env_cls, heuristic_selector,
        pregame_model=best_pregame, trump_lo=trump_lo, trump_hi=trump_hi,
        adversarial_defender_cls=adversarial_cls,
        silent_ulti=silent_ulti,
        silent_40100=silent_40100,
    )
    print('\nPre-game agent:')
    print_pregame_results(best_pregame, best_play, pregame_env_cls, trump_lo,
                          adversarial_defender_cls=adversarial_cls)
