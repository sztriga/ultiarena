"""
Train the Parti play agent (PLAY phase only) — Phase A of Parti training.

Trump is selected by heuristic (most cards in suit, random tiebreak).
Discards are applied by heuristic (lowest card from shortest suit).
The agent sees only the PLAY phase and learns pure trick-taking.

The resulting checkpoint warm-starts all composite subgame play agents
(40-100, Ulti, 40-100+Ulti share the same obs/action space).

Typical usage
-------------
    python train_parti_play.py --save models/parti_play

    # Continue from checkpoint
    python train_parti_play.py --load models/parti_play_best --save models/parti_play

    # Evaluate only
    python train_parti_play.py --eval-only --load models/parti_play_best
"""
from __future__ import annotations

import argparse
import os
import sys

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.parti import PartiPlayEnv
from training.registry import DEFENDERS, DEFENDER_POOL


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model: MaskablePPO, defender_cls, n_episodes: int = 500) -> dict:
    """Run n_episodes with fixed defender type; trump and discards are heuristic."""
    env = PartiPlayEnv(defender1=defender_cls().act, defender2=defender_cls().act)
    wins, total_pts, total_gp = 0, 0.0, 0.0
    silent_40100_count = 0
    silent_ulti_count  = 0
    def_sulti_count    = 0
    def_s40100_count   = 0

    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            mask = info['action_mask'].astype(bool)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        g = env.game
        gp = g.game_points()[0]
        wins       += int(gp > 0)
        total_pts  += g.payoffs()[0]
        total_gp   += gp
        if g._declarer_has_trump_pair and g.trick_scores[0] >= 60:
            silent_40100_count += 1
        if g._declarer_ulti:
            silent_ulti_count += 1
        if g._defender_ulti:
            def_sulti_count += 1
        if (g._defender_has_trump_pair and g.trick_scores[1] + g.trick_scores[2] >= 60):
            def_s40100_count += 1

    return {
        'win_rate':    wins            / n_episodes,
        'avg_pts':     total_pts       / n_episodes,
        'avg_gp':      total_gp        / n_episodes,
        's40100_pct':  silent_40100_count / n_episodes,
        'sulti_pct':   silent_ulti_count  / n_episodes,
        'ds_ulti_pct': def_sulti_count    / n_episodes,
        'ds40100_pct': def_s40100_count   / n_episodes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────────────────────

class PlayCallback(BaseCallback):
    """Periodic evaluation and best-model checkpointing."""

    def __init__(
        self,
        eval_freq: int = 20_000,
        n_eval: int = 200,
        save_path: str = None,
        patience: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._eval_freq = eval_freq
        self._n_eval    = n_eval
        self._save_path = save_path
        self._patience  = patience
        self._best_gp   = -float('inf')
        self._no_imp    = 0

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        results = {name: evaluate(self.model, cls, self._n_eval)
                   for name, cls in DEFENDERS.items()}

        mean_gp   = sum(r['avg_gp']      for r in results.values()) / len(results)
        mean_wr   = sum(r['win_rate']    for r in results.values()) / len(results)
        mean_pts  = sum(r['avg_pts']     for r in results.values()) / len(results)
        mean_s40  = sum(r['s40100_pct']  for r in results.values()) / len(results)
        mean_su   = sum(r['sulti_pct']   for r in results.values()) / len(results)
        mean_dsu  = sum(r['ds_ulti_pct'] for r in results.values()) / len(results)
        mean_ds40 = sum(r['ds40100_pct'] for r in results.values()) / len(results)

        self.logger.record('eval/mean_gp',           mean_gp)
        self.logger.record('eval/mean_win_rate',      mean_wr)
        self.logger.record('eval/mean_trick_pts',     mean_pts)
        self.logger.record('eval/silent_40100_pct',   mean_s40)
        self.logger.record('eval/silent_ulti_pct',    mean_su)
        self.logger.record('eval/def_silent_ulti',    mean_dsu)
        self.logger.record('eval/def_silent_40100',   mean_ds40)
        self.logger.dump(self.num_timesteps)

        line = (f"  step {self.num_timesteps:>9,}  "
                f"gp={mean_gp:+.2f}  wr={mean_wr:.3f}  pts={mean_pts:.0f}  "
                f"s40100={mean_s40:.1%}  sulti={mean_su:.1%}  "
                f"def_su={mean_dsu:.1%}  def_s40={mean_ds40:.1%}")

        if mean_gp > self._best_gp:
            self._best_gp = mean_gp
            self._no_imp  = 0
            if self._save_path:
                self.model.save(f"{self._save_path}_best")
                if hasattr(self.model.env, 'save'):
                    self.model.env.save(f"{self._save_path}_vecnorm.pkl")
                line += "  ← best saved"
        else:
            self._no_imp += 1
            if self._patience > 0:
                line += f"  (no imp {self._no_imp}/{self._patience})"

        if self.verbose:
            print(line)

        return not (self._patience > 0 and self._no_imp >= self._patience)


# ──────────────────────────────────────────────────────────────────────────────
# Print helper
# ──────────────────────────────────────────────────────────────────────────────

def _print_results(model: MaskablePPO, n_episodes: int = 500) -> None:
    all_r = {name: evaluate(model, cls, n_episodes) for name, cls in DEFENDERS.items()}
    mean  = lambda k: sum(r[k] for r in all_r.values()) / len(all_r)
    sep   = '─' * 80
    print(sep)
    print(f"  {'win%':>6}  {'avg gp':>7}  {'avg pts':>8}  "
          f"{'s40100%':>8}  {'sulti%':>7}  {'def_su%':>8}  {'def_s40%':>9}")
    print(sep)
    print(f"  {mean('win_rate')*100:>5.1f}%  {mean('avg_gp'):>+6.2f}   "
          f"{mean('avg_pts'):>8.1f}  {mean('s40100_pct')*100:>7.1f}%  "
          f"{mean('sulti_pct')*100:>6.1f}%  {mean('ds_ulti_pct')*100:>7.1f}%  "
          f"{mean('ds40100_pct')*100:>8.1f}%")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Parti play agent (PLAY phase only)')
    p.add_argument('--steps',     type=int, default=2_000_000)
    p.add_argument('--n-envs',    type=int, default=4)
    p.add_argument('--load',      default=None, help='Checkpoint to continue from')
    p.add_argument('--save',      default='models/parti_play')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--patience',  type=int, default=5)
    p.add_argument('--seed',      type=int, default=0)
    p.add_argument('--tb-suffix', default='')
    return p.parse_args()


def main():
    args = parse_args()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    if args.eval_only:
        if not args.load:
            print('--eval-only requires --load', file=sys.stderr)
            sys.exit(1)
        model = MaskablePPO.load(args.load)
        print(f'\nEvaluating {args.load}  (500 eps each) …')
        _print_results(model)
        return

    def _make_env():
        def _init():
            env = PartiPlayEnv(defender_pool=DEFENDER_POOL)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    raw = make_vec_env(_make_env(), n_envs=args.n_envs, seed=args.seed)
    vecnorm_path = f"{args.save}_vecnorm.pkl" if args.save else None
    if args.load and vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, raw)
        vec_env.training = True
    else:
        vec_env = VecNormalize(raw, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if args.load:
        print(f'Loading model from {args.load} …')
        model = MaskablePPO.load(args.load, env=vec_env)
    else:
        model = MaskablePPO(
            policy='MlpPolicy', env=vec_env,
            learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log='runs/', verbose=0, seed=args.seed,
        )

    callbacks = [
        PlayCallback(
            eval_freq=max(20_000 // args.n_envs, 1_000),
            n_eval=200, save_path=args.save, patience=args.patience,
        ),
    ]
    if args.save:
        callbacks.append(CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=os.path.dirname(args.save) or '.',
            name_prefix=os.path.basename(args.save),
        ))

    tb_name = '_'.join(filter(None, [
        os.path.basename(args.save) if args.save else 'parti_play',
        args.tb_suffix,
    ]))

    print(f'\nTraining Parti play agent for {args.steps:,} steps on {args.n_envs} envs …\n')
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        reset_num_timesteps=args.load is None,
        tb_log_name=tb_name,
    )

    if args.save:
        model.save(f"{args.save}_final")
        vec_env.save(vecnorm_path)

    best = MaskablePPO.load(f"{args.save}_best") if args.save else model
    print('\nFinal evaluation — best checkpoint (500 eps each):')
    _print_results(best)


if __name__ == '__main__':
    main()
