"""
Train the Ulti declarer with MaskablePPO.

Training always uses a mixed defender pool (all 4 agent types sampled
independently each episode) to prevent over-fitting to any single play style.
The agent always plays the full game: DISCARD → TRUMP → PLAY.

Typical usage
-------------
  python train.py --steps 2_000_000 --save models/ulti

  # Continue from a checkpoint
  python train.py --steps 2_000_000 --load models/ulti_best --save models/ulti

  # Just evaluate a saved model (no training)
  python train.py --eval-only --load models/ulti_best
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from envs.base import UltiDeclarerEnv
from ulti.game import Licit, _NEGYVENSZAZ_LICITS
from training.registry import DEFENDERS, DEFENDER_POOL


# ──────────────────────────────────────────────────────────────────────────────
# Env factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_env():
    """
    Return a zero-arg factory that creates one training env.

    Defenders are drawn from the full pool at each reset() — the agent faces
    every combination of (random, greedy, conservative, smart) over training.
    """
    def _init():
        env = UltiDeclarerEnv(defender_pool=DEFENDER_POOL)
        return ActionMasker(env, lambda e: e.action_masks())
    return _init


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: MaskablePPO,
    defender_cls,
    n_episodes: int = 500,
) -> dict:
    """
    Run `n_episodes` with a fixed defender type, deterministic policy.

    Returns a dict with keys:
        win_rate, avg_pts, avg_game_pts,
        piros_pct, silent_40100_pct, silent_ulti_pct,
        def_silent_ulti_pct, def_silent_40100_pct
    """
    env = UltiDeclarerEnv(
        defender1=defender_cls().act,
        defender2=defender_cls().act,
    )
    wins = 0
    total_pts = 0.0
    total_game_pts = 0.0
    piros_count = 0
    silent_40100_count = 0
    silent_ulti_count = 0
    def_silent_ulti_count = 0
    def_silent_40100_count = 0

    for seed in range(n_episodes):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            mask = info['action_mask'].astype(bool)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        gp = env.game.game_points()[0]
        won = gp > 0
        wins += int(won)
        total_pts      += env.game.payoffs()[0]
        total_game_pts += gp
        if env.game.licit == Licit.PIROS_PARTI:
            piros_count += 1
        if env.game._declarer_has_trump_pair and env.game.trick_scores[0] >= 60:
            silent_40100_count += 1
        if env.game._declarer_ulti:
            silent_ulti_count += 1
        if env.game._defender_ulti:
            def_silent_ulti_count += 1
        if (env.game.licit not in _NEGYVENSZAZ_LICITS
                and env.game._defender_has_trump_pair
                and env.game.trick_scores[1] + env.game.trick_scores[2] >= 60):
            def_silent_40100_count += 1

    return {
        'win_rate':               wins / n_episodes,
        'avg_pts':                total_pts / n_episodes,
        'avg_game_pts':           total_game_pts / n_episodes,
        'piros_pct':              piros_count / n_episodes,
        'silent_40100_pct':       silent_40100_count / n_episodes,
        'silent_ulti_pct':        silent_ulti_count / n_episodes,
        'def_silent_ulti_pct':    def_silent_ulti_count / n_episodes,
        'def_silent_40100_pct':   def_silent_40100_count / n_episodes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────────────────────

class WinRateCallback(BaseCallback):
    """
    Every `eval_freq` steps: evaluate against each defender type, log all win
    rates to TensorBoard, and save the model when the mean win rate improves.

    Early stopping
    --------------
    If `patience` > 0, training stops automatically after `patience`
    consecutive evaluations with no improvement in mean win rate.
    Set `patience=0` to disable (run for the full --steps budget).
    """

    def __init__(
        self,
        eval_freq: int = 20_000,
        n_eval_episodes: int = 200,
        save_path: str | None = None,
        patience: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._eval_freq        = eval_freq
        self._n_eval           = n_eval_episodes
        self._save_path        = save_path
        self._patience         = patience
        self._best_mean_gp     = -float('inf')
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        results = {name: evaluate(self.model, cls, n_episodes=self._n_eval)
                   for name, cls in DEFENDERS.items()}

        mean_wr       = sum(r['win_rate']              for r in results.values()) / len(results)
        mean_gp       = sum(r['avg_game_pts']          for r in results.values()) / len(results)
        mean_pts      = sum(r['avg_pts']               for r in results.values()) / len(results)
        mean_piros    = sum(r['piros_pct']             for r in results.values()) / len(results)
        mean_s40100   = sum(r['silent_40100_pct']      for r in results.values()) / len(results)
        mean_sulti    = sum(r['silent_ulti_pct']       for r in results.values()) / len(results)
        mean_ds_ulti  = sum(r['def_silent_ulti_pct']  for r in results.values()) / len(results)
        mean_ds40100  = sum(r['def_silent_40100_pct'] for r in results.values()) / len(results)

        self.logger.record('eval/mean_win_rate',         mean_wr)
        self.logger.record('eval/mean_game_pts',         mean_gp)
        self.logger.record('eval/mean_trick_pts',        mean_pts)
        self.logger.record('eval/piros_pct',             mean_piros)
        self.logger.record('eval/silent_40100_pct',      mean_s40100)
        self.logger.record('eval/silent_ulti_pct',       mean_sulti)
        self.logger.record('eval/def_silent_ulti_pct',   mean_ds_ulti)
        self.logger.record('eval/def_silent_40100_pct',  mean_ds40100)
        self.logger.dump(self.num_timesteps)

        line = (f"  step {self.num_timesteps:>9,}  "
                f"wr={mean_wr:.3f}  gp={mean_gp:+.2f}  pts={mean_pts:.0f}  "
                f"piros={mean_piros:.1%}  s40100={mean_s40100:.1%}  sulti={mean_sulti:.1%}  "
                f"def_sulti={mean_ds_ulti:.1%}  def_s40100={mean_ds40100:.1%}")

        if mean_gp > self._best_mean_gp:
            self._best_mean_gp     = mean_gp
            self._no_improve_count = 0
            if self._save_path:
                path = f"{self._save_path}_best"
                self.model.save(path)
                # Also save VecNormalize stats so they can be restored with the best model
                if hasattr(self.model.env, 'save'):
                    self.model.env.save(f"{self._save_path}_vecnorm.pkl")
                line += f"  ← best  saved → {path}"
        else:
            self._no_improve_count += 1
            if self._patience > 0:
                line += f"  (no improve {self._no_improve_count}/{self._patience})"

        if self.verbose:
            print(line)

        if self._patience > 0 and self._no_improve_count >= self._patience:
            print(f"\n  Early stop: no improvement for {self._patience} evaluations "
                  f"(best mean game pts: {self._best_mean_gp:+.2f})")
            return False

        return True


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation printer
# ──────────────────────────────────────────────────────────────────────────────

def _print_eval_results(model: MaskablePPO, n_episodes: int = 500) -> None:
    """Print aggregated evaluation stats across all defender types."""
    all_results = {name: evaluate(model, cls, n_episodes=n_episodes)
                   for name, cls in DEFENDERS.items()}
    mean_wr      = sum(r['win_rate']              for r in all_results.values()) / len(all_results)
    mean_gp      = sum(r['avg_game_pts']          for r in all_results.values()) / len(all_results)
    mean_pts     = sum(r['avg_pts']               for r in all_results.values()) / len(all_results)
    mean_piros   = sum(r['piros_pct']             for r in all_results.values()) / len(all_results)
    mean_s40100  = sum(r['silent_40100_pct']      for r in all_results.values()) / len(all_results)
    mean_sulti   = sum(r['silent_ulti_pct']       for r in all_results.values()) / len(all_results)
    mean_ds_ulti = sum(r['def_silent_ulti_pct']  for r in all_results.values()) / len(all_results)
    mean_ds40100 = sum(r['def_silent_40100_pct'] for r in all_results.values()) / len(all_results)
    sep = "─" * 88
    print(sep)
    print(f"  {'win%':>6}  {'avg gp':>7}  {'avg pts':>8}  {'piros%':>7}  "
          f"{'s40100%':>8}  {'sulti%':>7}  {'def_sulti%':>11}  {'def_s40100%':>12}")
    print(sep)
    print(f"  {mean_wr*100:>5.1f}%  {mean_gp:>+6.2f}   {mean_pts:>8.1f}  {mean_piros*100:>6.1f}%  "
          f"{mean_s40100*100:>7.1f}%  {mean_sulti*100:>6.1f}%  "
          f"{mean_ds_ulti*100:>10.1f}%  {mean_ds40100*100:>11.1f}%")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train Ulti declarer (MaskablePPO)')
    p.add_argument('--steps',     type=int, default=2_000_000,
                   help='Total training timesteps (default: 2_000_000)')
    p.add_argument('--n-envs',    type=int, default=4,
                   help='Parallel training envs (default: 4)')
    p.add_argument('--load',      default=None,
                   help='Path to a saved model to continue training from')
    p.add_argument('--save',      default='models/ulti',
                   help='Save prefix for checkpoints (default: models/ulti)')
    p.add_argument('--eval-only', action='store_true',
                   help='Skip training, only evaluate the --load model')
    p.add_argument('--patience',  type=int, default=5,
                   help='Early-stop after this many evals without improvement '
                        '(default: 5; set 0 to disable)')
    p.add_argument('--seed',      type=int, default=0)
    p.add_argument('--tb-suffix', default='',
                   help='Suffix appended to TensorBoard run name (used by train_all.py)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Create save directory ──────────────────────────────────────────────
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)

    # ── Eval-only mode ─────────────────────────────────────────────────────
    if args.eval_only:
        if not args.load:
            print("--eval-only requires --load <path>", file=sys.stderr)
            sys.exit(1)
        model = MaskablePPO.load(args.load)
        print(f"\nEvaluating {args.load}  (500 eps each) …")
        _print_eval_results(model, n_episodes=500)
        return

    # ── Create vectorised training envs (mixed pool) ───────────────────────
    vec_env = make_vec_env(
        _make_env(),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    # Normalize rewards to unit variance — stabilises training when the
    # game_points × 10 terminal bonus would otherwise create high-variance returns.
    # Observations are already in [0, 1] so we leave them untouched.
    vecnorm_path = f"{args.save}_vecnorm.pkl" if args.save else None
    if args.load and vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path} …")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # ── Model ──────────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=[256, 256])

    if args.load:
        print(f"Loading model from {args.load} …")
        model = MaskablePPO.load(args.load, env=vec_env)
    else:
        model = MaskablePPO(
            policy          = 'MlpPolicy',
            env             = vec_env,
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 128,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            ent_coef        = 0.01,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = 'runs/',
            verbose         = 0,
            seed            = args.seed,
        )

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        WinRateCallback(
            eval_freq       = max(20_000 // args.n_envs, 1000),
            n_eval_episodes = 200,
            save_path       = args.save,
            patience        = args.patience,
            verbose         = 1,
        ),
    ]
    if args.save:
        callbacks.append(
            CheckpointCallback(
                save_freq   = max(100_000 // args.n_envs, 1),
                save_path   = os.path.dirname(args.save) or '.',
                name_prefix = os.path.basename(args.save),
            )
        )

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\nTraining vs mixed defender pool "
          f"for {args.steps:,} steps on {args.n_envs} envs …\n")

    model.learn(
        total_timesteps     = args.steps,
        callback            = callbacks,
        reset_num_timesteps = args.load is None,
        tb_log_name         = '_'.join(filter(None, [os.path.basename(args.save) if args.save else 'parti', args.tb_suffix])),
    )

    # ── Save final model + VecNormalize stats ─────────────────────────────
    if args.save:
        final_path = f"{args.save}_final"
        model.save(final_path)
        vec_env.save(vecnorm_path)
        print(f"\nFinal model saved → {final_path}")
        print(f"VecNormalize stats saved → {vecnorm_path}")

    # ── Final evaluation ───────────────────────────────────────────────────
    best_model = MaskablePPO.load(f"{args.save}_best") if args.save else model
    print("\nFinal evaluation — best checkpoint (500 eps each, deterministic):")
    _print_eval_results(best_model, n_episodes=500)


if __name__ == '__main__':
    main()
