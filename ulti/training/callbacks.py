"""
SB3 callbacks for periodic evaluation and best-model checkpointing.

PlayCallback
    Periodic evaluation and checkpointing for the play agent.

PreGameCallback
    Periodic evaluation and checkpointing for the pre-game agent.
"""
from __future__ import annotations

from typing import Callable, Optional

from stable_baselines3.common.callbacks import BaseCallback

from .registry import DEFENDERS, make_pregame_selector
from .evaluation import evaluate_play, evaluate_pregame, _mean


class PlayCallback(BaseCallback):
    """Periodic evaluation and best-model checkpointing for the play agent."""

    def __init__(
        self,
        play_env_cls,
        heuristic_selector: Callable,
        trump_lo: int,
        trump_hi: int,
        tb_prefix: str,
        pregame_agent=None,
        adversarial_defender_cls=None,
        silent_ulti: bool = False,
        silent_40100: bool = False,
        eval_freq: int = 20_000,
        n_eval: int = 200,
        save_path: Optional[str] = None,
        patience: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._play_env_cls    = play_env_cls
        self._heuristic       = heuristic_selector
        self._trump_lo        = trump_lo
        self._trump_hi        = trump_hi
        self._tb_prefix       = tb_prefix
        self._pregame_agent   = pregame_agent
        self._adversarial_cls = adversarial_defender_cls
        self._silent_ulti     = silent_ulti
        self._silent_40100    = silent_40100
        self._eval_freq       = eval_freq
        self._n_eval          = n_eval
        self._save_path       = save_path
        self._patience        = patience
        self._best_gp         = -float('inf')
        self._no_imp          = 0

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        selector = (
            make_pregame_selector(self._pregame_agent, self._trump_lo, self._trump_hi)
            if self._pregame_agent is not None else self._heuristic
        )
        pregame_agent = self._pregame_agent

        if self._adversarial_cls is not None:
            r = evaluate_play(self.model, self._play_env_cls, self._heuristic,
                              self._adversarial_cls, self._n_eval,
                              trump_selector=selector, pregame_agent=pregame_agent)
            mean_gp      = r['avg_gp']
            mean_wr      = r['win_rate']
            decl_s_ulti  = r['decl_silent_ulti']
            def_s_ulti   = r['def_silent_ulti']
            decl_s40100  = r['decl_silent_40100']
            def_s40100   = r['def_silent_40100']
        else:
            results = {
                name: evaluate_play(self.model, self._play_env_cls, self._heuristic,
                                    cls, self._n_eval,
                                    trump_selector=selector, pregame_agent=pregame_agent)
                for name, cls in DEFENDERS.items()
            }
            mean_gp     = _mean(results, 'avg_gp')
            mean_wr     = _mean(results, 'win_rate')
            decl_s_ulti = _mean(results, 'decl_silent_ulti')
            def_s_ulti  = _mean(results, 'def_silent_ulti')
            decl_s40100 = _mean(results, 'decl_silent_40100')
            def_s40100  = _mean(results, 'def_silent_40100')

        ulti_tag = 'silent_ulti' if self._silent_ulti else 'ulti'
        self.logger.record(f'{self._tb_prefix}/play_mean_gp',         mean_gp)
        self.logger.record(f'{self._tb_prefix}/play_mean_wr',         mean_wr)
        self.logger.record(f'{self._tb_prefix}/play_decl_{ulti_tag}', decl_s_ulti)
        if self._silent_ulti:
            self.logger.record(f'{self._tb_prefix}/play_def_{ulti_tag}', def_s_ulti)
        if self._silent_40100:
            self.logger.record(f'{self._tb_prefix}/play_decl_silent_40100', decl_s40100)
            self.logger.record(f'{self._tb_prefix}/play_def_silent_40100',  def_s40100)
        self.logger.dump(self.num_timesteps)

        line = (f"  [play] step {self.num_timesteps:>9,}  "
                f"mean_gp={mean_gp:+.2f}  wr={mean_wr:.3f}")
        if mean_gp > self._best_gp:
            self._best_gp = mean_gp
            self._no_imp  = 0
            if self._save_path:
                self.model.save(f"{self._save_path}_play_best")
                line += "  ← best saved"
        else:
            self._no_imp += 1
            if self._patience > 0:
                line += f"  (no imp {self._no_imp}/{self._patience})"
        if self.verbose:
            print(line)
        return not (self._patience > 0 and self._no_imp >= self._patience)


class PreGameCallback(BaseCallback):
    """Periodic evaluation and best-model checkpointing for the pre-game agent."""

    def __init__(
        self,
        pregame_env_cls,
        play_model,
        trump_lo: int,
        tb_prefix: str,
        adversarial_defender_cls=None,
        eval_freq: int = 10_000,
        n_eval: int = 200,
        save_path: Optional[str] = None,
        patience: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._pregame_env_cls = pregame_env_cls
        self._play_model      = play_model
        self._trump_lo        = trump_lo
        self._tb_prefix       = tb_prefix
        self._adversarial_cls = adversarial_defender_cls
        self._eval_freq       = eval_freq
        self._n_eval          = n_eval
        self._save_path       = save_path
        self._patience        = patience
        self._best_reward     = -float('inf')
        self._no_imp          = 0

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        if self._adversarial_cls is not None:
            r = evaluate_pregame(self.model, self._play_model,
                                 self._pregame_env_cls, self._adversarial_cls,
                                 self._trump_lo, self._n_eval)
            mean_rew   = r['avg_reward']
            mean_passz = r['passz_rate']
            mean_wr    = r['win_rate']
            mean_piros = r['piros_pct']
        else:
            results = {
                name: evaluate_pregame(self.model, self._play_model,
                                       self._pregame_env_cls, cls,
                                       self._trump_lo, self._n_eval)
                for name, cls in DEFENDERS.items()
            }
            mean_rew   = _mean(results, 'avg_reward')
            mean_passz = _mean(results, 'passz_rate')
            mean_wr    = _mean(results, 'win_rate')
            mean_piros = _mean(results, 'piros_pct')

        self.logger.record(f'{self._tb_prefix}/pregame_mean_reward', mean_rew)
        self.logger.record(f'{self._tb_prefix}/pregame_mean_passz',  mean_passz)
        self.logger.record(f'{self._tb_prefix}/pregame_mean_wr',     mean_wr)
        self.logger.record(f'{self._tb_prefix}/pregame_mean_piros',  mean_piros)
        self.logger.dump(self.num_timesteps)

        line = (f"  [pregame] step {self.num_timesteps:>9,}  "
                f"passz={mean_passz:.1%}  wr={mean_wr:.3f}  "
                f"piros={mean_piros:.1%}  mean_rew={mean_rew:+.1f}")
        if mean_rew > self._best_reward:
            self._best_reward = mean_rew
            self._no_imp      = 0
            if self._save_path:
                self.model.save(f"{self._save_path}_pregame_best")
                line += "  ← best saved"
        else:
            self._no_imp += 1
            if self._patience > 0:
                line += f"  (no imp {self._no_imp}/{self._patience})"
        if self.verbose:
            print(line)
        return not (self._patience > 0 and self._no_imp >= self._patience)
