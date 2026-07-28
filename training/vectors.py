"""
VecEnv builders for play and pre-game agents.

build_play_vec
    make_vec_env-backed VecNormalize for the play agent.

build_pregame_vec
    DummyVecEnv-backed VecNormalize for the pre-game agent.
    DummyVecEnv is required because the play_agent captured inside
    pregame_env_init is not picklable for SubprocVecEnv.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def build_play_vec(
    play_env_init: Callable,
    n_envs: int,
    seed: int,
    vecnorm_path: Optional[str] = None,
) -> VecNormalize:
    """Build a make_vec_env-backed VecNormalize for the play agent."""
    raw = make_vec_env(play_env_init, n_envs=n_envs, seed=seed)
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, raw)
        vec.training = True
    else:
        vec = VecNormalize(raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return vec


def build_pregame_vec(
    pregame_env_init: Callable,
    n_envs: int,
    vecnorm_path: Optional[str] = None,
) -> VecNormalize:
    """Build a DummyVecEnv-backed VecNormalize for the pre-game agent."""
    raw = DummyVecEnv([pregame_env_init] * n_envs)
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, raw)
        vec.training = True
    else:
        vec = VecNormalize(raw, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return vec
