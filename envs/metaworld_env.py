"""Metaworld environment adapter for STORM.

Builds a Metaworld task with the same pipeline used by the DreamerV3 metaworld
sequential trainer (dreamer_sequential.make_env):

    Meta-World/MT1
      -> ProprioMultiImageObsWrapper (stack N cameras along channel axis)
      -> FirstTerminalObs (adds is_first / is_terminal)
      -> RewardTuningWrapperV2 ([-1, 1] -> [-1, 0])
      -> NormalizeActions ([-1, 1] box)
      -> TimeLimit

The STORM training loop expects a gymnasium vector env that produces:
    obs: (N, H, W, C) uint8 image
    info: dict with "life_loss" (always False for metaworld) and
          "episode_frame_number" (counted by this wrapper).

We also carry proprio through info["proprio"] so the training loop can push it
into the replay buffer.
"""
from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["EGL_LOG_LEVEL"] = "fatal"
import warnings
warnings.filterwarnings("ignore", message="Constant.*may be too high")
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

import sys
from pathlib import Path

import gymnasium
import gymnasium.vector
import numpy as np

# Make the top-level metaworld package importable.
_BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import metaworld
from metaworld.wrappers import ProprioMultiImageObsWrapper

# Reuse dreamer's metaworld reward + is_first/is_terminal wrappers so the
# reward shaping stays identical between dreamer and STORM.
from third_party.dreamerv3.envs import metaworld_wrappers as dreamer_mw_wrappers


class NormalizeActions(gymnasium.ActionWrapper):
    """Map agent actions in [-1, 1] to the underlying box range.

    Mirrors dreamer's wrappers.NormalizeActions so the continuous head can
    output squashed-normal samples in [-1, 1] regardless of the env range.
    """

    def __init__(self, env):
        super().__init__(env)
        low = env.action_space.low
        high = env.action_space.high
        self._mask = np.isfinite(low) & np.isfinite(high)
        self._low = np.where(self._mask, low, -1.0)
        self._high = np.where(self._mask, high, 1.0)
        self.action_space = gymnasium.spaces.Box(
            low=np.where(self._mask, -np.ones_like(low), low),
            high=np.where(self._mask, np.ones_like(high), high),
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        scaled = (action + 1.0) / 2.0 * (self._high - self._low) + self._low
        return np.where(self._mask, scaled, action)


class MetaworldSTORMWrapper(gymnasium.Wrapper):
    """Adapt a dreamer-style metaworld env to STORM's expected interface.

    - observation_space: Box(H, W, 3*num_cameras) uint8 (image only).
    - info["proprio"]: (proprio_dim,) float32 on every step/reset.
    - info["life_loss"]: always False (metaworld has no lives).
    - info["episode_frame_number"]: int step counter within the episode.

    Reward tuning is applied via dreamer's RewardTuningWrapperV2.
    """

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        image_space = env.observation_space["image"]
        self.observation_space = image_space
        self._proprio_dim = int(env.observation_space["proprio"].shape[0])
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

    @property
    def proprio_dim(self) -> int:
        return self._proprio_dim

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._step_count = 0
        info = dict(info) if isinstance(info, dict) else {}
        info["proprio"] = np.asarray(obs["proprio"], dtype=np.float32)
        info["life_loss"] = False
        info["episode_frame_number"] = self._step_count
        return np.asarray(obs["image"], dtype=np.uint8), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True
        info = dict(info) if isinstance(info, dict) else {}
        info["proprio"] = np.asarray(obs["proprio"], dtype=np.float32)
        info["life_loss"] = False
        info["episode_frame_number"] = self._step_count
        return (
            np.asarray(obs["image"], dtype=np.uint8),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


def build_single_metaworld_env(
    task_name: str,
    image_size: int,
    seed: int,
    camera_names=("topview", "front", "gripperPOV"),
    time_limit: int = 500,
    reward_range=(-1.0, 0.0),
) -> MetaworldSTORMWrapper:
    """Construct a single metaworld env matching dreamer's make_env recipe.

    `task_name` follows the dreamer convention, e.g. "metaworld_drawer-open-v3".
    """
    if task_name.startswith("metaworld_"):
        task = task_name.split("_", 1)[1]
    else:
        task = task_name

    env = gymnasium.make(
        "Meta-World/MT1",
        env_name=task,
        render_mode="rgb_array",
        max_episode_steps=time_limit,
    )
    env = ProprioMultiImageObsWrapper(
        env,
        image_height=image_size,
        image_width=image_size,
        camera_names=list(camera_names),
    )
    env = dreamer_mw_wrappers.FirstTerminalObs(env)
    env = dreamer_mw_wrappers.RewardTuningWrapperV2(
        env, target_reward_range=tuple(reward_range)
    )
    env = NormalizeActions(env)
    wrapped = MetaworldSTORMWrapper(env, max_episode_steps=time_limit)
    wrapped.reset(seed=seed)
    return wrapped


def build_metaworld_vec_env(
    task_name: str,
    image_size: int,
    num_envs: int,
    seed: int,
    camera_names=("topview", "front", "gripperPOV"),
    time_limit: int = 500,
    reward_range=(-1.0, 0.0),
) -> gymnasium.vector.SyncVectorEnv:
    """Return a SyncVectorEnv of `num_envs` metaworld envs.

    Metaworld uses a MuJoCo renderer and typically num_envs=1 here, so
    SyncVectorEnv avoids pickling the mujoco context across workers.
    """
    def _make(env_idx: int):
        def _thunk():
            return build_single_metaworld_env(
                task_name=task_name,
                image_size=image_size,
                seed=seed + env_idx,
                camera_names=camera_names,
                time_limit=time_limit,
                reward_range=reward_range,
            )
        return _thunk

    env_fns = [_make(i) for i in range(num_envs)]
    return gymnasium.vector.SyncVectorEnv(env_fns)
