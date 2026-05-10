"""Sequential STORM training with Experience Replay (ER) and cross-task evaluation.

Mirrors ``third_party/dreamerv3/er_training/dreamer_sequential_er.py`` but
for STORM, building on ``third_party/STORM/train_metaworld_sequential.py``.

What ER adds on top of the naive sequential trainer:
  * For task N>1, we reservoir-sample episodes from each previous task's
    on-disk ``train_eps/*.npz`` (budget = ``er_buffer_ratio`` × that task's
    BufferMaxLength) and pack them into an in-memory ER buffer.
  * Every world-model update samples a mix: ``batch_size`` fresh transitions
    from the current task's ring buffer, plus ``er_batch_size`` transitions
    drawn uniformly from the ER buffer. Imagination context for actor-critic
    updates is sampled the same way (current + ER mixed).
  * The shared world-model core (encoder, transformer, dist_head, image
    decoder, proprio encoder/decoder) keeps its Adam moments across tasks
    (see :func:`reset_task_heads`); reward/termination heads + actor-critic
    are reinitialised per task with fresh Adam state. This is the same
    "fresh state for fresh modules, preserved state for preserved modules"
    rule that ``dreamer_sequential_er.py`` follows.

Why a parallel ER buffer instead of pre-loading old episodes into the main
ring: STORM's main ``ReplayBuffer`` is a fixed-size ring of (obs, action,
reward, term, proprio) on GPU/CPU. If we mixed ER transitions in, they'd
get evicted by current-task data and we'd lose the ER guarantee. The
``ERReplayBuffer`` here is a separate immutable buffer holding only
sampled past-task transitions; it never grows or shrinks during training.

Split-checkpoint layout per task (identical to the naive seq trainer):
    <logdir>/task{N}_<name>/
        rssm.pt           — shared core (no reward/termination)
        task_heads.pt      — reward_decoder + termination_decoder
        actor_critic.pt    — ActorCriticAgent state dict
        manifest.json      — {iter, env_step, episodes_done, wandb_run_id}

Example:
    python train_metaworld_sequential_er.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --task-buffer-sizes 25000 62500 \\
        --er-buffer-ratio 0.05 \\
        --logdir ./runs/seq_storm_er \\
        --config_path config_files/STORM_metaworld.yaml \\
        --wandb-entity haoyu-a2i \\
        --wandb-project Metaworld_STORM_Sequential_ER
"""
import argparse
import gc
import json
import os
import sys
import warnings
from collections import deque
from pathlib import Path

import colorama
import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm

# Make the parent STORM dir importable so we reuse modules + helpers.
_HERE = Path(__file__).resolve().parent
_STORM_ROOT = _HERE.parent
sys.path.insert(0, str(_STORM_ROOT))

import agents
import envs.metaworld_env as mw_env
from replay_buffer import ReplayBuffer
from sub_models.world_models import (
    WorldModel,
    RewardDecoder,
    TerminationDecoder,
)
from utils import Logger, load_config, seed_np_torch

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# ===========================================================================
#  Logger wrappers (verbatim from train_metaworld_sequential.py)
# ===========================================================================

class WandbTBLogger:
    """Wraps STORM's TensorBoard Logger and mirrors scalars to wandb."""

    def __init__(self, tb_logger: Logger, use_wandb: bool):
        self._tb = tb_logger
        self._use_wandb = use_wandb and wandb is not None

    def log(self, tag, value):
        self._tb.log(tag, value)
        if not self._use_wandb:
            return
        if "video" in tag or "images" in tag or "hist" in tag:
            return
        try:
            wandb.log({tag: float(value)}, commit=False)
        except Exception:
            pass

    def commit(self, step=None):
        if self._use_wandb:
            try:
                if step is not None:
                    wandb.log({}, commit=True, step=step)
                else:
                    wandb.log({}, commit=True)
            except Exception:
                pass


class PrefixedLogger:
    def __init__(self, real_logger: WandbTBLogger, prefix: str):
        self._real = real_logger
        self._prefix = prefix

    def log(self, tag, value):
        self._real.log(f"{self._prefix}/{tag}", value)

    def commit(self, step=None):
        pass


# ===========================================================================
#  World-model checkpoint split/merge (verbatim from sequential trainer)
# ===========================================================================

TASK_HEAD_PREFIXES = ("reward_decoder.", "termination_decoder.")


def _is_task_head_key(k: str) -> bool:
    return any(k.startswith(p) for p in TASK_HEAD_PREFIXES)


def split_world_model_state_dict(sd: dict):
    shared = {k: v for k, v in sd.items() if not _is_task_head_key(k)}
    heads = {k: v for k, v in sd.items() if _is_task_head_key(k)}
    return shared, heads


def save_world_model_split(world_model, rssm_path: Path, heads_path: Path):
    shared_sd, heads_sd = split_world_model_state_dict(world_model.state_dict())
    torch.save(shared_sd, rssm_path)
    torch.save(heads_sd, heads_path)


def load_rssm_into(world_model, rssm_path: Path):
    shared = torch.load(rssm_path, map_location="cuda")
    result = world_model.load_state_dict(shared, strict=False)
    bad_missing = [k for k in result.missing_keys if not _is_task_head_key(k)]
    if bad_missing:
        raise RuntimeError(
            f"rssm.pt was missing shared-core keys: {bad_missing[:5]}..."
        )
    if result.unexpected_keys:
        raise RuntimeError(
            f"rssm.pt had unexpected keys (should only be shared-core): "
            f"{result.unexpected_keys[:5]}..."
        )


def load_task_heads_into(world_model, heads_path: Path):
    heads = torch.load(heads_path, map_location="cuda")
    bad = [k for k in heads if not _is_task_head_key(k)]
    if bad:
        raise RuntimeError(f"task_heads.pt contained non-task-head keys: {bad[:5]}")
    result = world_model.load_state_dict(heads, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"task_heads.pt had unexpected keys: {result.unexpected_keys[:5]}"
        )


def reset_task_heads(world_model, conf):
    """Rebuild reward/termination decoders in place while PRESERVING the
    shared core's Adam moments.

    Same surgical splice as the naive sequential trainer: drop OLD head
    state entries from optimizer.state, swap modules, append new params.
    Shared-core entries in optimizer.state are untouched, so encoder /
    transformer / dist_head / image_decoder / proprio_* keep their
    calibrated (m, v) across the task boundary.
    """
    hidden = int(conf.Models.WorldModel.TransformerHiddenDim)
    stoch_flat = int(world_model.stoch_flattened_dim)

    old_head_params = (
        list(world_model.reward_decoder.parameters())
        + list(world_model.termination_decoder.parameters())
    )
    old_ids = {id(p) for p in old_head_params}

    world_model.reward_decoder = RewardDecoder(
        num_classes=255,
        embedding_size=stoch_flat,
        transformer_hidden_dim=hidden,
    ).cuda()
    world_model.termination_decoder = TerminationDecoder(
        embedding_size=stoch_flat,
        transformer_hidden_dim=hidden,
    ).cuda()

    opt = world_model.optimizer
    pg = opt.param_groups[0]
    for p in old_head_params:
        opt.state.pop(p, None)
    pg["params"] = [p for p in pg["params"] if id(p) not in old_ids]
    pg["params"].extend(world_model.reward_decoder.parameters())
    pg["params"].extend(world_model.termination_decoder.parameters())


def snapshot_task_heads(world_model):
    sd = {}
    for k, v in world_model.state_dict().items():
        if _is_task_head_key(k):
            sd[k] = v.detach().cpu().clone()
    return sd


def restore_task_heads(world_model, snapshot_sd: dict):
    restored = {k: v.to("cuda") for k, v in snapshot_sd.items()}
    world_model.load_state_dict(restored, strict=False)


# ===========================================================================
#  Experience replay buffer
# ===========================================================================

def reservoir_sample_episode_files(directory: Path, budget: int, seed: int):
    """Reservoir-sample ``.npz`` episode files until ``budget`` transitions
    are accumulated. Mirrors ``dreamer_sequential_er.reservoir_sample_episodes``
    but only returns the loaded numpy episodes (no key renaming).
    """
    directory = Path(directory).expanduser()
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.npz"))
    if not files:
        return []
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(files))

    eps = []
    total = 0
    for idx in order:
        if total >= budget:
            break
        path = files[idx]
        try:
            with np.load(path) as raw:
                ep = {k: raw[k] for k in raw.files}
        except Exception as e:
            print(f"[ER] could not load {path.name}: {e}")
            continue
        n = int(ep["reward"].shape[0])
        if n < 1:
            continue
        eps.append(ep)
        total += n
    return eps


class ERReplayBuffer:
    """Immutable in-memory ring of past-task transitions, sampled with the
    same per-trajectory contiguous-window scheme STORM's ``ReplayBuffer``
    uses. Each "trajectory" here is one full episode from a previous task.

    Layout: numpy/torch arrays of shape (T, *), where T = sum of episode
    lengths. We additionally keep a per-trajectory ``starts`` / ``ends``
    index so that sampling never crosses an episode boundary — matching
    the implicit assumption STORM's main buffer makes via ``num_envs``
    column slicing.
    """

    def __init__(self, episodes, obs_shape, action_dim, proprio_dim,
                 store_on_gpu=False):
        self.use_proprio = proprio_dim > 0
        self.action_dim = action_dim
        self.store_on_gpu = store_on_gpu
        self._device = "cuda" if store_on_gpu else "cpu"

        if not episodes:
            self.length = 0
            self.starts = np.zeros((0,), dtype=np.int64)
            self.ends = np.zeros((0,), dtype=np.int64)
            return

        obs_chunks = [ep["obs"] for ep in episodes]
        act_chunks = [ep["action"] for ep in episodes]
        rew_chunks = [ep["reward"].astype(np.float32) for ep in episodes]
        term_chunks = [ep["termination"].astype(np.float32) for ep in episodes]
        prop_chunks = (
            [ep["proprio"].astype(np.float32) for ep in episodes]
            if self.use_proprio else None
        )

        lengths = np.asarray([c.shape[0] for c in rew_chunks], dtype=np.int64)
        ends = np.cumsum(lengths)
        starts = ends - lengths
        self.starts = starts
        self.ends = ends
        self.length = int(ends[-1])

        obs = np.concatenate(obs_chunks, axis=0).astype(np.uint8)
        action = np.concatenate(act_chunks, axis=0).astype(np.float32)
        reward = np.concatenate(rew_chunks, axis=0).astype(np.float32)
        termination = np.concatenate(term_chunks, axis=0).astype(np.float32)

        # Action shape: STORM stores discrete actions as scalars and continuous
        # as (A,). We always coerce to (T, A) here for consistency, then
        # squeeze on the way out if the main buffer would have stored scalars.
        if action.ndim == 1:
            action = action[:, None]
        if action.shape[-1] != action_dim and action_dim == 1:
            # Discrete env: keep as (T, 1) — sampler will squeeze if needed.
            action = action.reshape(-1, 1)

        if store_on_gpu:
            self._obs = torch.from_numpy(obs).to("cuda")
            self._action = torch.from_numpy(action).to("cuda")
            self._reward = torch.from_numpy(reward).to("cuda")
            self._termination = torch.from_numpy(termination).to("cuda")
            if self.use_proprio:
                self._proprio = torch.from_numpy(
                    np.concatenate(prop_chunks, axis=0)
                ).to("cuda")
        else:
            self._obs = obs
            self._action = action
            self._reward = reward
            self._termination = termination
            if self.use_proprio:
                self._proprio = np.concatenate(prop_chunks, axis=0)

    def __len__(self):
        return self.length

    def can_sample(self, batch_length):
        if self.length == 0:
            return False
        # We need at least one trajectory long enough for the window.
        return bool(((self.ends - self.starts) >= batch_length).any())

    def _sample_indexes(self, batch_size, batch_length):
        """Draw start indices that fit entirely within a single trajectory."""
        # Filter trajectories long enough for the window.
        valid = (self.ends - self.starts) >= batch_length
        if not valid.any():
            raise RuntimeError(
                f"ER buffer has no trajectory of length >= {batch_length}"
            )
        traj_starts = self.starts[valid]
        traj_lengths = (self.ends - self.starts)[valid]

        # Pick trajectories proportional to their length (uniform-over-
        # transitions sampling, same as the main buffer).
        probs = traj_lengths.astype(np.float64) / traj_lengths.sum()
        chosen = np.random.choice(len(traj_starts), size=batch_size, p=probs)
        offsets = np.array([
            np.random.randint(0, traj_lengths[c] - batch_length + 1)
            for c in chosen
        ], dtype=np.int64)
        return traj_starts[chosen] + offsets

    @torch.no_grad()
    def sample(self, batch_size, batch_length):
        """Return a sample matching the main buffer's output convention.

        Returns ``(obs, action, reward, termination[, proprio])`` where:
          obs:          (B, T, C, H, W) float in [0, 1] on cuda
          action:       (B, T) or (B, T, A) float32 on cuda
          reward:       (B, T) float32 on cuda
          termination:  (B, T) float32 on cuda
          proprio:      (B, T, D) float32 on cuda  (only if use_proprio)
        """
        idxs = self._sample_indexes(batch_size, batch_length)
        if self.store_on_gpu:
            obs = torch.stack([self._obs[i:i + batch_length] for i in idxs])
            action = torch.stack([self._action[i:i + batch_length] for i in idxs])
            reward = torch.stack([self._reward[i:i + batch_length] for i in idxs])
            termination = torch.stack([self._termination[i:i + batch_length] for i in idxs])
            if self.use_proprio:
                proprio = torch.stack(
                    [self._proprio[i:i + batch_length] for i in idxs]
                )
        else:
            obs = np.stack([self._obs[i:i + batch_length] for i in idxs])
            action = np.stack([self._action[i:i + batch_length] for i in idxs])
            reward = np.stack([self._reward[i:i + batch_length] for i in idxs])
            termination = np.stack([self._termination[i:i + batch_length] for i in idxs])
            if self.use_proprio:
                proprio = np.stack([self._proprio[i:i + batch_length] for i in idxs])
            obs = torch.from_numpy(obs).cuda()
            action = torch.from_numpy(action).cuda()
            reward = torch.from_numpy(reward).cuda()
            termination = torch.from_numpy(termination).cuda()
            if self.use_proprio:
                proprio = torch.from_numpy(proprio).cuda()

        # Match main buffer's output: float32 in [0, 1], (B, T, C, H, W).
        obs = obs.float() / 255.0
        obs = rearrange(obs, "B T H W C -> B T C H W")
        # Squeeze trailing dim for discrete-action setups (action_dim == 1
        # on the main buffer's scalar storage convention).
        if action.dim() == 3 and action.shape[-1] == 1 and self.action_dim == 1:
            action = action.squeeze(-1)
        if self.use_proprio:
            return obs, action, reward, termination, proprio
        return obs, action, reward, termination


def build_er_buffer_for_task(
    base_logdir: Path,
    task_idx: int,
    tasks,
    task_buffer_sizes,
    er_buffer_ratio: float,
    er_seed: int,
    obs_shape,
    action_dim: int,
    proprio_dim: int,
    store_on_gpu: bool,
):
    """Walk every previous task's ``train_eps`` directory, reservoir-sample
    episodes up to ``er_buffer_ratio * buffer_size_of_that_task`` transitions,
    and pack the union into a single ``ERReplayBuffer``."""
    if task_idx == 0 or er_buffer_ratio <= 0:
        return None
    all_episodes = []
    per_task_summary = []
    for j in range(task_idx):
        prev_buffer_size = int(task_buffer_sizes[j])
        budget = int(er_buffer_ratio * prev_buffer_size)
        if budget < 1:
            per_task_summary.append((j, 0, 0))
            continue
        prev_dir = base_logdir / f"task{j+1}_{tasks[j]}" / "train_eps"
        eps = reservoir_sample_episode_files(prev_dir, budget, seed=er_seed + j)
        if not eps:
            per_task_summary.append((j, 0, 0))
            continue
        transitions = sum(int(e["reward"].shape[0]) for e in eps)
        per_task_summary.append((j, len(eps), transitions))
        all_episodes.extend(eps)

    if not all_episodes:
        print(colorama.Fore.YELLOW
              + ">>> ER: no episodes loaded (ratio too small or no prior data)"
              + colorama.Style.RESET_ALL)
        return None

    er_buf = ERReplayBuffer(
        all_episodes,
        obs_shape=obs_shape,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        store_on_gpu=store_on_gpu,
    )
    print(colorama.Fore.GREEN
          + f">>> ER: loaded {len(er_buf)} transitions from {len(all_episodes)} "
            f"episodes across {task_idx} previous task(s)"
          + colorama.Style.RESET_ALL)
    for j, n_eps, n_trans in per_task_summary:
        print(f"     task{j+1} ({tasks[j]}): {n_eps} episodes, {n_trans} transitions")
    return er_buf


def _concat_buffer_samples(main_sample, er_sample, use_proprio):
    """Concatenate a main-buffer sample with an ER-buffer sample along
    the batch dim. Both must already be on cuda and have matching shapes
    along (T, ...).
    """
    if er_sample is None:
        return main_sample
    if use_proprio:
        m_obs, m_act, m_rew, m_term, m_prop = main_sample
        e_obs, e_act, e_rew, e_term, e_prop = er_sample
        return (
            torch.cat([m_obs, e_obs], dim=0),
            torch.cat([m_act, e_act], dim=0),
            torch.cat([m_rew, e_rew], dim=0),
            torch.cat([m_term, e_term], dim=0),
            torch.cat([m_prop, e_prop], dim=0),
        )
    m_obs, m_act, m_rew, m_term = main_sample
    e_obs, e_act, e_rew, e_term = er_sample
    return (
        torch.cat([m_obs, e_obs], dim=0),
        torch.cat([m_act, e_act], dim=0),
        torch.cat([m_rew, e_rew], dim=0),
        torch.cat([m_term, e_term], dim=0),
    )


# ===========================================================================
#  Sequential progress tracking
# ===========================================================================

def _progress_path(base_logdir: Path) -> Path:
    return base_logdir / "sequential_progress.json"


def save_sequential_progress(
    base_logdir: Path, task_idx: int,
    global_env_step_at_start=None,
    completed=False,
    global_env_step_at_end=None,
    task_name=None,
    wandb_run_id=None,
):
    path = _progress_path(base_logdir)
    prog = json.loads(path.read_text()) if path.exists() else {"tasks": {}}
    entry = prog["tasks"].setdefault(str(task_idx), {})
    if task_name is not None:
        entry["task_name"] = task_name
    if global_env_step_at_start is not None:
        entry["global_env_step_at_start"] = int(global_env_step_at_start)
    if completed:
        entry["completed"] = True
    if global_env_step_at_end is not None:
        entry["global_env_step_at_end"] = int(global_env_step_at_end)
    if wandb_run_id is not None:
        prog["wandb_run_id"] = wandb_run_id
    path.write_text(json.dumps(prog, indent=2))


def load_sequential_progress(base_logdir: Path):
    path = _progress_path(base_logdir)
    if path.exists():
        return json.loads(path.read_text())
    return None


# ===========================================================================
#  Model builders
# ===========================================================================

def build_world_model(conf, action_dim, proprio_dim):
    return WorldModel(
        in_channels=conf.Models.WorldModel.InChannels,
        action_dim=action_dim,
        transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
        transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
        transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
        transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
        continuous_action=conf.Models.WorldModel.ContinuousAction,
        proprio_dim=proprio_dim,
        proprio_hidden_dim=conf.Models.WorldModel.ProprioHiddenDim,
        proprio_embed_dim=conf.Models.WorldModel.ProprioEmbedDim,
        input_size=conf.BasicSettings.ImageSize,
    ).cuda()


def build_agent(conf, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=32 * 32 + conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
        continuous_action=conf.Models.WorldModel.ContinuousAction,
    ).cuda()


# ===========================================================================
#  Cross-task evaluation (verbatim from sequential trainer)
# ===========================================================================

@torch.no_grad()
def _rollout_one_episode(world_model, agent, env, image_size):
    world_model.eval()
    agent.eval()

    obs, info = env.reset()
    proprio = np.asarray(info["proprio"], dtype=np.float32) if "proprio" in info else None

    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    context_proprio = deque(maxlen=16)

    total_reward = 0.0
    steps = 0
    success = 0.0
    done = False

    while not done:
        if len(context_action) == 0:
            action = env.action_space.sample().astype(np.float32)
        else:
            ctx_obs = torch.cat(list(context_obs), dim=1)
            ctx_proprio = None
            if world_model.use_proprio:
                ctx_proprio = torch.cat(list(context_proprio), dim=1)
            context_latent = world_model.encode_obs(ctx_obs, ctx_proprio)
            ctx_action = np.stack(list(context_action), axis=1)
            ctx_action = torch.tensor(ctx_action, dtype=torch.float32, device="cuda")
            prior_flat, last_dist_feat = world_model.calc_last_dist_feat(
                context_latent, ctx_action,
            )
            action = agent.sample_as_env_action(
                torch.cat([prior_flat, last_dist_feat], dim=-1),
                greedy=True,
            )

        ctx_push = rearrange(
            torch.tensor(obs, dtype=torch.float32, device="cuda"),
            "B H W C -> B 1 C H W",
        ) / 255.0
        context_obs.append(ctx_push)
        if world_model.use_proprio and proprio is not None:
            context_proprio.append(
                torch.tensor(proprio, dtype=torch.float32, device="cuda").unsqueeze(1)
            )
        context_action.append(action)

        obs, reward, term, trunc, info = env.step(action)
        proprio = np.asarray(info["proprio"], dtype=np.float32) if "proprio" in info else None
        total_reward += float(np.asarray(reward).sum())
        steps += 1
        succ_val = 0.0
        for key in ("success", "is_success"):
            if key in info:
                v = info[key]
                if hasattr(v, "__len__"):
                    v = v[0]
                succ_val = max(succ_val, float(v))
        fi = info.get("final_info") if isinstance(info, dict) else None
        if fi is not None:
            for fi_i in (fi if hasattr(fi, "__iter__") else [fi]):
                if isinstance(fi_i, dict):
                    for key in ("success", "is_success"):
                        if key in fi_i:
                            succ_val = max(succ_val, float(fi_i[key]))
        success = max(success, succ_val)
        done = bool(np.asarray(term).any() or np.asarray(trunc).any())

    return total_reward, steps, success


def evaluate_all_tasks_so_far(
    task_idx, tasks, conf, world_model, agent,
    eval_env_cache, base_logdir, logger, episodes, global_env_step,
):
    current_heads = snapshot_task_heads(world_model)

    for j in range(task_idx + 1):
        task_name = tasks[j]
        prefix = f"eval/task{j+1}_{task_name}"
        prefixed = PrefixedLogger(logger, prefix)

        if j not in eval_env_cache:
            eval_env_cache[j] = mw_env.build_metaworld_vec_env(
                task_name=task_name,
                image_size=conf.BasicSettings.ImageSize,
                num_envs=1,
                seed=int(conf.BasicSettings.Seed) + 1000 + j,
                camera_names=tuple(conf.Env.CameraNames),
                time_limit=conf.Env.TimeLimit,
                reward_range=(conf.Env.RewardMin, conf.Env.RewardMax),
            )
        env_j = eval_env_cache[j]

        if j == task_idx:
            use_agent = agent
        else:
            prev_task_dir = base_logdir / f"task{j+1}_{tasks[j]}"
            heads_path = prev_task_dir / "task_heads.pt"
            ac_path = prev_task_dir / "actor_critic.pt"
            if not heads_path.exists() or not ac_path.exists():
                print(colorama.Fore.YELLOW
                      + f"  [eval] WARNING: missing ckpts for task {j+1} "
                        f"({tasks[j]}), skipping"
                      + colorama.Style.RESET_ALL)
                continue
            load_task_heads_into(world_model, heads_path)
            tmp_agent = build_agent(conf, action_dim=agent.action_dim)
            tmp_agent.load_state_dict(torch.load(ac_path, map_location="cuda"))
            tmp_agent.eval()
            use_agent = tmp_agent

        rewards, successes, lengths = [], [], []
        try:
            for _ in range(max(1, int(episodes))):
                r, s, suc = _rollout_one_episode(
                    world_model, use_agent, env_j,
                    image_size=conf.BasicSettings.ImageSize,
                )
                rewards.append(r)
                lengths.append(s)
                successes.append(suc)
        finally:
            if j != task_idx:
                restore_task_heads(world_model, current_heads)
                del tmp_agent  # noqa: F821

        mean_r = float(np.mean(rewards))
        mean_s = float(np.mean(successes))
        mean_l = float(np.mean(lengths))
        prefixed.log("episode_reward", mean_r)
        prefixed.log("episode_success", mean_s)
        prefixed.log("episode_length", mean_l)

        tag = colorama.Fore.GREEN + "[eval]" + colorama.Style.RESET_ALL
        print(
            f"  {tag} curr=T{task_idx+1}({tasks[task_idx]}) "
            f"-> eval=T{j+1}({task_name}): "
            f"R={mean_r:.2f} S={mean_s:.2f} L={mean_l:.0f}  "
            f"@g={global_env_step}"
        )

    if hasattr(logger, "commit"):
        logger.commit(step=int(global_env_step))


# ===========================================================================
#  Per-task training loop with ER mixing
# ===========================================================================

def train_one_task(
    task_idx, num_tasks, tasks, conf,
    world_model, agent, replay_buffer,
    er_buffer,
    er_batch_size,
    er_imagine_batch_size,
    logger, base_logdir,
    global_env_step_offset, max_env_steps,
    eval_every_env_steps, eval_episodes,
    save_every_env_steps,
    start_iter=0, episodes_done_init=0,
    episode_dir=None,
    buffer_max_length=None,
):
    """STORM joint training loop with current-task + ER mixing.

    For each world-model update (and each agent imagination context
    sample), we draw ``batch_size`` transitions from the current task's
    ring buffer and ``er_batch_size`` from the ER buffer (if any), then
    concatenate along the batch dim. The world model and agent see one
    homogeneous batch — STORM's losses are batch-mean reductions, so
    mixing is loss-equivalent to weighting current vs ER by
    ``batch_size : er_batch_size``.
    """
    task_name = tasks[task_idx]
    task_dir = base_logdir / f"task{task_idx+1}_{task_name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    save_episodes = episode_dir is not None
    if save_episodes:
        os.makedirs(episode_dir, exist_ok=True)

    num_envs = conf.JointTrainAgent.NumEnvs
    image_size = conf.BasicSettings.ImageSize
    train_dyn_every = conf.JointTrainAgent.TrainDynamicsEverySteps
    train_agent_every = conf.JointTrainAgent.TrainAgentEverySteps
    batch_size = conf.JointTrainAgent.BatchSize
    demo_batch_size = (conf.JointTrainAgent.DemonstrationBatchSize
                       if conf.JointTrainAgent.UseDemonstration else 0)
    batch_length = conf.JointTrainAgent.BatchLength
    imagine_batch_size = conf.JointTrainAgent.ImagineBatchSize
    imagine_demo_batch_size = (conf.JointTrainAgent.ImagineDemonstrationBatchSize
                               if conf.JointTrainAgent.UseDemonstration else 0)
    imagine_context_length = conf.JointTrainAgent.ImagineContextLength
    imagine_batch_length = conf.JointTrainAgent.ImagineBatchLength
    seed = conf.BasicSettings.Seed

    # ER is only mixed when the buffer is present AND has enough length
    # for the requested window. If the ER buffer fails the length check
    # for any of the two windows we use, we silently skip mixing for that
    # window — better than crashing mid-training.
    er_active = er_buffer is not None and len(er_buffer) > 0
    if er_active and not er_buffer.can_sample(batch_length):
        print(colorama.Fore.YELLOW
              + f">>> ER: buffer present but no trajectory >= "
                f"BatchLength={batch_length}; disabling ER for WM update."
              + colorama.Style.RESET_ALL)
    if er_active and not er_buffer.can_sample(imagine_context_length):
        print(colorama.Fore.YELLOW
              + f">>> ER: buffer present but no trajectory >= "
                f"ImagineContextLength={imagine_context_length}; "
                f"disabling ER for imagination context."
              + colorama.Style.RESET_ALL)

    vec_env = mw_env.build_metaworld_vec_env(
        task_name=task_name,
        image_size=image_size,
        num_envs=num_envs,
        seed=seed + task_idx * 10007,
        camera_names=tuple(conf.Env.CameraNames),
        time_limit=conf.Env.TimeLimit,
        reward_range=(conf.Env.RewardMin, conf.Env.RewardMax),
    )
    print(colorama.Fore.YELLOW + f"Current env: {task_name}"
          + colorama.Style.RESET_ALL)

    sum_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs, dtype=np.int64)
    episodes_done = episodes_done_init

    current_obs, current_info = vec_env.reset()
    current_proprio = np.asarray(current_info["proprio"], dtype=np.float32)

    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    context_proprio = deque(maxlen=16)

    current_episode = [
        {"obs": [], "action": [], "reward": [], "termination": [], "proprio": []}
        for _ in range(num_envs)
    ]

    total_iters = max_env_steps // num_envs
    eval_every_iters = max(1, eval_every_env_steps // num_envs)
    save_every_iters = max(1, save_every_env_steps // num_envs)

    pbar = tqdm(
        total=total_iters, initial=start_iter,
        desc=f">>> T{task_idx+1}/{num_tasks} {task_name}",
        unit="step", dynamic_ncols=True,
    )

    eval_env_cache = {}

    def _global_step(iter_i):
        return global_env_step_offset + iter_i * num_envs

    try:
        for total_steps in range(start_iter, total_iters):
            gstep = _global_step(total_steps)

            # --------- act ---------
            if replay_buffer.ready():
                world_model.eval()
                agent.eval()
                with torch.no_grad():
                    if len(context_action) == 0:
                        action = vec_env.action_space.sample().astype(np.float32)
                    else:
                        ctx_obs = torch.cat(list(context_obs), dim=1)
                        ctx_proprio = None
                        if replay_buffer.use_proprio:
                            ctx_proprio = torch.cat(list(context_proprio), dim=1)
                        context_latent = world_model.encode_obs(ctx_obs, ctx_proprio)
                        mca = np.stack(list(context_action), axis=1)
                        mca = torch.tensor(mca, dtype=torch.float32, device="cuda")
                        prior_flat, last_dist_feat = world_model.calc_last_dist_feat(
                            context_latent, mca,
                        )
                        action = agent.sample_as_env_action(
                            torch.cat([prior_flat, last_dist_feat], dim=-1),
                            greedy=False,
                        )
                ctx_push = rearrange(
                    torch.tensor(current_obs, dtype=torch.float32, device="cuda"),
                    "B H W C -> B 1 C H W",
                ) / 255.0
                context_obs.append(ctx_push)
                if replay_buffer.use_proprio:
                    context_proprio.append(
                        torch.tensor(current_proprio, dtype=torch.float32,
                                     device="cuda").unsqueeze(1)
                    )
                context_action.append(action)
            else:
                action = vec_env.action_space.sample().astype(np.float32)

            obs, reward, done, truncated, info = vec_env.step(action)
            next_proprio = np.asarray(info["proprio"], dtype=np.float32)

            term_signal = np.logical_or(done, info["life_loss"])
            replay_buffer.append(
                current_obs, action, reward, term_signal,
                proprio=current_proprio if replay_buffer.use_proprio else None,
            )

            if save_episodes:
                for i in range(num_envs):
                    current_episode[i]["obs"].append(current_obs[i])
                    current_episode[i]["action"].append(action[i])
                    current_episode[i]["reward"].append(np.float32(reward[i]))
                    current_episode[i]["termination"].append(np.float32(term_signal[i]))
                    if replay_buffer.use_proprio:
                        current_episode[i]["proprio"].append(current_proprio[i])

            sum_reward += reward
            episode_steps += 1
            done_flag = np.logical_or(done, truncated)
            if done_flag.any():
                for i in range(num_envs):
                    if done_flag[i]:
                        logger.log("train/episode_reward", float(sum_reward[i]))
                        logger.log("train/episode_steps", int(episode_steps[i]))
                        logger.log("train/current_task_idx", task_idx + 1)
                        logger.log("replay_buffer/length", len(replay_buffer))
                        if er_active:
                            logger.log("replay_buffer/er_length", len(er_buffer))
                        episodes_done += 1
                        sum_reward[i] = 0
                        episode_steps[i] = 0

                        if save_episodes and current_episode[i]["reward"]:
                            ep = {
                                "obs": np.stack(current_episode[i]["obs"], axis=0),
                                "action": np.stack(current_episode[i]["action"], axis=0),
                                "reward": np.asarray(current_episode[i]["reward"], dtype=np.float32),
                                "termination": np.asarray(current_episode[i]["termination"], dtype=np.float32),
                            }
                            if replay_buffer.use_proprio:
                                ep["proprio"] = np.stack(current_episode[i]["proprio"], axis=0)
                            try:
                                replay_buffer.save_episode(episode_dir, ep)
                            except Exception as e:
                                print(f"[seq_er_trainer] save_episode failed: {e}")
                            current_episode[i] = {
                                "obs": [], "action": [], "reward": [],
                                "termination": [], "proprio": [],
                            }
                context_obs.clear()
                context_action.clear()
                context_proprio.clear()

            current_obs = obs
            current_info = info
            current_proprio = next_proprio

            # --------- world model update ---------
            if replay_buffer.ready() and (
                total_steps % max(train_dyn_every // num_envs, 1) == 0
            ):
                main_sample = replay_buffer.sample(batch_size, demo_batch_size, batch_length)
                er_sample = None
                if er_active and er_batch_size > 0 and er_buffer.can_sample(batch_length):
                    er_sample = er_buffer.sample(er_batch_size, batch_length)
                merged = _concat_buffer_samples(
                    main_sample, er_sample, replay_buffer.use_proprio,
                )
                if replay_buffer.use_proprio:
                    s_obs, s_act, s_rew, s_term, s_prop = merged
                    world_model.update(s_obs, s_act, s_rew, s_term,
                                       logger=logger, proprio=s_prop)
                else:
                    s_obs, s_act, s_rew, s_term = merged
                    world_model.update(s_obs, s_act, s_rew, s_term, logger=logger)

            # --------- agent update (on imagined rollouts) ---------
            if replay_buffer.ready() and (
                total_steps % max(train_agent_every // num_envs, 1) == 0
            ):
                log_video = (total_steps % save_every_iters == 0)
                main_sample = replay_buffer.sample(
                    imagine_batch_size, imagine_demo_batch_size, imagine_context_length
                )
                er_sample = None
                er_n = 0
                if (er_active and er_imagine_batch_size > 0
                        and er_buffer.can_sample(imagine_context_length)):
                    er_sample = er_buffer.sample(er_imagine_batch_size, imagine_context_length)
                    er_n = er_imagine_batch_size
                merged = _concat_buffer_samples(
                    main_sample, er_sample, replay_buffer.use_proprio,
                )
                if replay_buffer.use_proprio:
                    s_obs, s_act, s_rew, s_term, s_prop = merged
                else:
                    s_obs, s_act, s_rew, s_term = merged
                    s_prop = None
                with torch.no_grad():
                    world_model.eval()
                    agent.eval()
                    latent, ac_action, rew_hat, term_hat = world_model.imagine_data(
                        agent, s_obs, s_act,
                        imagine_batch_size=(
                            imagine_batch_size + imagine_demo_batch_size + er_n
                        ),
                        imagine_batch_length=imagine_batch_length,
                        log_video=log_video,
                        logger=logger,
                        sample_proprio=s_prop,
                    )
                agent.update(
                    latent=latent, action=ac_action,
                    old_logprob=None, old_value=None,
                    reward=rew_hat, termination=term_hat,
                    logger=logger,
                )

            # --------- cross-task evaluation ---------
            if total_steps > 0 and total_steps % eval_every_iters == 0:
                evaluate_all_tasks_so_far(
                    task_idx=task_idx,
                    tasks=tasks,
                    conf=conf,
                    world_model=world_model,
                    agent=agent,
                    eval_env_cache=eval_env_cache,
                    base_logdir=base_logdir,
                    logger=logger,
                    episodes=eval_episodes,
                    global_env_step=gstep,
                )

            # --------- checkpointing ---------
            if total_steps % save_every_iters == 0 and total_steps > 0:
                save_world_model_split(
                    world_model,
                    task_dir / "rssm.pt",
                    task_dir / "task_heads.pt",
                )
                torch.save(agent.state_dict(), task_dir / "actor_critic.pt")
                manifest = {
                    "iter": int(total_steps),
                    "env_step": int(gstep),
                    "episodes_done": int(episodes_done),
                    "task_idx": int(task_idx),
                    "task_name": task_name,
                }
                (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

                if save_episodes and buffer_max_length is not None:
                    removed = ReplayBuffer.prune_episode_dir_to_cap(
                        episode_dir, int(buffer_max_length),
                    )
                    if removed:
                        print(f"[seq_er_trainer] pruned {removed} old episode "
                              f"files (cap={buffer_max_length})")

            # --------- progress bar ---------
            if total_steps % 20 == 0:
                pbar.set_postfix(
                    env_steps=gstep,
                    episodes=episodes_done,
                    buffer=len(replay_buffer),
                    er=len(er_buffer) if er_active else 0,
                )
            pbar.update(1)

            if hasattr(logger, "commit"):
                logger.commit(step=int(gstep))
    finally:
        pbar.close()
        for e in eval_env_cache.values():
            try:
                e.close()
            except Exception:
                pass
        try:
            vec_env.close()
        except Exception:
            pass

    save_world_model_split(
        world_model,
        task_dir / "rssm.pt",
        task_dir / "task_heads.pt",
    )
    torch.save(agent.state_dict(), task_dir / "actor_critic.pt")
    final_iter = total_iters
    final_gstep = _global_step(final_iter)
    (task_dir / "manifest.json").write_text(json.dumps({
        "iter": int(final_iter),
        "env_step": int(final_gstep),
        "episodes_done": int(episodes_done),
        "task_idx": int(task_idx),
        "task_name": task_name,
        "final": True,
    }, indent=2))

    return final_gstep


# ===========================================================================
#  Main
# ===========================================================================

def main(args, remaining_opts):
    assert torch.cuda.is_available(), "STORM requires CUDA."

    num_tasks = len(args.tasks)
    if len(args.task_steps) != num_tasks:
        raise ValueError("--task-steps must have one value per task.")
    if args.task_buffer_sizes is not None and len(args.task_buffer_sizes) != num_tasks:
        raise ValueError(
            "--task-buffer-sizes must have one value per task (or be omitted)."
        )

    conf = load_config(args.config_path)
    if remaining_opts:
        conf.defrost()
        conf.merge_from_list(remaining_opts)
        conf.freeze()

    base_logdir = Path(args.logdir).expanduser().resolve()
    base_logdir.mkdir(parents=True, exist_ok=True)

    seed_np_torch(seed=args.seed)

    # Resolve effective per-task buffer sizes (used both for replay sizing
    # and for ER budget calculations).
    if args.task_buffer_sizes is not None:
        effective_task_buffer_sizes = [int(s) for s in args.task_buffer_sizes]
    else:
        effective_task_buffer_sizes = [
            int(conf.JointTrainAgent.BufferMaxLength) for _ in range(num_tasks)
        ]

    # ---- Resume detection ---------------------------------------------------
    global_env_step = 0
    resume_from = 0
    prev_rssm_path = None
    progress = load_sequential_progress(base_logdir)
    if progress is not None:
        for i in range(num_tasks):
            entry = progress.get("tasks", {}).get(str(i))
            if entry and entry.get("completed"):
                resume_from = i + 1
                global_env_step = int(entry["global_env_step_at_end"])
                prev_dir = base_logdir / f"task{i+1}_{args.tasks[i]}"
                rssm_file = prev_dir / "rssm.pt"
                if rssm_file.exists():
                    prev_rssm_path = rssm_file
                print(colorama.Fore.CYAN
                      + f">>> RESUME: Task {i+1} ({args.tasks[i]}) already done "
                        f"(global_env_step={global_env_step})"
                      + colorama.Style.RESET_ALL)
            else:
                break

    if resume_from >= num_tasks:
        print(colorama.Fore.GREEN
              + ">>> All tasks already completed. Nothing to do."
              + colorama.Style.RESET_ALL)
        return

    # ---- Wandb init ---------------------------------------------------------
    stored_run_id = (progress or {}).get("wandb_run_id")
    effective_run_id = args.wandb_run_id or stored_run_id

    use_wandb = (not args.no_wandb) and wandb is not None
    active_run_id = None
    if use_wandb:
        run_name = args.wandb_run_name or (
            f"seq_storm_er_{'-'.join(args.tasks)}_s{args.seed}"
        )
        wandb_kwargs = dict(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            dir=str(base_logdir),
            config=dict(
                tasks=args.tasks,
                task_steps=args.task_steps,
                task_buffer_sizes=effective_task_buffer_sizes,
                er_buffer_ratio=args.er_buffer_ratio,
                er_seed=args.er_seed,
                er_batch_size=args.er_batch_size,
                er_imagine_batch_size=args.er_imagine_batch_size,
                seed=args.seed,
                config_path=args.config_path,
                **{f"conf/{k}": str(v) for k, v in conf.items()},
            ),
            tags=["sequential", "storm", "er"] + list(args.tasks),
        )
        if effective_run_id is not None:
            wandb_kwargs["id"] = effective_run_id
            wandb_kwargs["resume"] = "allow"
            print(colorama.Fore.CYAN
                  + f">>> RESUME: attaching to wandb run {effective_run_id}"
                  + colorama.Style.RESET_ALL)
        wandb.init(**wandb_kwargs)
        active_run_id = wandb.run.id if wandb.run is not None else None
    elif wandb is None and not args.no_wandb:
        print(colorama.Fore.YELLOW
              + ">>> wandb not installed, continuing TB-only."
              + colorama.Style.RESET_ALL)

    tb_logger = Logger(path=str(base_logdir))
    logger = WandbTBLogger(tb_logger, use_wandb=use_wandb)

    if active_run_id is not None:
        save_sequential_progress(base_logdir, 0, wandb_run_id=active_run_id)

    # ---- Plan summary -------------------------------------------------------
    print("=" * 64)
    print(colorama.Fore.CYAN + ">>> SEQUENTIAL STORM TRAINING WITH ER"
          + colorama.Style.RESET_ALL)
    print("=" * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = "  <-- resume here" if i == resume_from else ""
        print(f"  Task {i+1}: {t}  env_steps={s}  buffer={effective_task_buffer_sizes[i]}{mark}")
    print(f"  Logdir:               {base_logdir}")
    print(f"  Seed:                 {args.seed}")
    print(f"  ER buffer ratio:      {args.er_buffer_ratio}  "
          f"(ER size per prev task = ratio × that task's buffer)")
    print(f"  ER seed:              {args.er_seed}")
    print(f"  ER batch (WM):        {args.er_batch_size}")
    print(f"  ER batch (imagine):   {args.er_imagine_batch_size}")
    print(f"  Eval every:           {args.eval_every_steps} env steps")
    print(f"  Eval episodes:        {args.eval_episodes}")
    print("=" * 64)

    # ---- Action/proprio discovery via a dummy env --------------------------
    dummy_env = mw_env.build_single_metaworld_env(
        task_name=args.tasks[resume_from],
        image_size=conf.BasicSettings.ImageSize,
        seed=args.seed,
        camera_names=tuple(conf.Env.CameraNames),
        time_limit=conf.Env.TimeLimit,
        reward_range=(conf.Env.RewardMin, conf.Env.RewardMax),
    )
    action_space = dummy_env.action_space
    action_dim = int(np.prod(action_space.shape))
    proprio_dim = dummy_env.proprio_dim
    try:
        dummy_env.close()
    except Exception:
        pass
    print(colorama.Fore.GREEN
          + f"Detected action_dim={action_dim}, proprio_dim={proprio_dim}"
          + colorama.Style.RESET_ALL)

    # ---- Build world model once; keep it across tasks ----------------------
    world_model = build_world_model(conf, action_dim=action_dim, proprio_dim=proprio_dim)

    if hasattr(torch, "compile") and os.environ.get("STORM_COMPILE", "0") == "1":
        print(colorama.Fore.YELLOW + "Compiling storm_transformer..."
              + colorama.Style.RESET_ALL)
        world_model.storm_transformer = torch.compile(world_model.storm_transformer)

    if prev_rssm_path is not None:
        print(colorama.Fore.CYAN
              + f">>> Loading shared RSSM from {prev_rssm_path}"
              + colorama.Style.RESET_ALL)
        load_rssm_into(world_model, prev_rssm_path)

    obs_shape = (conf.BasicSettings.ImageSize,
                 conf.BasicSettings.ImageSize,
                 conf.Models.WorldModel.InChannels)

    # ---- Task loop ---------------------------------------------------------
    for task_idx in range(num_tasks):
        if task_idx < resume_from:
            continue

        task_name = args.tasks[task_idx]
        task_dir = base_logdir / f"task{task_idx+1}_{task_name}"
        task_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 64)
        print(colorama.Fore.CYAN
              + f">>> TASK {task_idx+1}/{num_tasks}: {task_name}"
              + colorama.Style.RESET_ALL)
        print(f"    env_steps: {args.task_steps[task_idx]}")
        print(f"    global env step (start): {global_env_step}")
        print("=" * 64)

        if task_idx > 0 or resume_from > 0:
            print(colorama.Fore.YELLOW
                  + ">>> Reinit reward/termination heads (fresh for this task)"
                  + colorama.Style.RESET_ALL)
            reset_task_heads(world_model, conf)

        agent = build_agent(conf, action_dim=action_dim)

        if task_idx == 0 and args.from_checkpoint is not None and resume_from == 0:
            ck_dir = Path(args.from_checkpoint)
            rssm_ck = ck_dir / "rssm.pt"
            heads_ck = ck_dir / "task_heads.pt"
            ac_ck = ck_dir / "actor_critic.pt"
            if rssm_ck.exists():
                print(colorama.Fore.CYAN
                      + f">>> Loading external rssm.pt from {rssm_ck}"
                      + colorama.Style.RESET_ALL)
                load_rssm_into(world_model, rssm_ck)
            if heads_ck.exists():
                print(colorama.Fore.CYAN
                      + f">>> Loading external task_heads.pt from {heads_ck}"
                      + colorama.Style.RESET_ALL)
                load_task_heads_into(world_model, heads_ck)
            if ac_ck.exists():
                print(colorama.Fore.CYAN
                      + f">>> Loading external actor_critic.pt from {ac_ck}"
                      + colorama.Style.RESET_ALL)
                agent.load_state_dict(torch.load(ac_ck, map_location="cuda"))

        buffer_max_length = effective_task_buffer_sizes[task_idx]

        replay_buffer = ReplayBuffer(
            obs_shape=obs_shape,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=buffer_max_length,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
        )

        # ---- Build the ER buffer for this task (None for task 1) ----
        er_buffer = build_er_buffer_for_task(
            base_logdir=base_logdir,
            task_idx=task_idx,
            tasks=args.tasks,
            task_buffer_sizes=effective_task_buffer_sizes,
            er_buffer_ratio=args.er_buffer_ratio,
            er_seed=args.er_seed,
            obs_shape=obs_shape,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
        )

        # ---- Within-task resume ----
        save_episodes = bool(getattr(
            conf.BasicSettings, "SaveEpisodesToDisk", True,
        ))
        episode_dir = task_dir / "train_eps"
        start_iter = 0
        episodes_done_init = 0
        manifest_path = task_dir / "manifest.json"
        if manifest_path.exists():
            try:
                mf = json.loads(manifest_path.read_text())
                start_iter = int(mf.get("iter", 0))
                episodes_done_init = int(mf.get("episodes_done", 0))
                saved_env_step = int(mf.get("env_step", global_env_step))
                global_env_step = saved_env_step
                print(colorama.Fore.CYAN
                      + f">>> Within-task resume: T{task_idx+1} @iter={start_iter}, "
                        f"global_env_step={global_env_step}, "
                        f"episodes_done={episodes_done_init}"
                      + colorama.Style.RESET_ALL)
                rssm_ck = task_dir / "rssm.pt"
                heads_ck = task_dir / "task_heads.pt"
                ac_ck = task_dir / "actor_critic.pt"
                if rssm_ck.exists():
                    load_rssm_into(world_model, rssm_ck)
                if heads_ck.exists():
                    load_task_heads_into(world_model, heads_ck)
                if ac_ck.exists():
                    agent.load_state_dict(torch.load(ac_ck, map_location="cuda"))
            except Exception as e:
                print(colorama.Fore.YELLOW
                      + f">>> Within-task resume FAILED ({e}); "
                        f"starting this task from scratch."
                      + colorama.Style.RESET_ALL)
                start_iter = 0
                episodes_done_init = 0

        if save_episodes and episode_dir.exists():
            stats = replay_buffer.load_from_directory(str(episode_dir))
            if stats["episodes_restored"] > 0:
                print(colorama.Fore.GREEN
                      + f">>> Reloaded {stats['transitions_restored']} "
                        f"transitions from {stats['episodes_restored']} "
                        f"episodes (buffer length: {len(replay_buffer)})"
                      + colorama.Style.RESET_ALL)
            if stats["kept_ids"]:
                removed = ReplayBuffer.erase_over_episode_files(
                    str(episode_dir), stats["kept_ids"],
                )
                if removed:
                    print(f">>> Pruned {removed} stale episode files "
                          f"(no longer in the bounded ring buffer)")

        save_sequential_progress(
            base_logdir, task_idx,
            task_name=task_name,
            global_env_step_at_start=global_env_step,
        )

        final_gstep = train_one_task(
            task_idx=task_idx,
            num_tasks=num_tasks,
            tasks=args.tasks,
            conf=conf,
            world_model=world_model,
            agent=agent,
            replay_buffer=replay_buffer,
            er_buffer=er_buffer,
            er_batch_size=int(args.er_batch_size),
            er_imagine_batch_size=int(args.er_imagine_batch_size),
            logger=logger,
            base_logdir=base_logdir,
            global_env_step_offset=global_env_step,
            max_env_steps=int(args.task_steps[task_idx]),
            eval_every_env_steps=int(args.eval_every_steps),
            eval_episodes=int(args.eval_episodes),
            save_every_env_steps=int(conf.JointTrainAgent.SaveEverySteps),
            start_iter=start_iter,
            episodes_done_init=episodes_done_init,
            episode_dir=(str(episode_dir) if save_episodes else None),
            buffer_max_length=buffer_max_length,
        )

        global_env_step = final_gstep
        save_sequential_progress(
            base_logdir, task_idx,
            task_name=task_name,
            completed=True,
            global_env_step_at_end=global_env_step,
        )

        del replay_buffer, agent, er_buffer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(colorama.Fore.GREEN
              + f">>> TASK {task_idx+1} ({task_name}) done @g={global_env_step}"
              + colorama.Style.RESET_ALL)
        print()

    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass

    print("=" * 64)
    print(colorama.Fore.GREEN + ">>> ALL TASKS COMPLETED."
          + colorama.Style.RESET_ALL)
    print(f">>> Final global env step: {global_env_step}")
    print("=" * 64)


# ===========================================================================
#  Entry point
# ===========================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    p = argparse.ArgumentParser(
        description="Sequential STORM training with Experience Replay",
    )
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--task-steps", nargs="+", type=int, required=True)
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--config_path", type=str,
                   default="config_files/STORM_metaworld.yaml")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    p.add_argument("--wandb-project", type=str,
                   default="Metaworld_STORM_Sequential_ER")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-run-id", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")

    p.add_argument("--from-checkpoint", type=str, default=None)
    p.add_argument("--skip-pretrain", action="store_true",
                   help="(reserved; STORM has no explicit pretrain phase)")

    p.add_argument("--eval-every-steps", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=3)

    p.add_argument(
        "--task-buffer-sizes", nargs="+", type=int, default=None,
        help="Optional per-task BufferMaxLength override (one value per "
             "task). Falls back to conf.JointTrainAgent.BufferMaxLength.",
    )

    # ER hyperparameters
    p.add_argument(
        "--er-buffer-ratio", type=float, default=0.05,
        help="Per-prev-task ER budget = ratio × that task's BufferMaxLength. "
             "0 disables ER (default 0.05).",
    )
    p.add_argument(
        "--er-seed", type=int, default=42,
        help="Seed for reservoir sampling of ER episodes.",
    )
    p.add_argument(
        "--er-batch-size", type=int, default=4,
        help="Number of ER trajectories mixed into each WM update batch "
             "(added to BatchSize).",
    )
    p.add_argument(
        "--er-imagine-batch-size", type=int, default=256,
        help="Number of ER context windows mixed into each agent imagine "
             "batch (added to ImagineBatchSize).",
    )

    main_args, remaining = p.parse_known_args()
    main(main_args, remaining)
