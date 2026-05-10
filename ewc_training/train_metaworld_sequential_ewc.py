"""Sequential STORM training with EWC (Elastic Weight Consolidation) and
cross-task evaluation.

Mirrors ``third_party/dreamerv3/ewc_training/dreamer_sequential_ewc.py``
but for STORM, building on ``third_party/STORM/train_metaworld_sequential.py``.

What EWC adds on top of the naive sequential trainer:
  * After each task completes, compute a diagonal Fisher Information Matrix
    over the shared-core parameters (everything except the per-task reward
    and termination decoders). Snapshot the current shared-core weights.
  * On every world-model update of subsequent tasks, add a quadratic
    penalty ``λ · Σ_t Σ_i F_{t,i} · (θ_i − θ*_{t,i})²`` to the loss before
    backward, anchoring the shared core toward weights that were
    "important" for previous tasks.
  * The EWC manager is persisted at each task boundary
    (``ewc_state_task{N}.pt``) for resume support.

Why this matters for STORM specifically: STORM's ``WorldModel.update``
runs forward+backward+optimizer-step in one call. To inject EWC cleanly,
we shadow that method with :func:`world_model_update_with_ewc`, which
mirrors the body but adds ``ewc_manager.penalty(world_model)`` to
``total_loss`` before scaling+backward. The shared-core Adam moments
are preserved across task boundaries via the same surgical splice that
``train_metaworld_sequential.py`` uses (``reset_task_heads``).

Split-checkpoint layout per task (with EWC additions):
    <logdir>/task{N}_<name>/
        rssm.pt           — shared core (no reward/termination)
        task_heads.pt      — reward_decoder + termination_decoder
        actor_critic.pt    — ActorCriticAgent state dict
        manifest.json      — {iter, env_step, episodes_done, wandb_run_id}
        ewc_state_task{N}.pt — Fisher + param snapshots up to task N

Example:
    python train_metaworld_sequential_ewc.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --task-buffer-sizes 25000 62500 \\
        --ewc-lambda 5000.0 \\
        --ewc-fisher-batches 50 \\
        --logdir ./runs/seq_storm_ewc \\
        --config_path config_files/STORM_metaworld.yaml \\
        --wandb-entity haoyu-a2i \\
        --wandb-project Metaworld_STORM_Sequential_EWC
"""
import argparse
import gc
import json
import os
import sys
import time as _time
import warnings
from collections import deque
from pathlib import Path

import colorama
import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm

_HERE = Path(__file__).resolve().parent
_STORM_ROOT = _HERE.parent
sys.path.insert(0, str(_STORM_ROOT))
sys.path.insert(0, str(_HERE))

import agents
import envs.metaworld_env as mw_env
from replay_buffer import ReplayBuffer
from sub_models.world_models import (
    WorldModel,
    RewardDecoder,
    TerminationDecoder,
)
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length
from utils import Logger, load_config, seed_np_torch

from ewc import EWCManager  # local module

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# ===========================================================================
#  Logger wrappers (verbatim from train_metaworld_sequential.py)
# ===========================================================================

class WandbTBLogger:
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
    """Surgical optimizer splice — preserve shared-core Adam moments."""
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
#  EWC-augmented WorldModel update
# ===========================================================================

def world_model_update_with_ewc(
    world_model, obs, action, reward, termination,
    ewc_manager: EWCManager,
    logger=None, proprio=None,
):
    """Mirror of ``WorldModel.update`` with the EWC penalty added inline.

    Why a parallel implementation rather than monkeypatching: STORM's
    ``update`` packs forward + GradScaler.backward + optimizer.step into
    one method. To get a single combined gradient (task loss + EWC penalty
    → one optimizer step), the cleanest path is to recompute forward here
    and add the penalty before backward. An alternative — calling
    ``world_model.update`` and then doing a separate EWC backward+step —
    would issue *two* Adam updates per iteration on shared-core params,
    effectively doubling the learning rate on those params for the EWC
    component, which is not what the EWC objective asks for.

    Returns nothing; mirrors ``world_model.update`` exactly otherwise.
    """
    world_model.train()
    batch_size, batch_length = obs.shape[:2]
    use_amp = world_model.use_amp

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        embedding = world_model._encode_combined(obs, proprio)
        post_logits = world_model.dist_head.forward_post(embedding)
        z = world_model.stright_throught_gradient(post_logits, "random_sample")
        flat_z = world_model.flatten_sample(z)

        obs_hat = world_model.image_decoder(flat_z)
        if world_model.use_proprio:
            proprio_hat = world_model.proprio_decoder(flat_z)

        mask = get_subsequent_mask_with_batch_length(batch_length, flat_z.device)
        dist_feat = world_model.storm_transformer(flat_z, action, mask)
        prior_logits = world_model.dist_head.forward_prior(dist_feat)
        reward_hat = world_model.reward_decoder(dist_feat)
        term_hat = world_model.termination_decoder(dist_feat)

        recon_loss = world_model.mse_loss_func(obs_hat, obs)
        if world_model.use_proprio:
            proprio_recon_loss = ((proprio_hat - proprio) ** 2).sum(dim=-1).mean()
        else:
            proprio_recon_loss = torch.tensor(0.0, device=obs.device)
        reward_loss = world_model.symlog_twohot_loss_func(reward_hat, reward)
        termination_loss = world_model.bce_with_logits_loss_func(term_hat, termination)
        dyn_loss, dyn_real_kl = world_model.categorical_kl_div_loss(
            post_logits[:, 1:].detach(), prior_logits[:, :-1],
        )
        rep_loss, rep_real_kl = world_model.categorical_kl_div_loss(
            post_logits[:, 1:], prior_logits[:, :-1].detach(),
        )
        task_loss = (
            recon_loss + proprio_recon_loss + reward_loss + termination_loss
            + 0.5 * dyn_loss + 0.1 * rep_loss
        )

    # Compute EWC penalty in fp32 OUTSIDE autocast — penalty is a sum of
    # quadratic terms over fp32 Fisher diagonals, no benefit from bf16.
    if ewc_manager is not None and ewc_manager.num_tasks_consolidated > 0:
        ewc_penalty = ewc_manager.penalty(world_model)
    else:
        ewc_penalty = torch.tensor(0.0, device=obs.device, dtype=torch.float32)

    total_loss = task_loss + ewc_penalty

    if not torch.isfinite(total_loss):
        stats = {
            "recon": recon_loss.item(), "reward": reward_loss.item(),
            "term": termination_loss.item(), "dyn": dyn_loss.item(),
            "rep": rep_loss.item(),
            "ewc_penalty": float(ewc_penalty.detach().item()),
        }
        raise RuntimeError(f"Non-finite world-model loss: {stats}")

    world_model.scaler.scale(total_loss).backward()
    world_model.scaler.unscale_(world_model.optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        world_model.parameters(), max_norm=100.0,
    )
    if torch.isfinite(grad_norm):
        world_model.scaler.step(world_model.optimizer)
    elif logger is not None:
        logger.log("WorldModel/skipped_step", 1.0)
    world_model.scaler.update()
    world_model.optimizer.zero_grad(set_to_none=True)

    if logger is not None:
        logger.log("WorldModel/reconstruction_loss", recon_loss.item())
        if world_model.use_proprio:
            logger.log("WorldModel/proprio_recon_loss", proprio_recon_loss.item())
        logger.log("WorldModel/reward_loss", reward_loss.item())
        logger.log("WorldModel/termination_loss", termination_loss.item())
        logger.log("WorldModel/dynamics_loss", dyn_loss.item())
        logger.log("WorldModel/dynamics_real_kl_div", dyn_real_kl.item())
        logger.log("WorldModel/representation_loss", rep_loss.item())
        logger.log("WorldModel/representation_real_kl_div", rep_real_kl.item())
        logger.log("WorldModel/task_loss", task_loss.item())
        logger.log("WorldModel/ewc_penalty", float(ewc_penalty.detach().item()))
        logger.log("WorldModel/total_loss", total_loss.item())
        logger.log("WorldModel/grad_norm", grad_norm.item())
        logger.log("WorldModel/reward_batch_absmax", reward.abs().max().item())


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
#  Cross-task evaluation
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
#  Per-task training loop with EWC penalty
# ===========================================================================

def train_one_task(
    task_idx, num_tasks, tasks, conf,
    world_model, agent, replay_buffer,
    ewc_manager,
    logger, base_logdir,
    global_env_step_offset, max_env_steps,
    eval_every_env_steps, eval_episodes,
    save_every_env_steps,
    start_iter=0, episodes_done_init=0,
    episode_dir=None,
    buffer_max_length=None,
):
    """STORM joint training loop with EWC penalty in the WM update.

    Identical to the naive sequential trainer except for the world-model
    update path: we call :func:`world_model_update_with_ewc` instead of
    ``world_model.update`` so the EWC penalty is added to the loss before
    backward+step. Agent (actor-critic) updates are unchanged — EWC only
    constrains the shared core; the per-task actor-critic is fresh anyway.
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

    ewc_active = (
        ewc_manager is not None and ewc_manager.num_tasks_consolidated > 0
    )
    if ewc_active:
        print(colorama.Fore.CYAN
              + f">>> EWC active: {ewc_manager.num_tasks_consolidated} "
                f"consolidated task(s), λ={ewc_manager.lambda_ewc}"
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
                                print(f"[seq_ewc_trainer] save_episode failed: {e}")
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

            # --------- world model update (with EWC) ---------
            if replay_buffer.ready() and (
                total_steps % max(train_dyn_every // num_envs, 1) == 0
            ):
                sample = replay_buffer.sample(batch_size, demo_batch_size, batch_length)
                if replay_buffer.use_proprio:
                    s_obs, s_act, s_rew, s_term, s_prop = sample
                    world_model_update_with_ewc(
                        world_model, s_obs, s_act, s_rew, s_term,
                        ewc_manager=ewc_manager, logger=logger, proprio=s_prop,
                    )
                else:
                    s_obs, s_act, s_rew, s_term = sample
                    world_model_update_with_ewc(
                        world_model, s_obs, s_act, s_rew, s_term,
                        ewc_manager=ewc_manager, logger=logger,
                    )

            # --------- agent update (on imagined rollouts) ---------
            if replay_buffer.ready() and (
                total_steps % max(train_agent_every // num_envs, 1) == 0
            ):
                log_video = (total_steps % save_every_iters == 0)
                sample = replay_buffer.sample(
                    imagine_batch_size, imagine_demo_batch_size, imagine_context_length
                )
                if replay_buffer.use_proprio:
                    s_obs, s_act, s_rew, s_term, s_prop = sample
                else:
                    s_obs, s_act, s_rew, s_term = sample
                    s_prop = None
                with torch.no_grad():
                    world_model.eval()
                    agent.eval()
                    latent, ac_action, rew_hat, term_hat = world_model.imagine_data(
                        agent, s_obs, s_act,
                        imagine_batch_size=imagine_batch_size + imagine_demo_batch_size,
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
                        print(f"[seq_ewc_trainer] pruned {removed} old episode "
                              f"files (cap={buffer_max_length})")

            if total_steps % 20 == 0:
                pbar.set_postfix(
                    env_steps=gstep,
                    episodes=episodes_done,
                    buffer=len(replay_buffer),
                    ewc=ewc_manager.num_tasks_consolidated if ewc_manager else 0,
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

    if args.task_buffer_sizes is not None:
        effective_task_buffer_sizes = [int(s) for s in args.task_buffer_sizes]
    else:
        effective_task_buffer_sizes = [
            int(conf.JointTrainAgent.BufferMaxLength) for _ in range(num_tasks)
        ]

    # ---- EWC manager (created up-front so resume can populate it) ----------
    ewc_manager = EWCManager(
        lambda_ewc=args.ewc_lambda,
        task_head_prefixes=TASK_HEAD_PREFIXES,
    )

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

    # ---- Resume EWC: load the most-recent ewc_state_task{N}.pt -------------
    if resume_from > 0:
        for j in range(resume_from - 1, -1, -1):
            ewc_file = base_logdir / f"task{j+1}_{args.tasks[j]}" / f"ewc_state_task{j+1}.pt"
            if ewc_file.exists():
                print(colorama.Fore.CYAN
                      + f">>> EWC: Loading state from {ewc_file}"
                      + colorama.Style.RESET_ALL)
                ewc_manager.load_state_dict(
                    torch.load(ewc_file, map_location="cpu"),
                )
                print(f">>> EWC: Restored {ewc_manager.num_tasks_consolidated} "
                      f"consolidated task(s)")
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
            f"seq_storm_ewc_{'-'.join(args.tasks)}_s{args.seed}"
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
                ewc_lambda=args.ewc_lambda,
                ewc_fisher_batches=args.ewc_fisher_batches,
                seed=args.seed,
                config_path=args.config_path,
                **{f"conf/{k}": str(v) for k, v in conf.items()},
            ),
            tags=["sequential", "storm", "ewc"] + list(args.tasks),
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
    print(colorama.Fore.CYAN + ">>> SEQUENTIAL STORM TRAINING WITH EWC"
          + colorama.Style.RESET_ALL)
    print("=" * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = "  <-- resume here" if i == resume_from else ""
        print(f"  Task {i+1}: {t}  env_steps={s}  buffer={effective_task_buffer_sizes[i]}{mark}")
    print(f"  Logdir:                {base_logdir}")
    print(f"  Seed:                  {args.seed}")
    print(f"  EWC λ:                 {args.ewc_lambda}")
    print(f"  EWC Fisher batches:    {args.ewc_fisher_batches}")
    print(f"  Eval every:            {args.eval_every_steps} env steps")
    print(f"  Eval episodes:         {args.eval_episodes}")
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

    # If we resumed EWC state, move the cached Fisher/param tensors onto
    # the live device and rebuild the penalty cache against the freshly
    # built world_model. Mirrors the dreamer ewc_training resume path.
    if ewc_manager.num_tasks_consolidated > 0:
        device = "cuda"
        for tid, reg in ewc_manager.regularization_terms.items():
            for k in reg["importance"]:
                reg["importance"][k] = reg["importance"][k].to(device)
                reg["task_param"][k] = reg["task_param"][k].to(device)
        ewc_manager._rebuild_penalty_cache(world_model)
        print(colorama.Fore.CYAN
              + f">>> EWC: Rebuilt penalty cache on {device} "
                f"({ewc_manager.num_tasks_consolidated} task(s))"
              + colorama.Style.RESET_ALL)

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
        print(f"    env_steps:               {args.task_steps[task_idx]}")
        print(f"    global env step (start): {global_env_step}")
        print(f"    EWC consolidated tasks:  {ewc_manager.num_tasks_consolidated}")
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
            obs_shape=(conf.BasicSettings.ImageSize,
                       conf.BasicSettings.ImageSize,
                       conf.Models.WorldModel.InChannels),
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=buffer_max_length,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
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
            ewc_manager=ewc_manager,
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

        # =================================================================
        # [EWC] Compute Fisher and consolidate (skip after the final task)
        # =================================================================
        if task_idx < num_tasks - 1:
            print(colorama.Fore.CYAN
                  + f">>> EWC: computing Fisher for task {task_idx+1} "
                    f"({task_name})..."
                  + colorama.Style.RESET_ALL)
            t0 = _time.time()

            importance, task_param = ewc_manager.compute_fisher(
                world_model=world_model,
                replay_buffer=replay_buffer,
                batch_size=conf.JointTrainAgent.BatchSize,
                batch_length=conf.JointTrainAgent.BatchLength,
                num_batches=int(args.ewc_fisher_batches),
                device="cuda",
            )
            ewc_manager.consolidate(world_model, importance, task_param, task_idx)
            elapsed = _time.time() - t0

            n_params = sum(v.numel() for v in importance.values())
            fisher_mean = sum(v.mean().item() for v in importance.values()) / max(len(importance), 1)
            fisher_max = max(v.max().item() for v in importance.values())
            print(colorama.Fore.GREEN
                  + f">>> EWC: consolidated task {task_idx+1} in {elapsed:.1f}s: "
                    f"{len(importance)} groups, {n_params:,} params, "
                    f"mean F={fisher_mean:.2e}, max F={fisher_max:.2e}"
                  + colorama.Style.RESET_ALL)

            logger.log("ewc/fisher_mean", fisher_mean)
            logger.log("ewc/fisher_max", fisher_max)
            logger.log("ewc/fisher_compute_time_s", elapsed)
            logger.log("ewc/num_tasks_consolidated", float(ewc_manager.num_tasks_consolidated))
            if hasattr(logger, "commit"):
                logger.commit(step=int(global_env_step))

            ewc_state_path = task_dir / f"ewc_state_task{task_idx+1}.pt"
            torch.save(ewc_manager.state_dict(), ewc_state_path)
            print(f">>> EWC: saved -> {ewc_state_path.name}")

        save_sequential_progress(
            base_logdir, task_idx,
            task_name=task_name,
            completed=True,
            global_env_step_at_end=global_env_step,
        )

        del replay_buffer, agent
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
        description="Sequential STORM training with EWC",
    )
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--task-steps", nargs="+", type=int, required=True)
    p.add_argument("--logdir", type=str, required=True)
    p.add_argument("--config_path", type=str,
                   default="config_files/STORM_metaworld.yaml")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    p.add_argument("--wandb-project", type=str,
                   default="Metaworld_STORM_Sequential_EWC")
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

    # EWC hyperparameters
    p.add_argument(
        "--ewc-lambda", type=float, default=5000.0,
        help="EWC regularization strength (default: 5000.0). "
             "Sweep [500, 1000, 5000, 10000, 50000] to calibrate.",
    )
    p.add_argument(
        "--ewc-fisher-batches", type=int, default=50,
        help="Number of mini-batches for Fisher estimation at each task "
             "boundary (default: 50, ~800 sequence samples at "
             "BatchSize=16, BatchLength=64).",
    )

    main_args, remaining = p.parse_known_args()
    main(main_args, remaining)
