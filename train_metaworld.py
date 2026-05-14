"""STORM training entrypoint for Meta-World (single task, continuous action).

Mirrors train.py but:
  * builds the Meta-World env via envs.metaworld_env (dreamer-style pipeline)
  * uses continuous actions end-to-end (no one-hot)
  * feeds proprio through the world model encoder/decoder
  * logs to Weights & Biases (and still writes TensorBoard via Logger)
  * prints progress in the same style as the dreamer sequential trainer

Example:
    python train_metaworld.py \
        -n mw_single_drawer_1201 \
        -seed 0 \
        -config_path config_files/STORM_metaworld.yaml \
        -env_name metaworld_drawer-open-v3 \
        -wandb_entity haoyu-a2i \
        -wandb_project Metaworld_STORM_Single
"""
import argparse
import json
import os
import re
import shutil
import warnings
from collections import deque
from pathlib import Path

import colorama
import numpy as np
import torch
from einops import rearrange
from tqdm.auto import tqdm

import agents
import envs.metaworld_env as mw_env
from replay_buffer import ReplayBuffer
from sub_models.world_models import WorldModel
from utils import Logger, load_config, seed_np_torch

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# ---------------------------------------------------------------------------
#  wandb-aware logger wrapper
# ---------------------------------------------------------------------------
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
            return  # skip media -- TB only
        try:
            wandb.log({tag: float(value)}, commit=False)
        except Exception:
            pass

    def commit(self, step=None):
        if self._use_wandb:
            try:
                wandb.log({}, commit=True, step=step) if step is not None else wandb.log({}, commit=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
#  Resume / checkpoint helpers
# ---------------------------------------------------------------------------
_CKPT_STEP_RE = re.compile(r"world_model_(\d+)\.pth$")


def _latest_ckpt_step(ckpt_dir):
    """Return the largest ``<step>`` for which both world_model and agent
    checkpoints exist under ``ckpt_dir``, or ``None`` if none found."""
    if not os.path.isdir(ckpt_dir):
        return None
    steps = []
    for fname in os.listdir(ckpt_dir):
        m = _CKPT_STEP_RE.match(fname)
        if not m:
            continue
        step = int(m.group(1))
        if os.path.exists(os.path.join(ckpt_dir, f"agent_{step}.pth")):
            steps.append(step)
    return max(steps) if steps else None


def _prune_old_ckpts(ckpt_dir, keep_last=2):
    """Keep only the ``keep_last`` most-recent world_model/agent pairs."""
    if not os.path.isdir(ckpt_dir):
        return
    steps = []
    for fname in os.listdir(ckpt_dir):
        m = _CKPT_STEP_RE.match(fname)
        if m:
            steps.append(int(m.group(1)))
    steps = sorted(set(steps))
    for step in steps[:-keep_last]:
        for name in (f"world_model_{step}.pth", f"agent_{step}.pth"):
            path = os.path.join(ckpt_dir, name)
            try:
                os.remove(path)
            except OSError:
                pass


def _save_manifest(path, manifest):
    """Atomically write ``manifest`` (a JSON-serializable dict) to ``path``."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, path)


def _load_manifest(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
#  Model builders
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
#  Training step helpers
# ---------------------------------------------------------------------------
def train_world_model_step(replay_buffer, world_model, batch_size, demo_batch_size, batch_length, logger):
    sample = replay_buffer.sample(batch_size, demo_batch_size, batch_length)
    if replay_buffer.use_proprio:
        obs, action, reward, termination, proprio = sample
        world_model.update(obs, action, reward, termination, logger=logger, proprio=proprio)
    else:
        obs, action, reward, termination = sample
        world_model.update(obs, action, reward, termination, logger=logger)


def world_model_imagine_data_grad(replay_buffer, world_model, agent,
                                  imagine_batch_size, imagine_demo_batch_size,
                                  imagine_context_length, imagine_batch_length):
    """Pathwise (dreamer-style) imagination — gradients PRESERVED.

    Used together with ActorCriticAgent.update_pathwise: imagined reward
    / termination are differentiable in actor params via the rollout's
    reparameterized actions, so the actor loss is `-mean(lambda_return)`
    rather than REINFORCE's `-(log_prob × advantage)`. See the same
    block in train_metaworld_sequential.py for the continual-learning
    motivation; for single-task training this gives lower-variance
    actor gradients (Heess et al. 2015 / DreamerV3).
    """
    world_model.eval()
    agent.train()
    sample = replay_buffer.sample(imagine_batch_size, imagine_demo_batch_size, imagine_context_length)
    if replay_buffer.use_proprio:
        sample_obs, sample_action, sample_reward, sample_termination, sample_proprio = sample
    else:
        sample_obs, sample_action, sample_reward, sample_termination = sample
        sample_proprio = None

    latent, action, reward_hat, termination_hat = world_model.imagine_data_grad(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size + imagine_demo_batch_size,
        imagine_batch_length=imagine_batch_length,
        sample_proprio=sample_proprio,
    )
    return latent, action, reward_hat, termination_hat


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------
def joint_train_world_model_agent(
    env_name, conf, replay_buffer, world_model, agent, logger, run_name,
    start_iter=0, episodes_done_init=0,
    episode_dir=None, save_episodes=True,
    wandb_run_id=None,
    ckpt_dir=None, manifest_path=None,
):
    # Backward compat: derive paths from run_name when caller doesn't pass them.
    if ckpt_dir is None:
        ckpt_dir = Path("ckpt") / run_name
    else:
        ckpt_dir = Path(ckpt_dir)
    if manifest_path is None:
        manifest_path = Path("runs") / run_name / "manifest.json"
    else:
        manifest_path = Path(manifest_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if save_episodes and episode_dir is not None:
        os.makedirs(episode_dir, exist_ok=True)

    num_envs = conf.JointTrainAgent.NumEnvs
    max_steps = conf.JointTrainAgent.SampleMaxSteps
    image_size = conf.BasicSettings.ImageSize
    train_dynamics_every = conf.JointTrainAgent.TrainDynamicsEverySteps
    train_agent_every = conf.JointTrainAgent.TrainAgentEverySteps
    batch_size = conf.JointTrainAgent.BatchSize
    demo_batch_size = conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0
    batch_length = conf.JointTrainAgent.BatchLength
    imagine_batch_size = conf.JointTrainAgent.ImagineBatchSize
    imagine_demo_batch_size = conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0
    imagine_context_length = conf.JointTrainAgent.ImagineContextLength
    imagine_batch_length = conf.JointTrainAgent.ImagineBatchLength
    save_every = conf.JointTrainAgent.SaveEverySteps
    seed = conf.BasicSettings.Seed

    vec_env = mw_env.build_metaworld_vec_env(
        task_name=env_name,
        image_size=image_size,
        num_envs=num_envs,
        seed=seed,
        camera_names=tuple(conf.Env.CameraNames),
        time_limit=conf.Env.TimeLimit,
        reward_range=(conf.Env.RewardMin, conf.Env.RewardMax),
    )
    print(colorama.Fore.YELLOW + f"Current env: {env_name}" + colorama.Style.RESET_ALL)

    sum_reward = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs, dtype=np.int64)
    episodes_done = episodes_done_init

    current_obs, current_info = vec_env.reset()
    current_proprio = np.asarray(current_info["proprio"], dtype=np.float32)

    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    context_proprio = deque(maxlen=16)

    # Per-env running episode accumulator for on-disk persistence.
    # Keys match ReplayBuffer.save_episode's expected schema.
    current_episode = [
        {"obs": [], "action": [], "reward": [], "termination": [], "proprio": []}
        for _ in range(num_envs)
    ]

    total_iters = max_steps // num_envs
    pbar = tqdm(total=total_iters, initial=start_iter,
                desc=">>> STORM training", unit="step")

    for total_steps in range(start_iter, total_iters):
        # -------- act --------
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample().astype(np.float32)
                else:
                    ctx_obs = torch.cat(list(context_obs), dim=1)          # (N, T, C, H, W)
                    ctx_proprio = None
                    if replay_buffer.use_proprio:
                        ctx_proprio = torch.cat(list(context_proprio), dim=1)  # (N, T, D)
                    context_latent = world_model.encode_obs(ctx_obs, ctx_proprio)
                    model_context_action = np.stack(list(context_action), axis=1)  # (N, T, A)
                    model_context_action = torch.tensor(model_context_action, dtype=torch.float32, device="cuda")
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(
                        context_latent, model_context_action
                    )
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False,
                    )
            # Push most-recent obs/proprio into the rollout context so next
            # step encodes with the freshest posterior.
            ctx_obs_push = rearrange(torch.tensor(current_obs, dtype=torch.float32, device="cuda"),
                                     "B H W C -> B 1 C H W") / 255.0
            context_obs.append(ctx_obs_push)
            if replay_buffer.use_proprio:
                context_proprio.append(
                    torch.tensor(current_proprio, dtype=torch.float32, device="cuda").unsqueeze(1)
                )
            context_action.append(action)
        else:
            action = vec_env.action_space.sample().astype(np.float32)

        obs, reward, done, truncated, info = vec_env.step(action)
        next_proprio = np.asarray(info["proprio"], dtype=np.float32)

        # Termination signal fed to the world model: terminated OR life_loss (Atari).
        # For metaworld, life_loss is always False so we just use `done`.
        term_signal = np.logical_or(done, info["life_loss"])
        replay_buffer.append(
            current_obs, action, reward, term_signal,
            proprio=current_proprio if replay_buffer.use_proprio else None,
        )

        # Mirror the append into each env's episode accumulator so we can
        # flush the full episode to disk on `done` / `truncated`. Slice per-env
        # to match ReplayBuffer's (N, ...) conventions.
        if save_episodes and episode_dir is not None:
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
                    logger.log(f"sample/{env_name}_reward", float(sum_reward[i]))
                    logger.log(f"sample/{env_name}_episode_steps", int(episode_steps[i]))
                    logger.log("replay_buffer/length", len(replay_buffer))
                    episodes_done += 1
                    sum_reward[i] = 0
                    episode_steps[i] = 0

                    if save_episodes and episode_dir is not None and current_episode[i]["reward"]:
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
                            print(f"[train_metaworld] save_episode failed: {e}")
                        current_episode[i] = {
                            "obs": [], "action": [], "reward": [],
                            "termination": [], "proprio": [],
                        }
            # Reset context on episode boundary.
            context_obs.clear()
            context_action.clear()
            context_proprio.clear()

        current_obs = obs
        current_info = info
        current_proprio = next_proprio

        # -------- world model update --------
        if replay_buffer.ready() and total_steps % (max(train_dynamics_every // num_envs, 1)) == 0:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demo_batch_size=demo_batch_size,
                batch_length=batch_length,
                logger=logger,
            )

        # -------- agent update (pathwise / dreamer-style) --------
        # Imagine WITH gradients so actor params receive a pathwise
        # gradient through reward / termination. Replaces the REINFORCE
        # path (agent.update on detached imagination) for consistency
        # with train_metaworld_sequential.py and to give the actor a
        # lower-variance, self-regulating training signal.
        if replay_buffer.ready() and total_steps % (max(train_agent_every // num_envs, 1)) == 0:
            world_model.requires_grad_(False)
            try:
                imagine_latent, agent_action, imagine_reward, imagine_termination = world_model_imagine_data_grad(
                    replay_buffer=replay_buffer,
                    world_model=world_model,
                    agent=agent,
                    imagine_batch_size=imagine_batch_size,
                    imagine_demo_batch_size=imagine_demo_batch_size,
                    imagine_context_length=imagine_context_length,
                    imagine_batch_length=imagine_batch_length,
                )
                agent.update_pathwise(
                    latent=imagine_latent,
                    action=agent_action,
                    reward=imagine_reward,
                    termination=imagine_termination,
                    logger=logger,
                )
            finally:
                world_model.requires_grad_(True)

        # -------- checkpointing --------
        if total_steps % (max(save_every // num_envs, 1)) == 0:
            torch.save(world_model.state_dict(), ckpt_dir / f"world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), ckpt_dir / f"agent_{total_steps}.pth")

            # Resume manifest sits alongside TB logs in run_dir.
            manifest = {
                "iter": int(total_steps),
                "env_step": int(total_steps * num_envs),
                "episodes_done": int(episodes_done),
                "wandb_run_id": wandb_run_id,
            }
            _save_manifest(manifest_path, manifest)

            # Bound disk: keep only the 2 most-recent (world_model, agent) pairs.
            _prune_old_ckpts(str(ckpt_dir), keep_last=2)

            # FIFO-prune on-disk episodes so total transitions stay ≤ BufferMaxLength.
            # Matches the in-memory ring-buffer cap, so disk footprint tracks RAM.
            if save_episodes and episode_dir is not None:
                removed = replay_buffer.prune_episode_dir_to_cap(
                    episode_dir, conf.JointTrainAgent.BufferMaxLength,
                )
                if removed:
                    print(f"[train_metaworld] pruned {removed} old episode files "
                          f"(cap={conf.JointTrainAgent.BufferMaxLength})")

        # -------- progress bar --------
        if total_steps % 20 == 0:
            pbar.set_postfix(
                env_steps=total_steps * num_envs,
                episodes=episodes_done,
                buffer=len(replay_buffer),
            )
        pbar.update(1)

        if hasattr(logger, "commit"):
            logger.commit()

    pbar.close()


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True, help="Run name")
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True,
                        help="e.g. metaworld_drawer-open-v3")
    parser.add_argument("-wandb_entity", type=str, default=None)
    parser.add_argument("-wandb_project", type=str, default="Metaworld_STORM_Single")
    parser.add_argument("-wandb_run_name", type=str, default=None)
    parser.add_argument("-wandb_run_id", type=str, default=None,
                        help="Pin the wandb run id (used by sweep launchers for deterministic resume).")
    parser.add_argument("-wandb_group", type=str, default=None,
                        help="WandB group; multiple seeds of one taskset share a group.")
    parser.add_argument("-wandb_tags", nargs="*", default=None,
                        help="WandB tags (space-separated).")
    parser.add_argument("-logdir", type=str, default=None,
                        help="If set, write runs/ + ckpt/ directly under this directory "
                             "instead of the cwd-relative runs/<n> and ckpt/<n>. The sweep "
                             "harness uses this to give each run its own folder.")
    parser.add_argument("-no_wandb", action="store_true")
    args, opts = parser.parse_known_args()

    conf = load_config(args.config_path)
    if opts:
        # Allow yacs-style "Key.SubKey value" overrides from the CLI / bash runner.
        conf.defrost()
        conf.merge_from_list(opts)
        conf.freeze()
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    if opts:
        print(colorama.Fore.MAGENTA + f"Config overrides: {opts}" + colorama.Style.RESET_ALL)
    print(colorama.Fore.CYAN + str(conf) + colorama.Style.RESET_ALL)

    seed_np_torch(seed=args.seed)

    if args.logdir:
        # Sweep mode: collapse runs/ckpt under one user-specified folder.
        logdir_root = Path(args.logdir)
        run_dir = logdir_root / "runs"
        ckpt_dir = logdir_root / "ckpt"
    else:
        # Backward compat with hand-launched training/*.sh scripts.
        run_dir = Path("runs") / args.n
        ckpt_dir = Path("ckpt") / args.n
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = Logger(path=str(run_dir))
    shutil.copy(args.config_path, run_dir / "config.yaml")

    # -------- auto-resume discovery --------
    # A resume is possible iff BOTH a model checkpoint AND a manifest exist.
    # (Either alone is evidence of a partial/corrupt state we don't trust.)
    manifest_path = run_dir / "manifest.json"
    episode_dir = run_dir / "train_eps"
    resume_step = _latest_ckpt_step(str(ckpt_dir))
    resume_manifest = _load_manifest(manifest_path)
    is_resume = resume_step is not None and resume_manifest is not None
    prior_wandb_run_id = resume_manifest.get("wandb_run_id") if resume_manifest else None
    if is_resume:
        print(colorama.Fore.GREEN
              + f">>> AUTO-RESUME: found ckpt step {resume_step} and manifest "
                f"(env_step={resume_manifest.get('env_step')}, "
                f"episodes={resume_manifest.get('episodes_done')}, "
                f"wandb_id={prior_wandb_run_id})"
              + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.YELLOW
              + f">>> AUTO-RESUME: starting fresh (ckpt_step={resume_step}, "
                f"manifest={'yes' if resume_manifest else 'no'})"
              + colorama.Style.RESET_ALL)

    use_wandb = (not args.no_wandb) and wandb is not None
    if use_wandb:
        wandb_kwargs = dict(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name or args.n,
            config=dict(
                env=args.env_name,
                seed=args.seed,
                config_path=args.config_path,
                **{f"conf/{k}": str(v) for k, v in conf.items()},
            ),
            dir=str(run_dir),
        )
        # CLI -wandb_run_id wins over the manifest's prior id; lets the sweep
        # launcher pin a deterministic id from the run_name hash. Falls back to
        # the manifest for hand-launched scripts that don't pass an id.
        pinned_id = args.wandb_run_id or (prior_wandb_run_id if is_resume else None)
        if pinned_id:
            wandb_kwargs["id"] = pinned_id
        # `allow` covers both first launch (creates a new run with our id) and
        # resume (re-attaches to it). `must` would reject the first-launch case.
        wandb_kwargs["resume"] = "allow"
        if args.wandb_group:
            wandb_kwargs["group"] = args.wandb_group
        if args.wandb_tags:
            wandb_kwargs["tags"] = list(args.wandb_tags)
        wandb.init(**wandb_kwargs)
        active_wandb_run_id = wandb.run.id if wandb.run is not None else None
    elif wandb is None and not args.no_wandb:
        print(colorama.Fore.YELLOW
              + "wandb not installed, continuing with TB-only logging"
              + colorama.Style.RESET_ALL)
        active_wandb_run_id = None
    else:
        active_wandb_run_id = None
    logger = WandbTBLogger(tb_logger, use_wandb=use_wandb)

    if conf.Task != "JointTrainAgent":
        raise NotImplementedError(f"Task {conf.Task} not implemented")

    assert conf.JointTrainAgent.NumEnvs == 1, (
        "Auto-resume currently assumes NumEnvs=1 "
        f"(got {conf.JointTrainAgent.NumEnvs})"
    )

    # Construct a dummy env to discover action / proprio dims.
    dummy_env = mw_env.build_single_metaworld_env(
        task_name=args.env_name,
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

    world_model = build_world_model(conf, action_dim=action_dim, proprio_dim=proprio_dim)
    # Wrap with torch.compile if available and explicitly enabled.
    # Disabled by default: Dynamo+CUDA 13 + driver 580.65.06 has produced
    # silent SIGBUS on L40S. Set STORM_COMPILE=1 to re-enable.
    if hasattr(torch, "compile") and os.environ.get("STORM_COMPILE", "0") == "1":
        print(colorama.Fore.YELLOW + "Compiling sub-modules with torch.compile..." + colorama.Style.RESET_ALL)
        world_model.storm_transformer = torch.compile(world_model.storm_transformer)
    agent = build_agent(conf, action_dim=action_dim)

    image_h = image_w = conf.BasicSettings.ImageSize
    in_channels = conf.Models.WorldModel.InChannels
    replay_buffer = ReplayBuffer(
        obs_shape=(image_h, image_w, in_channels),
        num_envs=conf.JointTrainAgent.NumEnvs,
        max_length=conf.JointTrainAgent.BufferMaxLength,
        warmup_length=conf.JointTrainAgent.BufferWarmUp,
        store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
    )

    # -------- apply resume (if any) --------
    save_episodes = bool(getattr(conf.BasicSettings, "SaveEpisodesToDisk", True))
    start_iter = 0
    episodes_done_init = 0
    if is_resume:
        wm_ckpt = ckpt_dir / f"world_model_{resume_step}.pth"
        ag_ckpt = ckpt_dir / f"agent_{resume_step}.pth"
        print(colorama.Fore.GREEN
              + f">>> Loading model weights: {wm_ckpt.name}, {ag_ckpt.name}"
              + colorama.Style.RESET_ALL)
        world_model.load_state_dict(torch.load(wm_ckpt, map_location="cuda"))
        agent.load_state_dict(torch.load(ag_ckpt, map_location="cuda"))
        start_iter = int(resume_manifest.get("iter", resume_step))
        episodes_done_init = int(resume_manifest.get("episodes_done", 0))

        # Reload episodes into the ring buffer. Buffer is auto-bounded by
        # its own capacity (keeps newest-first until full).
        if save_episodes:
            stats = replay_buffer.load_from_directory(str(episode_dir))
            print(colorama.Fore.GREEN
                  + f">>> Reloaded {stats['transitions_restored']} transitions "
                    f"from {stats['episodes_restored']} episodes "
                    f"(buffer length: {len(replay_buffer)})"
                  + colorama.Style.RESET_ALL)
            # Prune any episode files on disk that didn't make it back into
            # the (bounded) in-memory ring.
            if stats["kept_ids"]:
                removed = ReplayBuffer.erase_over_episode_files(
                    str(episode_dir), stats["kept_ids"])
                if removed:
                    print(f">>> Pruned {removed} stale episode files")
        else:
            print(colorama.Fore.YELLOW
                  + ">>> SaveEpisodesToDisk=False: buffer starts EMPTY on resume "
                    "(model weights still restored)"
                  + colorama.Style.RESET_ALL)

    try:
        joint_train_world_model_agent(
            env_name=args.env_name,
            conf=conf,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            logger=logger,
            run_name=args.n,
            start_iter=start_iter,
            episodes_done_init=episodes_done_init,
            episode_dir=str(episode_dir),
            save_episodes=save_episodes,
            wandb_run_id=active_wandb_run_id,
            ckpt_dir=str(ckpt_dir),
            manifest_path=str(manifest_path),
        )
        exit_code = 0
    except BaseException as e:
        print(colorama.Fore.RED + f"Training crashed: {type(e).__name__}: {e}" + colorama.Style.RESET_ALL)
        exit_code = 1
        raise
    finally:
        if use_wandb:
            try:
                wandb.finish(exit_code=exit_code)
            except Exception:
                pass
