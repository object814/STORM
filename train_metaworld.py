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
import os
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


@torch.no_grad()
def world_model_imagine_data(replay_buffer, world_model, agent,
                             imagine_batch_size, imagine_demo_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    world_model.eval()
    agent.eval()
    sample = replay_buffer.sample(imagine_batch_size, imagine_demo_batch_size, imagine_context_length)
    if replay_buffer.use_proprio:
        sample_obs, sample_action, sample_reward, sample_termination, sample_proprio = sample
    else:
        sample_obs, sample_action, sample_reward, sample_termination = sample
        sample_proprio = None

    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size + imagine_demo_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger,
        sample_proprio=sample_proprio,
    )
    return latent, action, None, None, reward_hat, termination_hat


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------
def joint_train_world_model_agent(
    env_name, conf, replay_buffer, world_model, agent, logger, run_name,
):
    os.makedirs(f"ckpt/{run_name}", exist_ok=True)

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
    episodes_done = 0

    current_obs, current_info = vec_env.reset()
    current_proprio = np.asarray(current_info["proprio"], dtype=np.float32)

    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    context_proprio = deque(maxlen=16)

    total_iters = max_steps // num_envs
    pbar = tqdm(total=total_iters, desc=">>> STORM training", unit="step")

    for total_steps in range(total_iters):
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

        # -------- agent update --------
        if replay_buffer.ready() and total_steps % (max(train_agent_every // num_envs, 1)) == 0:
            log_video = (total_steps % (max(save_every // num_envs, 1)) == 0)
            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_demo_batch_size=imagine_demo_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger,
            )
            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger,
            )

        # -------- checkpointing --------
        if total_steps % (max(save_every // num_envs, 1)) == 0:
            ckpt_dir = Path("ckpt") / run_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(world_model.state_dict(), ckpt_dir / f"world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), ckpt_dir / f"agent_{total_steps}.pth")

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

    run_dir = Path("runs") / args.n
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = Logger(path=str(run_dir))
    shutil.copy(args.config_path, run_dir / "config.yaml")

    use_wandb = (not args.no_wandb) and wandb is not None
    if use_wandb:
        wandb.init(
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
    elif wandb is None and not args.no_wandb:
        print(colorama.Fore.YELLOW
              + "wandb not installed, continuing with TB-only logging"
              + colorama.Style.RESET_ALL)
    logger = WandbTBLogger(tb_logger, use_wandb=use_wandb)

    if conf.Task != "JointTrainAgent":
        raise NotImplementedError(f"Task {conf.Task} not implemented")

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

    try:
        joint_train_world_model_agent(
            env_name=args.env_name,
            conf=conf,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            logger=logger,
            run_name=args.n,
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
