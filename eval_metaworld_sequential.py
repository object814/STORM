"""Standalone evaluation for a sequential-STORM task checkpoint.

Loads the three split checkpoints written by ``train_metaworld_sequential.py``
(``rssm.pt`` + ``task_heads.pt`` + ``actor_critic.pt``), wires them into a
``WorldModel`` + ``ActorCriticAgent``, and runs episodes in the real
metaworld env using the same rollout pipeline as the in-training
``_rollout_one_episode`` so eval behavior matches the wandb eval/* curves.

Usage:
    python eval_metaworld_sequential.py \\
        --task-dir logdir/sequential_storm/<run>/task2_metaworld_pick-place-v3 \\
        --task-name metaworld_pick-place-v3 \\
        --config_path config_files/STORM_metaworld.yaml \\
        --episodes 10

Optional: ``--greedy`` (deterministic policy mode), ``--save-video DIR``
(dump one mp4 per episode), ``--seed`` (env seed offset).
"""
import argparse
import os
import sys
from collections import deque
from pathlib import Path

import colorama
import numpy as np
import torch
from einops import rearrange

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import agents
import envs.metaworld_env as mw_env
from sub_models.world_models import WorldModel
from train_metaworld_sequential import (
    load_rssm_into,
    load_task_heads_into,
)
from utils import load_config, seed_np_torch


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


@torch.no_grad()
def rollout_episode(world_model, agent, env, greedy, frames_out=None):
    """Run one episode and return (total_reward, steps, success, action_stats).

    Mirrors `_rollout_one_episode` in train_metaworld_sequential.py so the
    behavior matches the in-training eval. If `frames_out` is a list,
    the per-step rendered RGB frame from the topview camera is appended.
    """
    world_model.eval()
    agent.eval()

    obs, info = env.reset()
    proprio = (np.asarray(info["proprio"], dtype=np.float32)
               if "proprio" in info else None)

    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    context_proprio = deque(maxlen=16)

    total_reward = 0.0
    steps = 0
    success = 0.0
    done = False

    abs_means = []
    sat_fracs = []

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
                greedy=greedy,
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

        # Action stats — picks up the corner-saturation pattern we suspect.
        a = np.asarray(action, dtype=np.float32)
        abs_means.append(float(np.abs(a).mean()))
        sat_fracs.append(float((np.abs(a) > 0.95).mean()))

        if frames_out is not None:
            # obs is (N, H, W, C); take env 0, channels 0:3 (topview).
            frames_out.append(np.asarray(obs[0, :, :, :3], dtype=np.uint8))

        obs, reward, term, trunc, info = env.step(action)
        proprio = (np.asarray(info["proprio"], dtype=np.float32)
                   if "proprio" in info else None)

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

    action_stats = {
        "abs_mean": float(np.mean(abs_means)) if abs_means else 0.0,
        "saturated_frac": float(np.mean(sat_fracs)) if sat_fracs else 0.0,
    }
    return total_reward, steps, success, action_stats


def save_video(frames, path, fps=30):
    """Best-effort video save. Falls back to a stack of PNGs if no encoder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(path), frames, fps=fps)
        return str(path)
    except Exception as e:
        # Fallback: write per-frame PNGs into a sibling dir.
        try:
            import imageio.v2 as imageio
            png_dir = path.with_suffix("")
            png_dir.mkdir(parents=True, exist_ok=True)
            for i, fr in enumerate(frames):
                imageio.imwrite(str(png_dir / f"f{i:04d}.png"), fr)
            return str(png_dir)
        except Exception as e2:
            print(f"[eval] video save failed: {e}; png fallback also failed: {e2}")
            return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-dir", type=str, required=True,
                   help="Path to task{N}_<name>/ with rssm.pt + task_heads.pt + actor_critic.pt")
    p.add_argument("--task-name", type=str, required=True,
                   help="metaworld task, e.g. metaworld_pick-place-v3")
    p.add_argument("--config_path", type=str,
                   default="config_files/STORM_metaworld.yaml")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true",
                   help="Use deterministic actor (mean action). Default: stochastic.")
    p.add_argument("--save-video", type=str, default=None,
                   help="If given, save mp4 of each episode under this dir.")
    args, remaining_opts = p.parse_known_args()

    task_dir = Path(args.task_dir).expanduser().resolve()
    rssm_path = task_dir / "rssm.pt"
    heads_path = task_dir / "task_heads.pt"
    ac_path = task_dir / "actor_critic.pt"
    for f in (rssm_path, heads_path, ac_path):
        if not f.exists():
            raise FileNotFoundError(f"Missing checkpoint: {f}")

    conf = load_config(args.config_path)
    if remaining_opts:
        # Same yacs-override convention as the training entry point so you
        # can pass e.g. `Models.WorldModel.TransformerHiddenDim 1024`.
        conf.defrost()
        conf.merge_from_list(remaining_opts)
        conf.freeze()
    seed_np_torch(seed=args.seed)

    # ---- Discover action_dim / proprio_dim from a dummy env ----
    dummy = mw_env.build_single_metaworld_env(
        task_name=args.task_name,
        image_size=conf.BasicSettings.ImageSize,
        seed=args.seed,
        camera_names=tuple(conf.Env.CameraNames),
        time_limit=conf.Env.TimeLimit,
        reward_range=(conf.Env.RewardMin, conf.Env.RewardMax),
    )
    action_dim = int(np.prod(dummy.action_space.shape))
    proprio_dim = dummy.proprio_dim
    try:
        dummy.close()
    except Exception:
        pass

    print(colorama.Fore.GREEN
          + f"action_dim={action_dim}, proprio_dim={proprio_dim}, "
            f"task={args.task_name}, greedy={args.greedy}"
          + colorama.Style.RESET_ALL)

    # ---- Build models and load weights ----
    world_model = build_world_model(conf, action_dim=action_dim, proprio_dim=proprio_dim)
    load_rssm_into(world_model, rssm_path)
    load_task_heads_into(world_model, heads_path)

    agent = build_agent(conf, action_dim=action_dim)
    agent.load_state_dict(torch.load(ac_path, map_location="cuda"))

    # ---- Build the eval env (vec env with num_envs=1, same as in-training eval) ----
    env = mw_env.build_metaworld_vec_env(
        task_name=args.task_name,
        image_size=conf.BasicSettings.ImageSize,
        num_envs=1,
        seed=args.seed + 1000,
        camera_names=tuple(conf.Env.CameraNames),
        time_limit=conf.Env.TimeLimit,
        reward_range=(conf.Env.RewardMin, conf.Env.RewardMax),
    )

    rewards, lengths, successes = [], [], []
    abs_means, sat_fracs = [], []
    video_dir = Path(args.save_video).expanduser().resolve() if args.save_video else None

    try:
        for ep in range(args.episodes):
            frames = [] if video_dir is not None else None
            r, s, suc, stats = rollout_episode(
                world_model, agent, env,
                greedy=args.greedy, frames_out=frames,
            )
            rewards.append(r)
            lengths.append(s)
            successes.append(suc)
            abs_means.append(stats["abs_mean"])
            sat_fracs.append(stats["saturated_frac"])
            print(
                f"  ep {ep+1:>3}/{args.episodes}: "
                f"R={r:8.2f} L={s:>4} success={suc:.1f}  "
                f"|a|={stats['abs_mean']:.3f}  sat={stats['saturated_frac']:.2f}"
            )
            if video_dir is not None and frames:
                out_path = video_dir / f"ep{ep+1:03d}.mp4"
                saved = save_video(frames, out_path, fps=30)
                if saved:
                    print(f"      video -> {saved}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    print()
    print(colorama.Fore.CYAN + "=" * 60 + colorama.Style.RESET_ALL)
    print(f"  Task:          {args.task_name}")
    print(f"  Checkpoint:    {task_dir}")
    print(f"  Episodes:      {args.episodes}  (greedy={args.greedy})")
    print(f"  Mean reward:   {np.mean(rewards):8.2f}  "
          f"(min {np.min(rewards):.2f}, max {np.max(rewards):.2f})")
    print(f"  Mean length:   {np.mean(lengths):.1f}")
    print(f"  Success rate:  {np.mean(successes):.2%}")
    print(f"  Action |a|:    {np.mean(abs_means):.3f}  "
          f"(std across eps {np.std(abs_means):.3f})")
    print(f"  Action sat>.95 frac: {np.mean(sat_fracs):.3f}  "
          f"(std across eps {np.std(sat_fracs):.3f})")
    print(colorama.Fore.CYAN + "=" * 60 + colorama.Style.RESET_ALL)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("EGL_LOG_LEVEL", "fatal")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
