import numpy as np
import os
import random
import unittest
import uuid
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, obs_shape, num_envs, max_length=int(1E6), warmup_length=50000, store_on_gpu=False,
                 action_dim=None, proprio_dim=0) -> None:
        self.store_on_gpu = store_on_gpu
        # action shape: None/0/1 = scalar (Atari), >1 = vector (continuous).
        self.action_is_vector = action_dim is not None and action_dim > 1
        self.action_dim = action_dim if self.action_is_vector else 1
        self.proprio_dim = proprio_dim
        self.use_proprio = proprio_dim > 0
        action_shape = (self.action_dim,) if self.action_is_vector else ()
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device="cuda", requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs, *action_shape), dtype=torch.float32, device="cuda", requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device="cuda", requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device="cuda", requires_grad=False)
            if self.use_proprio:
                self.proprio_buffer = torch.empty((max_length//num_envs, num_envs, proprio_dim), dtype=torch.float32, device="cuda", requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs, *action_shape), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            if self.use_proprio:
                self.proprio_buffer = np.empty((max_length//num_envs, num_envs, proprio_dim), dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def load_trajectory(self, path):
        buffer = pickle.load(open(path, "rb"))
        if self.store_on_gpu:
            self.external_buffer = {name: torch.from_numpy(buffer[name]).to("cuda") for name in buffer}
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]

    def sample_external(self, batch_size, batch_length, to_device="cuda"):
        indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
        if self.store_on_gpu:
            obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        else:
            obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        return obs, action, reward, termination

    def save_episode(self, directory, episode):
        """Write one completed episode to ``directory/{uuid}-{len}.npz``.

        ``episode`` is a dict of 1D/ND numpy arrays with keys ``obs``,
        ``action``, ``reward``, ``termination`` and optionally ``proprio``.
        Uses an atomic ``.tmp`` → rename so a kill mid-write cannot leave a
        truncated file that breaks the next ``load_from_directory`` call.
        """
        assert self.num_envs == 1, "save_episode currently assumes NumEnvs=1"
        os.makedirs(directory, exist_ok=True)
        length = int(episode["reward"].shape[0])
        ep_id = uuid.uuid4().hex
        filename = f"{ep_id}-{length}.npz"
        final_path = os.path.join(directory, filename)
        tmp_path = final_path + ".tmp"
        np.savez_compressed(tmp_path, **episode)
        # np.savez appends ".npz" if the path does not already end in it.
        tmp_with_ext = tmp_path if tmp_path.endswith(".npz") else tmp_path + ".npz"
        os.replace(tmp_with_ext, final_path)
        return final_path

    def load_from_directory(self, directory):
        """Reload episodes from ``directory`` back into the ring buffer.

        Reads newest-first (reverse sorted by filename), appending transitions
        back into the ring until ``self.max_length`` would be exceeded. Returns
        a dict with:
          - ``transitions_restored``: int — total transitions appended
          - ``episodes_restored``: int — number of .npz files consumed
          - ``kept_ids``: set[str] — stems of files kept (for pruning)
        Files older than the ring-buffer cap are listed in the result but not
        deleted here; the caller decides via ``erase_over_episode_files``.
        """
        assert self.num_envs == 1, "load_from_directory currently assumes NumEnvs=1"
        if not os.path.isdir(directory):
            return {"transitions_restored": 0, "episodes_restored": 0, "kept_ids": set()}

        # Sorted ascending so we replay episodes in chronological order; the
        # ring buffer's last_pointer will end up at the most-recent transition.
        files = sorted(f for f in os.listdir(directory) if f.endswith(".npz"))
        if not files:
            return {"transitions_restored": 0, "episodes_restored": 0, "kept_ids": set()}

        cap = self.max_length // self.num_envs
        # Pick the suffix that fits: keep newest until cap is reached.
        lengths = []
        for fname in files:
            try:
                stem = os.path.splitext(fname)[0]
                length = int(stem.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                length = 0
            lengths.append(length)

        # Walk from newest backwards, collect files until we hit cap.
        kept = []
        total = 0
        for fname, length in reversed(list(zip(files, lengths))):
            if total + length > cap and kept:
                break
            kept.append(fname)
            total += length
        kept.reverse()  # chronological again

        transitions_restored = 0
        episodes_restored = 0
        for fname in kept:
            path = os.path.join(directory, fname)
            try:
                with np.load(path) as ep:
                    obs = ep["obs"]
                    action = ep["action"]
                    reward = ep["reward"]
                    termination = ep["termination"]
                    proprio = ep["proprio"] if "proprio" in ep.files else None
            except (OSError, ValueError, EOFError) as e:
                print(f"[ReplayBuffer] skipping corrupt episode {fname}: {e}")
                continue

            n = reward.shape[0]
            for t in range(n):
                step_obs = obs[t:t + 1]  # (1, H, W, C)
                step_action = action[t:t + 1]
                step_reward = reward[t:t + 1]
                step_term = termination[t:t + 1]
                step_proprio = proprio[t:t + 1] if proprio is not None else None
                self.append(step_obs, step_action, step_reward, step_term,
                            proprio=step_proprio)
            transitions_restored += n
            episodes_restored += 1

        kept_ids = {os.path.splitext(f)[0] for f in kept}
        return {
            "transitions_restored": transitions_restored,
            "episodes_restored": episodes_restored,
            "kept_ids": kept_ids,
        }

    @staticmethod
    def erase_over_episode_files(directory, kept_ids):
        """Delete any ``.npz`` in ``directory`` whose stem is not in ``kept_ids``."""
        if not os.path.isdir(directory):
            return 0
        removed = 0
        for fname in os.listdir(directory):
            if not fname.endswith(".npz"):
                continue
            if os.path.splitext(fname)[0] in kept_ids:
                continue
            try:
                os.remove(os.path.join(directory, fname))
                removed += 1
            except OSError:
                pass
        return removed

    @staticmethod
    def prune_episode_dir_to_cap(directory, max_total_steps):
        """FIFO-prune old ``.npz`` files so the directory holds at most
        ``max_total_steps`` transitions across all remaining episodes.

        Filenames must follow the ``{id}-{length}.npz`` convention; files that
        don't parse are left alone. Deletes oldest-first (sorted ascending by
        filename, which for uuid4 hex prefixes is effectively random but
        stable — sufficient for FIFO at our granularity).
        """
        if not os.path.isdir(directory) or max_total_steps <= 0:
            return 0
        entries = []
        for fname in os.listdir(directory):
            if not fname.endswith(".npz"):
                continue
            path = os.path.join(directory, fname)
            try:
                stem = os.path.splitext(fname)[0]
                length = int(stem.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            entries.append((mtime, fname, length, path))

        # Oldest first by mtime so FIFO is actually first-in-first-out.
        entries.sort(key=lambda e: e[0])
        total = sum(e[2] for e in entries)
        removed = 0
        idx = 0
        while total > max_total_steps and idx < len(entries):
            _, _, length, path = entries[idx]
            try:
                os.remove(path)
                total -= length
                removed += 1
            except OSError:
                pass
            idx += 1
        return removed

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, to_device="cuda"):
        if self.store_on_gpu:
            obs, action, reward, termination, proprio = [], [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(torch.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    if self.use_proprio:
                        proprio.append(torch.stack([self.proprio_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)
            if self.use_proprio:
                proprio = torch.cat(proprio, dim=0)
        else:
            obs, action, reward, termination, proprio = [], [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(np.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(np.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(np.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(np.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    if self.use_proprio:
                        proprio.append(np.stack([self.proprio_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, to_device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float().cuda() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).cuda()
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).cuda()
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).cuda()
            if self.use_proprio:
                proprio = torch.from_numpy(np.concatenate(proprio, axis=0)).cuda()

        if self.use_proprio:
            return obs, action, reward, termination, proprio
        return obs, action, reward, termination

    def append(self, obs, action, reward, termination, proprio=None):
        # obs: (N, H, W, C) uint8
        # action: (N,) int, or (N, A) float for continuous
        # reward/termination: (N,) float/bool
        # proprio (optional): (N, D) float
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
            self.action_buffer[self.last_pointer] = torch.from_numpy(np.asarray(action, dtype=np.float32))
            self.reward_buffer[self.last_pointer] = torch.from_numpy(np.asarray(reward, dtype=np.float32))
            self.termination_buffer[self.last_pointer] = torch.from_numpy(np.asarray(termination, dtype=np.float32))
            if self.use_proprio and proprio is not None:
                self.proprio_buffer[self.last_pointer] = torch.from_numpy(np.asarray(proprio, dtype=np.float32))
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination
            if self.use_proprio and proprio is not None:
                self.proprio_buffer[self.last_pointer] = proprio

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs
