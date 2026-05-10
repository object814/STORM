"""EWC (Elastic Weight Consolidation) manager for STORM's WorldModel.

Standard per-task EWC (Kirkpatrick et al. 2017):
  L_total = L_task(θ) + λ · Σ_t Σ_i F_{t,i} · (θ_i - θ*_{t,i})²

Adapted from ``third_party/dreamerv3/ewc_training/ewc.py``. Key differences:

* "Shared core" in STORM is the WorldModel minus the per-task reward and
  termination decoders. Rather than enumerate positive prefixes (encoder.,
  storm_transformer., dist_head., image_decoder., proprio_*), we treat the
  task-head prefixes as a deny-list and protect everything else. New
  shared-core modules added to WorldModel later are automatically tracked
  without code changes (mirrors the convention in
  ``train_metaworld_sequential.py:TASK_HEAD_PREFIXES``).
* Fisher is computed by replicating the forward path of
  ``WorldModel.update`` (without the optimizer step) using mini-batches
  drawn from the current task's ReplayBuffer.
* AMP (bfloat16) is kept on for the forward pass; squared gradients are
  cast to FP32 before accumulation, identical to the dreamer impl.
"""

from typing import Dict, List, Tuple

import torch

from sub_models.attention_blocks import get_subsequent_mask_with_batch_length


# Same prefix list the sequential trainer uses to split state dicts.
TASK_HEAD_PREFIXES = ("reward_decoder.", "termination_decoder.")


def _strip_compile_prefix(name: str) -> str:
    """Normalise away ``_orig_mod.`` prefixes that ``torch.compile`` injects."""
    return name.replace("._orig_mod.", ".").replace("_orig_mod.", "")


class EWCManager:
    """EWC regularization manager for STORM WorldModel shared core.

    Standard per-task EWC: each completed task contributes one (Fisher,
    param-snapshot) pair, and the penalty sums all of them.
    """

    def __init__(self, lambda_ewc: float = 5000.0,
                 task_head_prefixes: tuple = TASK_HEAD_PREFIXES):
        self.lambda_ewc = lambda_ewc
        self.task_head_prefixes = task_head_prefixes

        self.regularization_terms: Dict[int, dict] = {}
        self.num_tasks_consolidated: int = 0

        # Pre-flattened penalty cache (rebuilt after each consolidation).
        self._penalty_cache_valid = False
        self._cached_fishers: List[torch.Tensor] = []
        self._cached_params: List[torch.Tensor] = []
        self._param_name_to_slice: Dict[str, Tuple[int, int]] = {}
        self._rssm_param_names: List[str] = []

    def _is_rssm_param(self, name: str) -> bool:
        """A param is shared-core iff its (compile-stripped) name does NOT
        start with any per-task head prefix."""
        norm = _strip_compile_prefix(name)
        return not any(norm.startswith(p) for p in self.task_head_prefixes)

    # ------------------------------------------------------------------
    # Fisher computation (called once per task boundary)
    # ------------------------------------------------------------------
    def compute_fisher(
        self,
        world_model,
        replay_buffer,
        batch_size: int,
        batch_length: int,
        num_batches: int = 50,
        device: str = "cuda",
    ):
        """Compute diagonal Fisher for shared-core parameters.

        Replicates ``WorldModel.update`` forward path but skips the optimizer
        step. AMP stays enabled for forward+backward; squared gradients are
        cast to FP32 before accumulating to avoid bf16 underflow.

        Returns ``(importance, task_param)`` dicts keyed by parameter name.
        Both task-head params and shared-core params receive gradients during
        backward (so head-loss gradients still flow through the transformer
        into the encoder), but only shared-core gradients are stored.
        """
        importance = {}
        task_param = {}
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name):
                importance[name] = torch.zeros_like(
                    param.data, dtype=torch.float32, device=device,
                )
                task_param[name] = param.data.clone().float()

        was_training = world_model.training
        world_model.eval()
        # All params need grad for the forward to produce non-None .grad after
        # backward; we already filter by name when accumulating.
        world_model.requires_grad_(True)

        use_amp = world_model.use_amp
        ce_kl = world_model.categorical_kl_div_loss
        mse = world_model.mse_loss_func
        bce = world_model.bce_with_logits_loss_func
        symlog = world_model.symlog_twohot_loss_func

        for _ in range(num_batches):
            sample = replay_buffer.sample(batch_size, 0, batch_length)
            if world_model.use_proprio:
                obs, action, reward, termination, proprio = sample
            else:
                obs, action, reward, termination = sample
                proprio = None

            world_model.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                # ---- forward (mirrors WorldModel.update body) ----
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

                recon_loss = mse(obs_hat, obs)
                if world_model.use_proprio:
                    proprio_loss = ((proprio_hat - proprio) ** 2).sum(dim=-1).mean()
                else:
                    proprio_loss = torch.tensor(0.0, device=obs.device)
                reward_loss = symlog(reward_hat, reward)
                term_loss = bce(term_hat, termination)
                dyn_loss, _ = ce_kl(post_logits[:, 1:].detach(), prior_logits[:, :-1])
                rep_loss, _ = ce_kl(post_logits[:, 1:], prior_logits[:, :-1].detach())
                total_loss = (
                    recon_loss + proprio_loss + reward_loss + term_loss
                    + 0.5 * dyn_loss + 0.1 * rep_loss
                )

            # No GradScaler — we want raw gradients for Fisher.
            total_loss.backward()

            for name, param in world_model.named_parameters():
                if name in importance and param.grad is not None:
                    grad_fp32 = param.grad.data.float()
                    importance[name] += (grad_fp32 ** 2) / num_batches

        world_model.zero_grad(set_to_none=True)
        world_model.requires_grad_(False)
        if was_training:
            world_model.train()

        return importance, task_param

    # ------------------------------------------------------------------
    # Consolidation (called once per task boundary)
    # ------------------------------------------------------------------
    def consolidate(self, world_model, importance, task_param, task_idx: int):
        """Store Fisher + param snapshot and rebuild penalty cache."""
        self.regularization_terms[task_idx] = {
            "importance": {k: v.clone() for k, v in importance.items()},
            "task_param": {k: v.clone() for k, v in task_param.items()},
        }
        self.num_tasks_consolidated += 1
        self._rebuild_penalty_cache(world_model)

    def _rebuild_penalty_cache(self, world_model):
        """Pre-flatten Fisher diagonals and param snapshots for fast penalty."""
        if self.num_tasks_consolidated == 0:
            self._penalty_cache_valid = False
            return

        device = next(iter(
            next(iter(self.regularization_terms.values()))["importance"].values()
        )).device

        self._rssm_param_names = []
        self._param_name_to_slice = {}
        offset = 0
        first_reg = next(iter(self.regularization_terms.values()))
        for name, param in world_model.named_parameters():
            if self._is_rssm_param(name) and name in first_reg["importance"]:
                n = param.numel()
                self._rssm_param_names.append(name)
                self._param_name_to_slice[name] = (offset, offset + n)
                offset += n

        total_params = offset

        self._cached_fishers = []
        self._cached_params = []
        for task_idx in sorted(self.regularization_terms.keys()):
            reg = self.regularization_terms[task_idx]
            fisher_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            params_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            for name in self._rssm_param_names:
                s, e = self._param_name_to_slice[name]
                fisher_flat[s:e] = reg["importance"][name].flatten()
                params_flat[s:e] = reg["task_param"][name].flatten()
            self._cached_fishers.append(fisher_flat)
            self._cached_params.append(params_flat)

        self._penalty_cache_valid = True

    # ------------------------------------------------------------------
    # Penalty (called every WM update — must be fast)
    # ------------------------------------------------------------------
    def penalty(self, world_model) -> torch.Tensor:
        """Compute EWC penalty using pre-flattened tensors. Returns a scalar
        already multiplied by ``lambda_ewc``."""
        device = next(world_model.parameters()).device
        if not self._penalty_cache_valid or self.num_tasks_consolidated == 0:
            return torch.tensor(0.0, device=device)

        # Map normalised name → live param so we tolerate torch.compile
        # injecting/removing _orig_mod. between Fisher and penalty time.
        param_dict = {
            _strip_compile_prefix(n): p for n, p in world_model.named_parameters()
        }

        reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for name in self._rssm_param_names:
            norm = _strip_compile_prefix(name)
            s, e = self._param_name_to_slice[name]
            param_flat = param_dict[norm].flatten().float()
            for fisher_flat, params_flat in zip(self._cached_fishers, self._cached_params):
                diff = param_flat - params_flat[s:e]
                reg_loss = reg_loss + (fisher_flat[s:e] * diff ** 2).sum()

        return self.lambda_ewc * reg_loss

    # ------------------------------------------------------------------
    # Save / Load (resume support)
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "regularization_terms": self.regularization_terms,
            "num_tasks_consolidated": self.num_tasks_consolidated,
            "lambda_ewc": self.lambda_ewc,
        }

    def load_state_dict(self, state, world_model=None):
        self.regularization_terms = state["regularization_terms"]
        self.num_tasks_consolidated = state["num_tasks_consolidated"]
        self.lambda_ewc = state.get("lambda_ewc", self.lambda_ewc)
        self._penalty_cache_valid = False
        if world_model is not None and self.num_tasks_consolidated > 0:
            self._rebuild_penalty_cache(world_model)
