"""PackNet manager (conservative / paper-faithful) for STORM's WorldModel.

Adapted from ``third_party/dreamerv3/packnet_training/packnet.py``. Same
two-tier protocol as the original PackNet (Mallya & Lazebnik, CVPR 2018):

  - PRUNABLE  (ndim >= 2): Linear / Conv weight matrices in the shared
    world-model core. Get the full PackNet treatment — magnitude prune,
    retrain, freeze surviving weights, per-task evaluation masks.
  - SHARED    (ndim == 1): biases, BatchNorm/LayerNorm weight & bias.
    Frozen after the first task's prune+retrain cycle and shared across
    all subsequent tasks unchanged. Matches the paper's "we did not find
    it necessary to learn task-specific biases" / norm-param policy.

Differences vs the dreamer adaptation:
  * STORM's "shared core" is the WorldModel minus the per-task reward and
    termination decoders. We treat the task-head prefixes as a deny-list
    (any param NOT under those prefixes is RSSM) — same convention used
    by ``train_metaworld_sequential.py:TASK_HEAD_PREFIXES`` and the EWC
    adaptation. New shared-core modules added later are auto-tracked.
  * Note on BatchNorm: STORM's encoder uses BatchNorm2d, whose
    ``running_mean`` / ``running_var`` are **buffers**, not parameters.
    They are NOT frozen by this manager and will continue to drift on
    task 2+ data unless the world model is held in ``eval()`` mode (it
    isn't — ``world_model.update`` calls ``self.train()``). This is a
    known limitation of PackNet+BN and matches the dreamer behavior.
"""

from typing import Dict, List

import torch


# Same prefix list the sequential trainer uses to split state dicts.
TASK_HEAD_PREFIXES = ("reward_decoder.", "termination_decoder.")


def _strip_compile_prefix(name: str) -> str:
    return name.replace("._orig_mod.", ".").replace("_orig_mod.", "")


class PackNetManager:
    """Paper-faithful PackNet manager for STORM WorldModel shared core."""

    def __init__(
        self,
        prune_ratio: float = 0.75,
        task_head_prefixes: tuple = TASK_HEAD_PREFIXES,
    ):
        self.prune_ratio = prune_ratio
        self.task_head_prefixes = task_head_prefixes

        # Prunable (weight) state
        self.frozen_mask: Dict[str, torch.Tensor] = {}
        self.task_masks: Dict[int, Dict[str, torch.Tensor]] = {}

        # Shared (bias / norm) state
        self._shared_param_names: List[str] = []
        self._shared_params_frozen: bool = False

        self.num_tasks_packed: int = 0

        # Retrain state
        self._retrain_mode: bool = False
        self._retrain_mask: Dict[str, torch.Tensor] = {}
        self._retrain_task_mask: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Parameter classification
    # ------------------------------------------------------------------
    def _is_rssm_param(self, name: str) -> bool:
        """Shared-core iff the (compile-stripped) name does NOT start with
        any task-head prefix."""
        norm = _strip_compile_prefix(name)
        return not any(norm.startswith(p) for p in self.task_head_prefixes)

    def _is_prunable(self, name: str, param: torch.Tensor) -> bool:
        return self._is_rssm_param(name) and param.ndim >= 2

    def _is_shared(self, name: str, param: torch.Tensor) -> bool:
        return self._is_rssm_param(name) and param.ndim == 1

    # ------------------------------------------------------------------
    # Initialisation: discover shared params
    # ------------------------------------------------------------------
    def register_rssm_params(self, world_model):
        """Call once after the WorldModel is created (and again after any
        ``reset_task_heads`` if shared param identities ever change — they
        don't, but cheap to recall)."""
        self._shared_param_names = []
        n_prunable = 0
        n_prunable_params = 0
        n_shared = 0
        n_shared_params = 0

        for name, param in world_model.named_parameters():
            if self._is_prunable(name, param):
                n_prunable += 1
                n_prunable_params += param.numel()
            elif self._is_shared(name, param):
                self._shared_param_names.append(name)
                n_shared += 1
                n_shared_params += param.numel()

        print(f">>> PackNet: {n_prunable} prunable param groups "
              f"({n_prunable_params:,} params, ndim>=2)")
        print(f">>> PackNet: {n_shared} shared param groups "
              f"({n_shared_params:,} params, ndim==1, frozen after task 1)")

    # ------------------------------------------------------------------
    # Gradient masking (called every WM update step, between unscale + clip)
    # ------------------------------------------------------------------
    def apply_gradient_mask(self, world_model):
        """Zero gradients on frozen weights and (post-task-1) shared params.

        Called AFTER ``scaler.unscale_(optimizer)`` and BEFORE
        ``clip_grad_norm_`` + ``scaler.step``. Uses indexing assignment
        (``grad[mask] = 0``) instead of multiplication to avoid AMP
        ``inf * 0 = nan`` propagation.
        """
        for name, param in world_model.named_parameters():
            if param.grad is None:
                continue

            if self._is_prunable(name, param):
                if self._retrain_mode and name in self._retrain_mask:
                    param.grad.data[self._retrain_mask[name] == 0] = 0.0
                elif name in self.frozen_mask:
                    param.grad.data[self.frozen_mask[name].bool()] = 0.0

            elif self._shared_params_frozen and self._is_shared(name, param):
                param.grad.data.zero_()

    def apply_weight_mask(self, world_model):
        """Re-zero pruned weights after the optimizer step (retrain only).

        Adam's momentum/weight-decay terms can revive pruned weights even
        when their gradients are zero, so we re-mask after every step
        during the retrain phase.
        """
        if not self._retrain_mode:
            return
        for name, param in world_model.named_parameters():
            if name in self._retrain_task_mask:
                param.data.mul_(self._retrain_task_mask[name])

    # ------------------------------------------------------------------
    # Pruning (called once after main training of a task)
    # ------------------------------------------------------------------
    def prune(self, world_model, task_idx: int) -> Dict[str, torch.Tensor]:
        """Magnitude-prune the current task's free weight params, per layer.

        Returns ``{name: mask}`` for prunable params only — 1.0 = keep
        (frozen + this-task surviving), 0.0 = pruned.
        """
        task_mask = {}
        for name, param in world_model.named_parameters():
            if not self._is_prunable(name, param):
                continue

            frozen = self.frozen_mask.get(name, torch.zeros_like(param.data))
            free_mask = (1.0 - frozen).bool()
            num_free = int(free_mask.sum().item())

            if num_free > 0:
                free_magnitudes = param.data.abs()[free_mask]
                k = int(num_free * self.prune_ratio)
                if 0 < k < num_free:
                    threshold = torch.kthvalue(free_magnitudes, k).values.item()
                    survive = frozen.bool() | (
                        free_mask & (param.data.abs() > threshold)
                    )
                elif k >= num_free:
                    survive = frozen.bool()
                else:
                    survive = torch.ones_like(param.data, dtype=torch.bool)
            else:
                survive = frozen.bool()

            mask = survive.float()
            task_mask[name] = mask
            param.data.mul_(mask)

        n_total = sum(
            p.numel() for n, p in world_model.named_parameters()
            if self._is_prunable(n, p)
        )
        n_frozen = sum(v.sum().item() for v in self.frozen_mask.values())
        n_surviving = sum(v.sum().item() for v in task_mask.values())
        n_pruned = n_total - n_surviving
        n_task_surviving = n_surviving - n_frozen
        print(
            f">>> PackNet: Pruned task {task_idx+1} weights "
            f"(ratio={self.prune_ratio:.2f}): "
            f"{int(n_total):,} total, "
            f"{int(n_frozen):,} previously frozen, "
            f"{int(n_task_surviving):,} this-task surviving, "
            f"{int(n_pruned):,} pruned (free for future)"
        )
        return task_mask

    # ------------------------------------------------------------------
    # Retrain mode (called between prune and freeze)
    # ------------------------------------------------------------------
    def start_retrain(self, task_mask: Dict[str, torch.Tensor]):
        self._retrain_mode = True
        self._retrain_task_mask = {k: v.clone() for k, v in task_mask.items()}
        self._retrain_mask = {}
        for name, mask in task_mask.items():
            frozen = self.frozen_mask.get(name, torch.zeros_like(mask))
            self._retrain_mask[name] = mask * (1.0 - frozen)

    def end_retrain(self):
        self._retrain_mode = False
        self._retrain_mask = {}
        self._retrain_task_mask = {}

    # ------------------------------------------------------------------
    # Freeze (called after retrain, finalises the task)
    # ------------------------------------------------------------------
    def freeze_task(self, task_mask: Dict[str, torch.Tensor], task_idx: int):
        self.task_masks[task_idx] = {k: v.clone() for k, v in task_mask.items()}

        for name, mask in task_mask.items():
            if name in self.frozen_mask:
                self.frozen_mask[name] = torch.max(self.frozen_mask[name], mask)
            else:
                self.frozen_mask[name] = mask.clone()

        if not self._shared_params_frozen:
            self._shared_params_frozen = True
            print(f">>> PackNet: Freezing {len(self._shared_param_names)} shared "
                  f"(bias/norm) param groups after task {task_idx+1}")

        self.num_tasks_packed += 1

        n_frozen_w = sum(v.sum().item() for v in self.frozen_mask.values())
        n_total_w = sum(v.numel() for v in self.frozen_mask.values())
        print(
            f">>> PackNet: Frozen task {task_idx+1}: "
            f"{int(n_frozen_w):,}/{int(n_total_w):,} weight params frozen "
            f"({100*n_frozen_w/max(n_total_w,1):.1f}%), "
            f"shared params frozen={self._shared_params_frozen}"
        )

    # ------------------------------------------------------------------
    # Eval-mask application
    # ------------------------------------------------------------------
    def save_rssm_weights(self, world_model) -> Dict[str, torch.Tensor]:
        """Snapshot prunable RSSM weights so eval-mask application is reversible."""
        saved = {}
        for name, param in world_model.named_parameters():
            if self._is_prunable(name, param):
                saved[name] = param.data.clone()
        return saved

    def restore_rssm_weights(self, world_model, saved: Dict[str, torch.Tensor]):
        for name, param in world_model.named_parameters():
            if name in saved:
                param.data.copy_(saved[name])

    def apply_eval_mask(self, world_model, task_idx: int):
        """Zero non-this-task weight positions for evaluation. Always wrap
        in ``save_rssm_weights`` / ``restore_rssm_weights`` so training
        weights are not corrupted."""
        if task_idx not in self.task_masks:
            return
        mask = self.task_masks[task_idx]
        for name, param in world_model.named_parameters():
            if name in mask:
                param.data.mul_(mask[name])

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "frozen_mask": {k: v.cpu() for k, v in self.frozen_mask.items()},
            "task_masks": {
                tid: {k: v.cpu() for k, v in masks.items()}
                for tid, masks in self.task_masks.items()
            },
            "num_tasks_packed": self.num_tasks_packed,
            "prune_ratio": self.prune_ratio,
            "shared_params_frozen": self._shared_params_frozen,
            "shared_param_names": self._shared_param_names,
        }

    def load_state_dict(self, state: dict):
        self.frozen_mask = state["frozen_mask"]
        self.task_masks = state["task_masks"]
        self.num_tasks_packed = state["num_tasks_packed"]
        self.prune_ratio = state.get("prune_ratio", self.prune_ratio)
        self._shared_params_frozen = state.get("shared_params_frozen", False)
        self._shared_param_names = state.get("shared_param_names", [])

    def to_device(self, device: str):
        self.frozen_mask = {k: v.to(device) for k, v in self.frozen_mask.items()}
        self.task_masks = {
            tid: {k: v.to(device) for k, v in masks.items()}
            for tid, masks in self.task_masks.items()
        }
        if self._retrain_mode:
            self._retrain_mask = {
                k: v.to(device) for k, v in self._retrain_mask.items()
            }
            self._retrain_task_mask = {
                k: v.to(device) for k, v in self._retrain_task_mask.items()
            }
