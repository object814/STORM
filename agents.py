import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # Build the returns as a Python list and stack at the end. The original
    # implementation used in-place index assignment (gamma_return[:, t] = ...)
    # which is incompatible with autograd when the result feeds a loss that
    # requires gradient through `rewards` — every assignment bumps the
    # tensor's version counter, and earlier iterations' saved tensors
    # then fail the version check on backward (RuntimeError: variable
    # modified by inplace operation, output of AsStridedBackward0). REINFORCE
    # always detaches lambda_return before any backward path, so it never
    # tripped this; the dreamer-style pathwise actor update does not detach
    # and does trip it. Listing + stack is the same math, no in-place ops.
    next_return = values[:, -1]
    returns = []  # will hold returns for t = batch_length-1 .. 0 (reversed)
    for t in reversed(range(batch_length)):
        cur_return = (
            rewards[:, t]
            + gamma * inv_termination[:, t] * (1 - lam) * values[:, t]
            + gamma * inv_termination[:, t] * lam * next_return
        )
        returns.append(cur_return)
        next_return = cur_return
    # Reverse to put t=0 first, then stack along time axis.
    returns.reverse()
    return torch.stack(returns, dim=1)


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef, continuous_action=False) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.continuous_action = continuous_action
        self.action_dim = action_dim
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
        # For continuous actions, the head outputs mean and std.
        actor_out_dim = 2 * action_dim if continuous_action else action_dim
        self.actor = nn.Sequential(
            *actor,
            nn.Linear(hidden_dim, actor_out_dim)
        )
        # Bounds for the squashed normal std (matches dreamer "normal" dist).
        self._min_std = 0.1
        self._max_std = 1.0

        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])

        self.critic = nn.Sequential(
            *critic,
            nn.Linear(hidden_dim, 255)
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def _build_action_dist(self, raw):
        """Return a torch distribution over actions.

        Discrete: Categorical over logits.
        Continuous: tanh-squashed diagonal Normal with bounded std
                    (mean is tanh'd). Matches the "normal" variant in
                    dreamer's MLP head.
        """
        if not self.continuous_action:
            return distributions.Categorical(logits=raw)
        mean, std = torch.chunk(raw, 2, dim=-1)
        mean = torch.tanh(mean)
        std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
        base = distributions.Normal(mean, std)
        return distributions.Independent(base, 1)

    def policy(self, x):
        logits = self.actor(x)
        return logits

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            raw = self.policy(latent)
            dist = self._build_action_dist(raw)
            if self.continuous_action:
                if greedy:
                    action = torch.tanh(torch.chunk(raw, 2, dim=-1)[0])
                else:
                    action = dist.sample()
                action = action.clamp(-1.0, 1.0)
            else:
                if greedy:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
        return action

    def sample_grad(self, latent):
        """Reparameterized sampler used inside differentiable imagination.

        Unlike `sample`, this method:
          * does NOT detach (no @torch.no_grad),
          * uses Normal.rsample instead of Normal.sample,
        so the returned action carries gradient back to actor params via
        the reparameterization `action = tanh(mean) + std * eps`. Mirrors
        dreamer's tools.ContDist.sample, which under the hood calls
        rsample (third_party/dreamerv3/tools.py:608-612).

        No second tanh is applied after rsample — matching dreamer's
        `dist: 'normal'` actor (configs.yaml:50). The bounded mean
        (via tanh) plus bounded std (via sigmoid) keep typical samples
        in roughly [-2, 2]; the env wrappers handle final clipping for
        any out-of-range samples when this action is actually executed.

        Continuous actions only — discrete distributions don't admit
        reparameterization and would need a different gradient estimator
        (Gumbel-softmax) which we don't support here. STORM's metaworld
        config uses continuous actions, so this is fine.
        """
        assert self.continuous_action, (
            "sample_grad is only implemented for continuous actions. "
            "Discrete policies must use the REINFORCE path (agent.update)."
        )
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            raw = self.policy(latent)
            mean, std = torch.chunk(raw, 2, dim=-1)
            mean = torch.tanh(mean)
            std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
            base = distributions.Normal(mean, std)
            dist = distributions.Independent(base, 1)
            action = dist.rsample()
        return action

    def update_pathwise(self, latent, action, reward, termination, logger=None):
        """Dreamer-style actor-critic update via pathwise (dynamics) gradients.

        Inputs come from WorldModel.imagine_data_grad and still carry
        autograd from the actor's parameters through:
            actor -> action -> world model -> reward / termination
        so backprop on the lambda return updates the actor directly,
        without REINFORCE's score-function term.

        Critic loss: unchanged from `update` — symlog-twohot regression
        to detached lambda-return targets, plus slow-critic regularization.

        Actor loss: `-mean(normalized lambda_return)`. Gradient flows
        through `lambda_return -> reward(imag_feat) -> imag_action ->
        actor_params`. The entropy bonus is preserved as a regularizer
        with the same `entropy_coef` as the REINFORCE path.

        This is the only thing that prevents the corner-saturation failure
        mode at task boundaries in sequential learning: when the reward
        decoder is action-blind (∂reward/∂action ≈ 0, e.g. early in a new
        task before it has fit task-specific reward variance), this
        actor loss has gradient ≈ 0 on `mean` and the actor doesn't drift.
        REINFORCE's `log_prob * advantage` would still drift `mean` under
        critic noise and bias, leading to bang-bang corner policies.
        """
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = self._build_action_dist(logits[:, :-1])
            entropy = dist.entropy()

            # Lambda return computed against the slow critic bootstrap.
            # `reward` and `termination` are differentiable in actor params
            # via the imagined rollout's reparameterized actions.
            slow_value = self.slow_value(latent)
            lambda_return = calc_lambda_return(
                reward, slow_value, termination, self.gamma, self.lambd,
            )
            value = self.symlog_twohot_loss.decode(raw_value)

            # Critic loss — same as REINFORCE path (targets detached).
            value_loss = self.symlog_twohot_loss(
                raw_value[:, :-1], lambda_return.detach(),
            )
            slow_lambda_return = calc_lambda_return(
                reward.detach(), slow_value,
                termination.detach(), self.gamma, self.lambd,
            )
            slow_value_regularization_loss = self.symlog_twohot_loss(
                raw_value[:, :-1], slow_lambda_return.detach(),
            )

            # Reward-scale normalization via the same 5/95-percentile EMA
            # used by the REINFORCE path. The EMA is a statistic that
            # MUST be detached: EMAScalar.update writes
            # `self.scalar = self.scalar * decay + value * (1-decay)`
            # WITHOUT detaching, so without an explicit .detach() here the
            # EMA's running state would accumulate the autograd graph
            # across calls. REINFORCE detaches its policy_loss before
            # backward and never traverses this graph, so it's a silent
            # memory leak there. Pathwise backprops through `norm_ratio`,
            # which depends on the EMA scalar, and would try to backward
            # through previously-backwarded-and-freed graphs.
            lower_bound = self.lowerbound_ema(
                percentile(lambda_return, 0.05).detach()
            )
            upper_bound = self.upperbound_ema(
                percentile(lambda_return, 0.95).detach()
            )
            S = upper_bound - lower_bound
            norm_ratio = torch.max(torch.ones(1, device=lambda_return.device), S)
            # `norm_ratio` is a scaling statistic — detach so gradient flows
            # ONLY through the numerator `lambda_return`, not through the
            # normalization itself (matches dreamer's normed_target where
            # offset/scale come from a detached reward EMA).
            actor_objective = (lambda_return / norm_ratio.detach()).mean()
            policy_loss = -actor_objective

            entropy_loss = entropy.mean()

            loss = (policy_loss + value_loss + slow_value_regularization_loss
                    - self.entropy_coef * entropy_loss)

        if not torch.isfinite(loss):
            stats = {
                "policy": policy_loss.item(),
                "value": value_loss.item(),
                "entropy": entropy_loss.item(),
                "logits_absmax": logits.abs().max().item(),
                "reward_min_max": (reward.min().item(), reward.max().item()),
                "lambda_return_absmax": lambda_return.abs().max().item(),
            }
            raise RuntimeError(
                f"Non-finite actor-critic loss (pathwise): {stats}"
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        if torch.isfinite(grad_norm):
            self.scaler.step(self.optimizer)
        elif logger is not None:
            logger.log('ActorCritic/skipped_step', 1.0)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item())
            logger.log('ActorCritic/value_loss', value_loss.item())
            logger.log('ActorCritic/entropy_loss', entropy_loss.item())
            logger.log('ActorCritic/S', S.item())
            logger.log('ActorCritic/norm_ratio', norm_ratio.item())
            logger.log('ActorCritic/total_loss', loss.item())
            logger.log('ActorCritic/grad_norm', grad_norm.item())

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        if self.continuous_action:
            # (B, 1, A) -> (B, A) numpy
            return action.detach().cpu().squeeze(1).float().numpy()
        return action.detach().cpu().squeeze(-1).numpy()

    def update(self, latent, action, old_logprob, old_value, reward, termination, logger=None):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = self._build_action_dist(logits[:, :-1])
            if self.continuous_action:
                log_prob = dist.log_prob(action.clamp(-0.999, 0.999))
            else:
                log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        if not torch.isfinite(loss):
            stats = {
                "policy": policy_loss.item(), "value": value_loss.item(),
                "entropy": entropy_loss.item(),
                "logits_absmax": logits.abs().max().item(),
                "reward_min_max": (reward.min().item(), reward.max().item()),
                "lambda_return_absmax": lambda_return.abs().max().item(),
            }
            raise RuntimeError(f"Non-finite actor-critic loss: {stats}")

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        if torch.isfinite(grad_norm):
            self.scaler.step(self.optimizer)
        elif logger is not None:
            logger.log('ActorCritic/skipped_step', 1.0)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item())
            logger.log('ActorCritic/value_loss', value_loss.item())
            logger.log('ActorCritic/entropy_loss', entropy_loss.item())
            logger.log('ActorCritic/S', S.item())
            logger.log('ActorCritic/norm_ratio', norm_ratio.item())
            logger.log('ActorCritic/total_loss', loss.item())
            logger.log('ActorCritic/grad_norm', grad_norm.item())
