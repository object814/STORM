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
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device="cuda")
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]


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
