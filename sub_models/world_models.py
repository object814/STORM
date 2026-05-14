import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

from sub_models.functions_losses import SymLogTwoHotLoss
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from sub_models.transformer_model import StochasticTransformerKVCache
import agents


class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width, input_size=64) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        # Each stride-2 conv halves the spatial resolution. We start after
        # the stem (//2) and keep halving until we hit `final_feature_width`.
        # Supporting arbitrary `input_size` lets us train at 128x128 etc.
        feature_width = input_size // 2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels*final_feature_width*final_feature_width, bias=False))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_channels, H=final_feature_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs.clamp_min(1e-8))
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        # inplace=False — the backbone is called T times inside the pathwise
        # imagination graph (imagine_data_grad); in-place ReLU at step i+1
        # would clobber a tensor saved for step i's backward. See the same
        # note in StochasticTransformerKVCache.stem.
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        # inplace=False — see RewardDecoder note.
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class ProprioEncoder(nn.Module):
    """Small MLP that embeds a proprio vector into a flat feature."""
    def __init__(self, proprio_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(proprio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, D)
        B, L = x.shape[:2]
        x = x.reshape(B * L, -1)
        x = self.net(x)
        return x.reshape(B, L, -1)


class ProprioDecoder(nn.Module):
    """MLP head that reconstructs proprio from the stochastic latent."""
    def __init__(self, stoch_dim, hidden_dim, proprio_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(stoch_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, proprio_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, sample):
        B, L = sample.shape[:2]
        x = sample.reshape(B * L, -1)
        x = self.net(x)
        return x.reshape(B, L, -1)


class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads,
                 continuous_action=False, proprio_dim=0, proprio_hidden_dim=256, proprio_embed_dim=256,
                 input_size=64):
        super().__init__()
        self.transformer_hidden_dim = transformer_hidden_dim
        self.final_feature_width = 4
        self.stoch_dim = 32
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim
        # DEBUG: AMP disabled to test the BN+bf16 overflow hypothesis. BN
        # backward in train mode involves 1/var^(3/2); with task-1-tuned
        # encoder applied to task-2 images, low-variance channels can
        # produce Inf in bf16 but stay finite in fp32. If sequential
        # learning works with use_amp=False, the architectural fix is to
        # replace BatchNorm with LayerNorm (matching dreamer). Restore to
        # True after the experiment.
        self.use_amp = False
        # DEBUG: end AMP-disable
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.continuous_action = continuous_action
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.use_proprio = proprio_dim > 0
        self.input_size = input_size

        self.encoder = EncoderBN(
            in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width,
            input_size=input_size,
        )
        image_feat_dim = self.encoder.last_channels*self.final_feature_width*self.final_feature_width
        if self.use_proprio:
            self.proprio_encoder = ProprioEncoder(
                proprio_dim=proprio_dim, hidden_dim=proprio_hidden_dim, out_dim=proprio_embed_dim
            )
            combined_feat_dim = image_feat_dim + proprio_embed_dim
            self.proprio_decoder = ProprioDecoder(
                stoch_dim=self.stoch_flattened_dim, hidden_dim=proprio_hidden_dim, proprio_dim=proprio_dim
            )
        else:
            combined_feat_dim = image_feat_dim
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_dim,
            feat_dim=transformer_hidden_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            max_length=transformer_max_length,
            dropout=0.1,
            continuous_action=continuous_action,
        )
        self.dist_head = DistHead(
            image_feat_dim=combined_feat_dim,
            transformer_hidden_dim=transformer_hidden_dim,
            stoch_dim=self.stoch_dim
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=in_channels,
            stem_channels=32,
            final_feature_width=self.final_feature_width
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=transformer_hidden_dim
        )
        # Zero-init head output layers. Matches dreamerv3's outscale=0 for
        # reward / continuation heads (configs.yaml; see uniform_weight_init
        # with given_scale=0 → limit=0 → all weights 0). Fresh heads then
        # predict exactly 0 on the first forward pass, so initial reward /
        # termination losses are tiny and don't shock the rest of the model
        # via backprop on the first few batches.
        torch.nn.init.zeros_(self.reward_decoder.head.weight)
        torch.nn.init.zeros_(self.reward_decoder.head.bias)
        torch.nn.init.zeros_(self.termination_decoder.head[0].weight)
        torch.nn.init.zeros_(self.termination_decoder.head[0].bias)

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _encode_combined(self, obs, proprio=None):
        """Encode image (and optional proprio) into the combined flat embedding
        fed to the post distribution head."""
        image_embedding = self.encoder(obs)
        if self.use_proprio and proprio is not None:
            proprio_embedding = self.proprio_encoder(proprio)
            return torch.cat([image_embedding, proprio_embedding], dim=-1)
        return image_embedding

    def encode_obs(self, obs, proprio=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self._encode_combined(obs, proprio)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            prior_logits = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding
            prior_sample = self.stright_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            if self.continuous_action:
                action_buffer_size = (imagine_batch_size, imagine_batch_length, self.action_dim)
            else:
                action_buffer_size = scalar_size
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(action_buffer_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger,
                     sample_proprio=None):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)
        obs_hat_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context (uses real obs+proprio for the initial posterior)
        context_latent = self.encode_obs(sample_obs, sample_proprio)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

        if log_video:
            logger.log("Imagine/predict_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())

        return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def predict_next_grad(self, last_flattened_sample, action):
        """Differentiable variant of predict_next, used by imagine_data_grad.

        Two differences from predict_next:
          * termination is returned as a smooth sigmoid probability rather
            than a hard ``> 0`` boolean. This is required so the lambda
            return is differentiable.
          * obs_hat is never decoded (image_decoder isn't called in
            imagination; saves compute).
        Reward is decoded via SymLogTwoHotLoss.decode, which IS already
        smooth in dist_feat, so it carries gradient like dreamer's
        symexp-twohot reward head.
        """
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(
                last_flattened_sample, action,
            )
            prior_logits = self.dist_head.forward_prior(dist_feat)
            prior_sample = self.stright_throught_gradient(
                prior_logits, sample_mode="random_sample",
            )
            prior_flattened_sample = self.flatten_sample(prior_sample)
            reward_logits = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_logits)
            term_logits = self.termination_decoder(dist_feat)
            term_prob = torch.sigmoid(term_logits)
        return reward_hat, term_prob, prior_flattened_sample, dist_feat

    def imagine_data_grad(self, agent, sample_obs, sample_action,
                          imagine_batch_size, imagine_batch_length,
                          sample_proprio=None):
        """Pathwise-differentiable variant of imagine_data.

        Mirrors dreamer's _imagine (models.py): builds an imagined rollout
        where every operation is differentiable in the actor's parameters.
        Used by ActorCriticAgent.update_pathwise to backprop the lambda
        return all the way to the actor's weights, instead of REINFORCE's
        score-function estimator on detached imagination.

        Key differences from imagine_data:
          * No in-place buffer writes (those break autograd). The rollout
            is built as Python lists of per-step tensors, then concatenated.
          * Calls agent.sample_grad(feat.detach()), which uses Normal.rsample
            so the action carries gradient to actor params. The feat is
            detached on the way INTO the policy (matching dreamer:
            ``inp = feat.detach()``) so the actor only receives gradient
            from its action's downstream effect on future reward, not from
            how it conditions on the current state — keeping the actor's
            gradient separate from the world-model's representation gradient.
          * Termination is a smooth sigmoid probability (see predict_next_grad).

        Returns (feat, action, reward, term_prob) of shapes:
            feat:       (B, T+1, latent_dim + transformer_hidden_dim)
            action:     (B, T, action_dim)
            reward:     (B, T)
            term_prob:  (B, T)
        """
        # Reset KV cache. dtype must match what gets cached during the rollout
        # since the cache concatenates into a single tensor across steps.
        self.storm_transformer.reset_kv_cache_list(
            imagine_batch_size, dtype=self.tensor_dtype,
        )

        # Burn in the context using real observations. We only need the
        # final latent + dist_feat to seed imagination; intermediate values
        # are part of the world model's own forward, not the actor's graph.
        context_latent = self.encode_obs(sample_obs, sample_proprio)
        last_latent = None
        last_dist_feat = None
        for i in range(sample_obs.shape[1]):
            _, _, _, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=False,
            )

        latents = [last_latent]
        dist_feats = [last_dist_feat]
        actions = []
        rewards = []
        terms = []

        for _ in range(imagine_batch_length):
            cur_feat = torch.cat([latents[-1], dist_feats[-1]], dim=-1)
            # Detach the feat on the way INTO the policy (dreamer convention).
            # Gradient still flows from this step's ACTION through the world
            # model into next step's feat / reward — that's the path we want
            # to optimize.
            action = agent.sample_grad(cur_feat.detach())
            reward_hat, term_prob, next_latent, next_dist_feat = \
                self.predict_next_grad(latents[-1], action)

            actions.append(action)
            rewards.append(reward_hat)
            terms.append(term_prob)
            latents.append(next_latent)
            dist_feats.append(next_dist_feat)

        feat = torch.cat(
            [torch.cat(latents, dim=1), torch.cat(dist_feats, dim=1)],
            dim=-1,
        )
        action_seq = torch.cat(actions, dim=1)
        reward_seq = torch.cat(rewards, dim=1)
        term_seq = torch.cat(terms, dim=1)
        return feat, action_seq, reward_seq, term_seq

    def update(self, obs, action, reward, termination, logger=None, proprio=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            # encoding (image + optional proprio)
            embedding = self._encode_combined(obs, proprio)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample)
            if self.use_proprio:
                proprio_hat = self.proprio_decoder(flattened_sample)

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
            prior_logits = self.dist_head.forward_prior(dist_feat)
            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)
            if self.use_proprio:
                proprio_recon_loss = ((proprio_hat - proprio) ** 2).sum(dim=-1).mean()
            else:
                proprio_recon_loss = torch.tensor(0.0, device=obs.device)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            total_loss = reconstruction_loss + proprio_recon_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss

        if not torch.isfinite(total_loss):
            stats = {
                "recon": reconstruction_loss.item(), "reward": reward_loss.item(),
                "term": termination_loss.item(), "dyn": dynamics_loss.item(),
                "rep": representation_loss.item(),
                "post_logits_absmax": post_logits.abs().max().item(),
                "prior_logits_absmax": prior_logits.abs().max().item(),
                "obs_min_max": (obs.min().item(), obs.max().item()),
                "reward_min_max": (reward.min().item(), reward.max().item()),
            }
            raise RuntimeError(f"Non-finite world-model loss: {stats}")

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        if torch.isfinite(grad_norm):
            self.scaler.step(self.optimizer)
        elif logger is not None:
            logger.log("WorldModel/skipped_step", 1.0)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
            if self.use_proprio:
                logger.log("WorldModel/proprio_recon_loss", proprio_recon_loss.item())
            logger.log("WorldModel/reward_loss", reward_loss.item())
            logger.log("WorldModel/termination_loss", termination_loss.item())
            logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
            logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
            logger.log("WorldModel/representation_loss", representation_loss.item())
            logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
            logger.log("WorldModel/total_loss", total_loss.item())
            logger.log("WorldModel/grad_norm", grad_norm.item())
            logger.log("WorldModel/reward_batch_absmax", reward.abs().max().item())
