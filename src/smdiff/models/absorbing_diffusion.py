import math
import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from tqdm import tqdm
from smdiff.losses import MMDFMDRegularizer, resolve_loss_id
from .sampler import Sampler


class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id):
        super().__init__(H)
        self.seed = H.seed
        self.monotonicity_loss = H.monotonicity_loss
        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.total_steps

        self._denoise_fn = denoise_fn
        self.sampling_batch_size = H.sampling_batch_size
        self.loss_type = resolve_loss_id(getattr(H, 'loss_type', 'mmd_fmd_loss')).id
        self.mask_schedule = H.mask_schedule
        self.sample_schedule = H.sample_schedule
        self.register_buffer('mask_id', torch.tensor(mask_id))

        # Partial masking strategy
        self.masking_strategy = getattr(H, 'masking_strategy', None)
        self.mmd_fmd_cfg = getattr(H, 'mmd_fmd', {}) or {}
        self.strict_fmd_cfg = getattr(H, 'strict_fmd', {}) or {}
        self.sync_bar_ddpm_cfg = getattr(H, 'sync_bar_ddpm', {}) or {}
        self.hierarchical_masking = getattr(H, 'hierarchical_masking', {}) or {}
        self.loss_weights = getattr(H, "loss_weights", None)
        self._use_eos = bool(getattr(H, 'eos', False))
        self._codebook_size = tuple(int(v) for v in H.codebook_size)
        self.strict_fmd_transform_eps = float(self.strict_fmd_cfg.get("transform_eps", 1e-6))
        self.strict_fmd_memory_size = int(self.strict_fmd_cfg.get("memory_size", 2048))
        self.strict_fmd_enqueue_cap = int(self.strict_fmd_cfg.get("enqueue_cap", 0))
        self.sync_bar_use_uncertainty_bias = bool(self.sync_bar_ddpm_cfg.get("use_uncertainty_bias", False))
        self.sync_bar_uncertainty_alpha = float(self.sync_bar_ddpm_cfg.get("uncertainty_alpha", 1.0))
        self.sync_bar_uncertainty_ema = float(self.sync_bar_ddpm_cfg.get("uncertainty_ema", 0.9))
        self.sync_bar_min_full_bar_masks = int(self.sync_bar_ddpm_cfg.get("min_full_bar_masks", 1))
        self.sync_bar_min_block_size = max(1, int(self.sync_bar_ddpm_cfg.get("min_block_size", 1)))
        self.sync_bar_max_block_size = max(self.sync_bar_min_block_size, int(self.sync_bar_ddpm_cfg.get("max_block_size", 16)))
        self.sync_bar_challenge_center = float(self.sync_bar_ddpm_cfg.get("challenge_center", 0.5))
        self.sync_bar_challenge_width = max(1e-3, float(self.sync_bar_ddpm_cfg.get("challenge_width", 0.35)))

        # Optional differentiable regularization terms (used by mmd_fmd_loss and mmd_loss)
        self.mmd_fmd_regularizer = MMDFMDRegularizer(
            mmd_weight=float(self.mmd_fmd_cfg.get("mmd_weight", 0.01)),
            fmd_weight=float(self.mmd_fmd_cfg.get("fmd_weight", 0.005)),
            bandwidths=self.mmd_fmd_cfg.get("bandwidths", [0.5, 1.0, 2.0, 4.0]),
            max_samples=int(self.mmd_fmd_cfg.get("max_samples", 4096)),
            covariance_shrinkage_alpha=float(self.strict_fmd_cfg.get("covariance_shrinkage_alpha", 0.0)),
            eps=float(self.mmd_fmd_cfg.get("eps", 1e-6)),
        )

        feature_dim = len(self._codebook_size)
        self.register_buffer('strict_fmd_ref_mean', torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer('strict_fmd_ref_cov', torch.eye(feature_dim, dtype=torch.float32))
        self.register_buffer('strict_fmd_ref_ready', torch.tensor(0, dtype=torch.uint8))

        if self.strict_fmd_memory_size > 0:
            self.register_buffer(
                'strict_fmd_gen_memory',
                torch.zeros((self.strict_fmd_memory_size, feature_dim), dtype=torch.float32),
            )
        else:
            self.register_buffer('strict_fmd_gen_memory', torch.zeros((0, feature_dim), dtype=torch.float32))
        self.register_buffer('strict_fmd_gen_mem_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('strict_fmd_gen_mem_ptr', torch.tensor(0, dtype=torch.long))

        if self.loss_type == 'strict_fmd':
            ref_mean, ref_cov = self._compute_strict_fmd_reference_stats(getattr(H, 'dataset_path', None))
            self.strict_fmd_ref_mean.copy_(ref_mean)
            self.strict_fmd_ref_cov.copy_(ref_cov)
            self.strict_fmd_ref_ready.fill_(1)

        # Track loss at each time step for importance sampling
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))
        bar_vocab = int(self._codebook_size[0]) + (1 if self._use_eos else 0)
        self.register_buffer('sync_bar_ddpm_bar_uncertainty', torch.zeros(bar_vocab, dtype=torch.float32))
        self.register_buffer('sync_bar_ddpm_seen_count', torch.zeros(bar_vocab, dtype=torch.long))
        assert self.mask_schedule in ['random', 'fixed']

        self.task_queue = []

        # Set seed
        # torch.manual_seed(self.seed) # Handled globally in trainer

    def _reference_channel_denominators(self):
        if self._use_eos:
            denoms = [max(v, 1) for v in self._codebook_size]
        else:
            denoms = [max(v - 1, 1) for v in self._codebook_size]
        return np.array(denoms, dtype=np.float64)

    def _iter_raw_sequences(self, data_array):
        if data_array.ndim == 3:
            for i in range(data_array.shape[0]):
                yield data_array[i]
            return

        if data_array.ndim == 1:
            for item in data_array:
                yield item
            return

        raise ValueError(f"Unsupported dataset shape for strict_fmd: {data_array.shape}")

    def _prepare_reference_sequence(self, raw_item):
        x = raw_item

        if isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype == object:
            x = x.item()

        if isinstance(x, (str, np.str_)):
            x = np.load(x, allow_pickle=True)

        if not isinstance(x, np.ndarray) or x.dtype == object:
            x = np.array(x, dtype=np.int64)

        if x.ndim != 2:
            return None

        seq_len = self.shape[0]
        channels = self.shape[1]
        if x.shape[1] != channels:
            return None

        x = x.astype(np.int64, copy=False)
        length = x.shape[0]

        if length > seq_len:
            x = x[:seq_len]
        elif length < seq_len:
            x_new = np.full((seq_len, channels), -1, dtype=np.int64)
            x_new[:length] = x
            if self._use_eos and length < seq_len:
                eos_token = np.array(self._codebook_size, dtype=np.int64)
                x_new[length] = eos_token
            x = x_new

        return x

    def _compute_strict_fmd_reference_stats(self, dataset_path):
        if not dataset_path:
            raise ValueError("strict_fmd requires dataset_path to compute fixed reference statistics.")

        data_array = np.load(dataset_path, allow_pickle=True)
        denoms = self._reference_channel_denominators()
        channels = len(self._codebook_size)
        max_tokens = int(self.strict_fmd_cfg.get("reference_max_tokens", 0))
        eps = float(self.strict_fmd_cfg.get("reference_eps", 1e-6))

        sum_feat = np.zeros(channels, dtype=np.float64)
        sum_outer = np.zeros((channels, channels), dtype=np.float64)
        count = 0

        for raw_item in self._iter_raw_sequences(data_array):
            seq = self._prepare_reference_sequence(raw_item)
            if seq is None:
                continue

            token_valid = np.any(seq != -1, axis=-1)
            if not np.any(token_valid):
                continue

            feat = seq[token_valid].astype(np.float64) / denoms[None, :]
            if max_tokens > 0 and count + feat.shape[0] > max_tokens:
                keep = max_tokens - count
                if keep <= 0:
                    break
                feat = feat[:keep]

            count += feat.shape[0]
            sum_feat += feat.sum(axis=0)
            sum_outer += feat.T @ feat

            if max_tokens > 0 and count >= max_tokens:
                break

        if count < 2:
            raise ValueError(
                f"strict_fmd requires at least 2 valid feature rows for reference stats, got {count}."
            )

        mean = sum_feat / float(count)
        cov = (sum_outer - float(count) * np.outer(mean, mean)) / float(max(count - 1, 1))
        cov = cov + np.eye(channels, dtype=np.float64) * eps

        mean_t = torch.tensor(mean, dtype=torch.float32)
        cov_t = torch.tensor(cov, dtype=torch.float32)
        return mean_t, cov_t

    def _get_strict_fmd_memory(self):
        n = int(self.strict_fmd_gen_memory.shape[0])
        if n == 0:
            return self.strict_fmd_gen_memory

        count = int(self.strict_fmd_gen_mem_count.item())
        ptr = int(self.strict_fmd_gen_mem_ptr.item())
        filled = min(count, n)
        if filled == 0:
            return self.strict_fmd_gen_memory[:0]

        if count < n:
            return self.strict_fmd_gen_memory[:filled]

        if ptr == 0:
            return self.strict_fmd_gen_memory

        return torch.cat([
            self.strict_fmd_gen_memory[ptr:],
            self.strict_fmd_gen_memory[:ptr],
        ], dim=0)

    def _update_strict_fmd_memory(self, new_features: torch.Tensor, enqueue_cap: int = 0):
        n = int(self.strict_fmd_gen_memory.shape[0])
        if n == 0 or new_features.numel() == 0:
            return

        feats = new_features.detach().to(self.strict_fmd_gen_memory.device, dtype=torch.float32)
        # Cap per-step memory enqueue rows to reduce queue-stat churn.
        if enqueue_cap > 0 and feats.shape[0] > enqueue_cap:
            idx = torch.randperm(feats.shape[0], device=feats.device)[:enqueue_cap]
            feats = feats[idx]

        if feats.shape[0] >= n:
            feats = feats[-n:]

        k = int(feats.shape[0])
        ptr = int(self.strict_fmd_gen_mem_ptr.item())

        first = min(k, n - ptr)
        self.strict_fmd_gen_memory[ptr:ptr + first] = feats[:first]
        rem = k - first
        if rem > 0:
            self.strict_fmd_gen_memory[:rem] = feats[first:]

        self.strict_fmd_gen_mem_ptr.fill_((ptr + k) % n)
        self.strict_fmd_gen_mem_count.add_(k)

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
            # get its probability
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # Randomly set *tokens* to mask with probability t/T.
        #
        # For Octuple (or Trio/Melody), we interpret a "token" as one timestep in the
        # sequence, i.e. we mask all channels at that position together.
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        b, seq_len = x_t.shape[0], x_t.shape[1]
        device = x_t.device

        # mask positions (B, L) with prob t/T per sample
        time_prob = (t.float() / self.num_timesteps).view(-1, 1)
        mask_pos = torch.rand((b, seq_len), device=device) < time_prob

        # expand to all channels (B, L, C)
        mask = mask_pos.unsqueeze(-1).expand_as(x_t)

        for i in range(len(self.mask_id)):
            x_t[:, :, i][mask_pos] = self.mask_id[i]

        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def _get_hierarchical_config(self, num_channels):
        cfg = self.hierarchical_masking if isinstance(self.hierarchical_masking, dict) else {}

        priors = cfg.get("channel_priors")
        if not isinstance(priors, list) or len(priors) != num_channels:
            priors = [1.0] * num_channels

        return {
            "channel_priors": priors,
            "structure_bias": float(cfg.get("structure_bias", 1.5)),
            "content_bias": float(cfg.get("content_bias", 1.5)),
            "schedule_midpoint": float(cfg.get("schedule_midpoint", 0.5)),
            "schedule_steepness": float(cfg.get("schedule_steepness", 8.0)),
            "bar_locality": float(cfg.get("bar_locality", 0.2)),
            "eval_constrain_to_window": bool(cfg.get("eval_constrain_to_window", True)),
        }

    def _sync_bar_challenge(self, ratio: float) -> float:
        # Bell-shaped challenge score centered at challenge_center.
        dist = abs(ratio - self.sync_bar_challenge_center)
        challenge = max(0.0, 1.0 - dist / self.sync_bar_challenge_width)
        return min(challenge, 1.0)

    def _sync_bar_pick_block_size(self, ratio: float, n_bars: int) -> int:
        max_block = min(self.sync_bar_max_block_size, max(1, n_bars))
        candidates = [c for c in [1, 2, 4, 8, 16] if self.sync_bar_min_block_size <= c <= max_block]
        if not candidates:
            return 1
        if len(candidates) == 1:
            return candidates[0]

        challenge = self._sync_bar_challenge(ratio)
        weights = []
        for c in candidates:
            if challenge >= 0.6 and c == max(candidates):
                w = 4.0
            elif challenge <= 0.2 and c == min(candidates):
                w = 4.0
            else:
                target = min(candidates) + challenge * (max(candidates) - min(candidates))
                w = math.exp(-abs(math.log2(c) - math.log2(max(target, 1.0))))
            weights.append(w)

        w_tensor = torch.tensor(weights, dtype=torch.float)
        idx = torch.multinomial(w_tensor, num_samples=1, replacement=True).item()
        return int(candidates[idx])

    def _sync_bar_bar_weights(self, start_bar: int, n_bars: int, device: torch.device) -> torch.Tensor:
        if not self.sync_bar_use_uncertainty_bias:
            return torch.ones(n_bars, device=device, dtype=torch.float)

        bar_ids = torch.arange(start_bar, start_bar + n_bars, device=device, dtype=torch.long)
        unc_size = int(self.sync_bar_ddpm_bar_uncertainty.shape[0])
        in_range = (bar_ids >= 0) & (bar_ids < unc_size)

        weights = torch.ones(n_bars, device=device, dtype=torch.float)
        if in_range.any():
            vals = self.sync_bar_ddpm_bar_uncertainty[bar_ids[in_range]].float()
            if vals.numel() > 0:
                vals = vals - vals.min()
                denom = vals.max().clamp(min=1e-6)
                vals = vals / denom
                weights[in_range] = 1.0 + self.sync_bar_uncertainty_alpha * vals
        return torch.clamp(weights, min=1e-6)

    @torch.no_grad()
    def _update_sync_bar_uncertainty(self, x_0_hat_logits, x_0_ignore, x_0):
        if self.masking_strategy != 'sync_bar_ddpm' or not self.sync_bar_use_uncertainty_bias:
            return

        bar_logits = x_0_hat_logits[0]  # (B, V, L)
        probs = F.softmax(bar_logits.float(), dim=1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)  # (B, L)

        valid_mask = (x_0_ignore[:, :, 0] != -1)
        bar_tokens = x_0[:, :, 0].long()
        unc_size = int(self.sync_bar_ddpm_bar_uncertainty.shape[0])

        for i in range(bar_tokens.shape[0]):
            valid = valid_mask[i]
            if not valid.any():
                continue
            bars_i = bar_tokens[i][valid]
            ent_i = entropy[i][valid]
            uniq_bars = torch.unique(bars_i)
            for bval in uniq_bars:
                b_idx = int(bval.item())
                if b_idx < 0 or b_idx >= unc_size:
                    continue
                b_mask = (bars_i == bval)
                if not b_mask.any():
                    continue
                b_ent = ent_i[b_mask].mean()
                old = self.sync_bar_ddpm_bar_uncertainty[b_idx]
                updated = self.sync_bar_uncertainty_ema * old + (1.0 - self.sync_bar_uncertainty_ema) * b_ent
                self.sync_bar_ddpm_bar_uncertainty[b_idx] = updated
                self.sync_bar_ddpm_seen_count[b_idx] += 1

    def _hierarchical_schedule(self, ratio, midpoint, steepness):
        return torch.sigmoid((ratio - midpoint) * steepness)

    def build_hierarchical_mask(self, x_0, t, window_start=None, window_end=None, preserve_structure=False):
        """Build a continuous prior-guided hierarchical mask over (bar, channel) units."""
        b, seq_len, num_channels = x_0.shape
        device = x_0.device

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device, dtype=torch.long)
        t = t.to(device)
        if t.dim() == 0:
            t = t.expand(b)
        elif t.size(0) != b:
            raise ValueError(f"Expected t to have batch size {b}, got {t.size(0)}")

        cfg = self._get_hierarchical_config(num_channels)
        priors = torch.tensor(cfg["channel_priors"], device=device, dtype=torch.float)

        bar_indices = x_0[:, :, 0]
        mask = torch.zeros_like(x_0, dtype=torch.bool, device=device)

        for i in range(b):
            sample_bars = torch.unique(bar_indices[i])
            n_bars = int(sample_bars.numel())
            if n_bars == 0:
                continue

            ratio = torch.clamp(t[i].float() / self.num_timesteps, 0.0, 1.0)
            total_units = n_bars * num_channels
            k = int(torch.round(total_units * ratio).item())
            if k <= 0:
                continue

            schedule = self._hierarchical_schedule(ratio, cfg["schedule_midpoint"], cfg["schedule_steepness"])

            channel_weights = priors.clone()
            n_struct = min(2, num_channels)
            channel_weights[:n_struct] *= (1.0 + (1.0 - schedule) * cfg["structure_bias"])
            if num_channels > n_struct:
                channel_weights[n_struct:] *= (1.0 + schedule * cfg["content_bias"])

            locality = max(0.0, min(1.0, cfg["bar_locality"]))
            bar_weights = torch.ones(n_bars, device=device, dtype=torch.float)
            if locality > 0.0 and n_bars > 1:
                center_idx = torch.randint(0, n_bars, (1,), device=device).item()
                rel_pos = torch.arange(n_bars, device=device, dtype=torch.float)
                dist = torch.abs(rel_pos - center_idx) / max(1.0, float(n_bars - 1))
                sigma = max(0.1, 1.0 - 0.8 * locality)
                gauss = torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))
                bar_weights = (1.0 - locality) + locality * gauss

            unit_weights = torch.outer(bar_weights, channel_weights).reshape(-1)
            if torch.all(unit_weights <= 0):
                unit_weights = torch.ones_like(unit_weights)

            k = min(k, int(unit_weights.numel()))
            selected = torch.multinomial(unit_weights, num_samples=k, replacement=False)
            selected_bar_idx = selected // num_channels
            selected_channel_idx = selected % num_channels
            selected_bars = sample_bars[selected_bar_idx]

            for bar_val, att_idx in zip(selected_bars.tolist(), selected_channel_idx.tolist()):
                pos_mask = (bar_indices[i] == bar_val)
                mask[i, pos_mask, att_idx] = True

        if window_start is not None or window_end is not None:
            start = 0 if window_start is None else max(0, int(window_start))
            end = seq_len if window_end is None else min(seq_len, int(window_end))
            window_mask = torch.zeros((b, seq_len), dtype=torch.bool, device=device)
            if end > start:
                window_mask[:, start:end] = True
            mask = mask & window_mask.unsqueeze(-1)

        if preserve_structure and num_channels >= 2:
            mask[:, :, 0] = False
            mask[:, :, 1] = False

        return mask

    def q_sample_partial(self, x_0, t):
        """
        Implementation of partial masking strategies for Octuple MIDI.
        x_0 shape: (Batch, SeqLen, 8)
        """
        x_t, x_0_ignore = x_0.clone(), x_0.clone()
        b, seq_len, _ = x_0.shape
        device = x_0.device

        # Initialize mask with False
        mask = torch.zeros_like(x_t, dtype=torch.bool, device=device)
        
        # Time-based probability of masking: t/T
        time_prob = t.float() / self.num_timesteps # (Batch,)
        
        # Strategy Implementation
        current_strategy = self.masking_strategy

        if current_strategy == 'mixed':
            # Randomly select a strategy from the pool
            strategies = [
                'bar_all',
                'bar_attribute', 
                'random'
            ]
            # Select one strategy for the entire batch (simplest implementation)
            current_strategy = strategies[np.random.randint(len(strategies))]
        
        # If 'random' is selected (either explicitly or via mixed), use standard q_sample
        # (token-level/octuple masking).
        if current_strategy == 'random':
            return self.q_sample(x_0=x_0, t=t)

        if current_strategy == 'hierarchical':
            mask = self.build_hierarchical_mask(x_0=x_0, t=t)

            for i in range(len(self.mask_id)):
                x_t[:, :, i][mask[:, :, i]] = self.mask_id[i]

            x_0_ignore[torch.bitwise_not(mask)] = -1
            return x_t, x_0_ignore, mask

        if current_strategy in ['sync_bar', 'sync_bar_position', 'sync_bar_ddpm']:
            # Implementation of "Synchronized Masking" for learning sequentiality.
            # Optimized with lookup table.
            
            bar_indices = x_0[:, :, 0] # (B, Time)
            
            # CONFIG: Decide which channels are Block vs Unit masked
            if current_strategy == 'sync_bar_position':
                block_channels = [0, 1]                       # Mask Bar & Pos in blocks
                target_attributes_inner = torch.arange(2, 8, device=device) # Rest are units
                adaptive_ddpm = False
            elif current_strategy == 'sync_bar_ddpm':
                block_channels = [0]                          # DDPM sync_bar variant masks bar channel in adaptive blocks
                target_attributes_inner = torch.arange(1, 8, device=device) # Rest are 1-bar units
                adaptive_ddpm = True
            else:
                block_channels = [0]                          # Sync Bar default
                target_attributes_inner = torch.arange(1, 8, device=device) # Rest are units
                adaptive_ddpm = False

            num_attrs_inner = len(target_attributes_inner)
            
            BAR_BLOCK_SIZE = 16     # Legacy fixed block size for sync_bar variants

            # 1. Pre-calculate Lookup Table [Batch, MaxBar+1, Channels]
            max_bar = bar_indices.max().item()
            mask_lookup = torch.zeros((b, max_bar + 1, 8), dtype=torch.bool, device=device)

            # --- OPTIMIZATION START ---
            # Pre-calculate counts and targets for the whole batch
            min_bars = bar_indices.amin(dim=1)
            max_bars = bar_indices.amax(dim=1)
            n_bars_batch = (max_bars - min_bars + 1) # Shape: (B,)
            ratios = t.float() / self.num_timesteps

            # Vectorized Target Calculation for Block Channels
            # Shape (B, num_block_channels)
            target_vals_blocks = (n_bars_batch.float() * ratios).unsqueeze(1).repeat(1, len(block_channels))
            targets_blocks = target_vals_blocks.floor().long()
            targets_blocks += torch.bernoulli(target_vals_blocks - targets_blocks.float()).long()

            # Vectorized Target Calculation for Other Channels
            total_units_inner = n_bars_batch * num_attrs_inner
            target_vals_units = total_units_inner.float() * ratios
            targets_units = target_vals_units.floor().long()
            targets_units += torch.bernoulli(target_vals_units - targets_units.float()).long()

            for i in range(b):
                # --- A. Optimized Block Masking (Channels in block_channels) ---
                nb = n_bars_batch[i].item()
                start_bar = min_bars[i].item() 
                ratio_i = float(ratios[i].item())
                block_size_i = self._sync_bar_pick_block_size(ratio_i, nb) if adaptive_ddpm else BAR_BLOCK_SIZE
                num_blocks = math.ceil(nb / max(1, block_size_i))
                bar_weights = self._sync_bar_bar_weights(start_bar=start_bar, n_bars=nb, device=device) if adaptive_ddpm else None
                
                for idx, ch in enumerate(block_channels):
                    tgt = targets_blocks[i, idx].item()
                    if adaptive_ddpm:
                        tgt = max(tgt, self.sync_bar_min_full_bar_masks)
                    if tgt > 0:
                        if adaptive_ddpm and bar_weights is not None:
                            block_scores = torch.zeros(num_blocks, device=device, dtype=torch.float)
                            for b_idx in range(num_blocks):
                                rel_start = b_idx * block_size_i
                                rel_end = min(rel_start + block_size_i, nb)
                                block_scores[b_idx] = bar_weights[rel_start:rel_end].mean()
                            if torch.all(block_scores <= 0):
                                block_indices = torch.randperm(num_blocks, device=device).tolist()
                            else:
                                block_indices = torch.multinomial(block_scores, num_samples=num_blocks, replacement=False).tolist()
                        else:
                            # Randomly visit blocks to fill quota
                            block_indices = torch.randperm(num_blocks, device=device).tolist()
                        
                        needed = tgt
                        for b_idx in block_indices:
                            if needed <= 0: break
                            
                            # Relative indices (0..N)
                            rel_start = b_idx * block_size_i
                            rel_end = min(rel_start + block_size_i, nb)
                            block_len = rel_end - rel_start
                            
                            # Absolute indices for lookup table
                            abs_start = start_bar + rel_start
                            
                            if block_len <= needed:
                                # Take whole block
                                mask_lookup[i, abs_start : abs_start + block_len, ch] = True
                                needed -= block_len
                            else:
                                # Take partial block (random slice) to fill remainder
                                off = torch.randint(0, block_len - needed + 1, (1,), device=device).item()
                                slice_start = abs_start + off
                                mask_lookup[i, slice_start : slice_start + needed, ch] = True
                                needed = 0

                # --- B. Optimized Unit Masking (Channels 1..7 or 2..7) ---
                k_units = targets_units[i].item()
                if k_units > 0:
                    nb = n_bars_batch[i].item()
                    start_bar = min_bars[i].item()
                    total_units = nb * num_attrs_inner
                    
                    if adaptive_ddpm:
                        bar_weights = self._sync_bar_bar_weights(start_bar=start_bar, n_bars=nb, device=device)
                        unit_weights = bar_weights.repeat_interleave(num_attrs_inner)
                        n_samples = min(k_units, int(unit_weights.numel()))
                        perm = torch.multinomial(unit_weights, num_samples=n_samples, replacement=False)
                    else:
                        # Sample k_units indices
                        perm = torch.randperm(total_units, device=device)[:k_units]
                    
                    # Decode indices to (Bar, Attribute)
                    rel_bar_indices = perm.div(num_attrs_inner, rounding_mode='floor')
                    attr_indices_idx = perm % num_attrs_inner
                    
                    # Convert to Absolute Bar Index
                    abs_bar_indices = rel_bar_indices + start_bar
                    actual_attrs = target_attributes_inner[attr_indices_idx]
                    
                    # Vectorized update
                    mask_lookup[i, abs_bar_indices, actual_attrs] = True
            # --- OPTIMIZATION END ---

            # 2. Vectorized Application
            # Gather mask from lookup table based on Bar Token values in sequence
            # batch_ids: (B, L)
            batch_ids = torch.arange(b, device=device)[:, None].expand_as(bar_indices)
            mask = mask_lookup[batch_ids, bar_indices]

            # Apply per-channel mask token
            for i in range(len(self.mask_id)):
                x_t[:, :, i][mask[:, :, i]] = self.mask_id[i]
            
            x_0_ignore[torch.bitwise_not(mask)] = -1
            return x_t, x_0_ignore, mask

        if current_strategy == 'bar_all':
            # Time-Dependent Bar Count
            # Select K bars where K ~ t/T * TotalBars
            # Mask ALL attributes for those bars.
            
            bar_indices = x_0[:, :, 0]
            
            for i in range(b):
                u_bars = torch.unique(bar_indices[i])
                n_bars = len(u_bars)
                
                # t[i] is 1..T
                ratio = t[i].float() / self.num_timesteps
                k = torch.round(n_bars * ratio).long().item()
                
                if k > 0:
                    perm = torch.randperm(n_bars, device=device)
                    selected_bars = u_bars[perm[:k]]
                    
                    # mask[i] = (bar_indices[i] in selected_bars)
                    sample_mask = torch.isin(bar_indices[i], selected_bars)
                    mask[i, :, :] = sample_mask.unsqueeze(-1).expand(-1, 8)
            
            # Apply per-channel mask token
            for i in range(len(self.mask_id)):
                x_t[:, :, i][mask[:, :, i]] = self.mask_id[i]
            
            x_0_ignore[torch.bitwise_not(mask)] = -1
            return x_t, x_0_ignore, mask

        if current_strategy == 'bar_attribute':
            # Select K (Bar, Attribute) pairs where K ~ t/T * TotalUnits
            # Mask specific attributes in specific bars.
            
            bar_indices = x_0[:, :, 0]
            # target all attributes
            target_attributes = torch.arange(0, 8, device=device)
            num_attrs = len(target_attributes)
            
            for i in range(b):
                u_bars = torch.unique(bar_indices[i])
                n_bars = len(u_bars)
                total_units = n_bars * num_attrs
                
                # t[i] is 1..T
                ratio = t[i].float() / self.num_timesteps
                k = torch.round(total_units * ratio).long().item()
                
                if k > 0:
                     # Sample k units from total_units
                     perm = torch.randperm(total_units, device=device)[:k]
                     
                     # Map back to (bar_index_idx, attr_index_idx)
                     sel_bar_indices = perm // num_attrs
                     sel_attr_indices = perm % num_attrs
                     
                     # For each selected bar, gather which attributes to mask
                     # To avoid loop over k, loop over unique bars present in selection
                     unique_sel_bar_indices = torch.unique(sel_bar_indices)
                     
                     for bar_idx_idx in unique_sel_bar_indices:
                         # Get actual bar value
                         bar_val = u_bars[bar_idx_idx]
                         
                         # Which attributes for this bar?
                         # Indices in 'perm' where bar is this one
                         current_bar_match = (sel_bar_indices == bar_idx_idx)
                         attrs_to_mask_indices = sel_attr_indices[current_bar_match]
                         actual_attrs = target_attributes[attrs_to_mask_indices]
                         
                         # Apply to mask
                         # Find positions of this bar
                         pos_mask = (bar_indices[i] == bar_val) # (SeqLen,)
                         
                         for att in actual_attrs:
                             mask[i, pos_mask, att] = True

            # Apply per-channel mask token
            for i in range(len(self.mask_id)):
                x_t[:, :, i][mask[:, :, i]] = self.mask_id[i]
            
            x_0_ignore[torch.bitwise_not(mask)] = -1
            return x_t, x_0_ignore, mask
            
        else:
             # Fallback or error usage
             raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

    def _train_loss(self, x_0):
        x_0 = x_0.long()
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, 'uniform')

        # make x noisy and denoise
        if self.masking_strategy is not None:
             x_t, x_0_ignore, mask = self.q_sample_partial(x_0=x_0, t=t)
        elif self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)
            
        # The model cannot take -1 as an input index. We temporarily swap it to 0.
        # The attention mask (if used) or the loss mask will handle the rest.
        x_t_input = x_t.clone()
        x_t_input[x_t == -1] = 0

        # sample p(x_0 | x_t)
        raw_logits = self._denoise_fn(x_t_input, t=t)
        if self.training and self.masking_strategy == 'sync_bar_ddpm' and self.sync_bar_use_uncertainty_bias:
            self._update_sync_bar_uncertainty(raw_logits, x_0_ignore, x_0)
        x_0_hat_logits = [el.permute(0, 2, 1) for el in raw_logits]
        
        cross_entropy_loss_per_channel = [F.cross_entropy(x, x_0_ignore[:, :, i], ignore_index=-1, reduction='none').sum(1)
                              for i, x in enumerate(x_0_hat_logits)]
        cross_entropy_loss_stack = torch.stack(cross_entropy_loss_per_channel)

        # --- ENHANCEMENT: Channel Weighting Option ---
        # Detect Octuple encoding (8 channels)
        is_octuple = (len(x_0_hat_logits) == 8)
        
        if is_octuple:
            if self.loss_weights is not None:
                weights = torch.tensor(self.loss_weights, device=device)
            else:
                weights = torch.tensor([1.0] * 8, device=device)
            
            if len(weights) != 8:
                raise ValueError(f"Expected 8 weights, got {len(weights)} instead!")

            weighted_ce = cross_entropy_loss_stack * weights.unsqueeze(1)
            cross_entropy_loss = weighted_ce.sum(0)
        
        else:
            cross_entropy_loss = cross_entropy_loss_stack.sum(0)

        # Differentiable MMD + Frechet-style regularization (weak by default)
        mmd_loss = torch.tensor(0.0, device=device)
        fmd_loss = torch.tensor(0.0, device=device)
        reg_total = torch.tensor(0.0, device=device)
        strict_fmd_loss = torch.tensor(0.0, device=device)
        if self.loss_type in ['mmd_fmd_loss', 'mmd_loss']:
            valid_mask = (x_0_ignore != -1)
            reg_total, mmd_loss, fmd_loss = self.mmd_fmd_regularizer(
                x_0_hat_logits,
                x_0,
                valid_mask,
            )
        elif self.loss_type == 'strict_fmd':
            if int(self.strict_fmd_ref_ready.item()) == 0:
                raise RuntimeError("strict_fmd reference statistics are not initialized.")
            valid_mask = (x_0_ignore != -1)
            memory_features = self._get_strict_fmd_memory()
            strict_fmd_loss, current_pred_features = self.mmd_fmd_regularizer.strict_frechet_to_reference_with_memory(
                x_0_hat_logits,
                valid_mask,
                self.strict_fmd_ref_mean,
                self.strict_fmd_ref_cov,
                memory_features=memory_features,
            )
            # Keep strict_fmd memory fixed during eval so validation does not shift train dynamics.
            if self.training:
                self._update_strict_fmd_memory(current_pred_features, enqueue_cap=self.strict_fmd_enqueue_cap)
            fmd_loss = strict_fmd_loss

        # --- Test: Structure Regression Loss (Bar & Position) ---
        aux_loss = 0.0
        if is_octuple and self.monotonicity_loss:
            SCALE_FACTOR = 200.0
            
            # 1. PREPARE TARGETS (Ground Truth)
            # ERROR WAS HERE: x_0 is (Batch, Time, Channels)
            # We want all time steps (dim 1), specific channel (dim 2)
            target_bar = x_0[:, :, 0].float()  # Shape: (B, T)
            target_pos = x_0[:, :, 1].float()  # Shape: (B, T)
            
            target_global = target_bar * SCALE_FACTOR + target_pos

            bar_logits = x_0_hat_logits[0] 
            pos_logits = x_0_hat_logits[1] 
            
            # Expected Bar
            probs_bar = F.softmax(bar_logits, dim=1)
            indices_bar = torch.arange(probs_bar.shape[1], device=device).float().view(1, -1, 1)
            expected_bar = (probs_bar * indices_bar).sum(1) # Sum over Vocab dim -> (B, T)
            
            # Expected Position
            probs_pos = F.softmax(pos_logits, dim=1)
            indices_pos = torch.arange(probs_pos.shape[1], device=device).float().view(1, -1, 1)
            expected_pos = (probs_pos * indices_pos).sum(1)
            
            expected_global = expected_bar * SCALE_FACTOR + expected_pos

            mse_loss = F.mse_loss(expected_global, target_global, reduction='none')
            
            # Correct mask slicing as well
            valid_mask = (x_0_ignore[:, :, 0] != -1).float()
            
            structure_loss = (mse_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
            aux_loss = structure_loss * 1e-4

        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        if self.loss_type == 'elbo':
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'mmd_loss':
            # Pure MMD objective: optimize raw MMD only, without CE/FMD contributions.
            loss = mmd_loss
        elif self.loss_type == 'strict_fmd':
            # Pure strict Fréchet objective against fixed training-data reference stats.
            loss = torch.sqrt(strict_fmd_loss + self.strict_fmd_transform_eps)
        elif self.loss_type in ['plain_ce_loss', 'reweighted_elbo', 'mmd_fmd_loss']:
            # Fix: Use (T+1) to ensure weight is never exactly 0 at t=T
            weight = (1 - (t / (self.num_timesteps + 1)))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        # Add auxiliary structure losses for CE/ELBO-style objectives only.
        if self.loss_type not in ['mmd_loss', 'strict_fmd']:
            loss = loss + aux_loss
        if self.loss_type == 'mmd_fmd_loss':
            loss = loss + reg_total

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (0.1 * loss + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)

        self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach().to(self.loss_history.dtype)
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).to(self.loss_history.dtype))

        return loss.mean(), vb_loss.mean(), mmd_loss, fmd_loss

    def sample(self, temp=1.0, sample_steps=None, x_T=None, B=None, progress_handler=None):
        b, device = self.sampling_batch_size, 'cuda'
        if B is not None:
            b = B
        if x_T is None:
            x_T = torch.ones((b, *self.shape), device=device).long() * self.mask_id
        b = x_T.shape[0]
        unmasked = torch.zeros_like(x_T, device=device, dtype=torch.bool)
        unmasked[x_T != self.mask_id] = True

        if sample_steps:
            sample_steps = min(sample_steps, (~unmasked).sum())

        sample_steps = list(range(1, sample_steps + 1))
        last_progress = 0

        for t in reversed(sample_steps):

            p = int(100 * (len(sample_steps) - t) / len(sample_steps))
            if progress_handler and p > last_progress:
                last_progress = p
                progress_handler(p)

            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_T.shape, device=device) < 1 / t.float().view(-1, *((1,) * (len(x_T.shape) - 1)))
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits = self._denoise_fn(x_T, t=t)
            # scale by temperature
            x_0_logits = [x / temp for x in x_0_logits]
            x_0_dist = [dists.Categorical(
                logits=x) for x in x_0_logits]
            x_0_hat = torch.stack([xd.sample().long() for xd in x_0_dist], -1)
            x_T[changes] = x_0_hat[changes]

        if progress_handler:
            progress_handler(100)

        return x_T

    def queue_sample_task(self, progress_handler, finished_handler, sample_steps=None, x_T=None, b=1):
        device = 'cuda'

        if x_T is None:
            x_T = torch.ones((b, *self.shape), device=device).long() * self.mask_id

        unmasked = torch.zeros_like(x_T, device=device, dtype=torch.bool)
        unmasked[x_T != self.mask_id] = True

        if sample_steps:
            sample_steps = min(sample_steps, (~unmasked).sum())

        sample_steps = list(range(1, sample_steps + 1))

        for tensor in x_T:
            self.task_queue.append((tensor, sample_steps, progress_handler, finished_handler))

    def sample_worker(self, temp=1.0):
        device = 'cuda'
        last_progress = 0

        x_T = torch.ones((0, *self.shape), device=device).long() * self.mask_id

        while True:

            while x_T.shape[0] < 1:

                x_T = torch.stack((x_T, ))

            for t in reversed(sample_steps):

                p = int(100 * (len(sample_steps) - t) / len(sample_steps))
                if progress_handler and p > last_progress:
                    last_progress = p
                    progress_handler(p)

                print(f'Sample timestep {t:4d}', end='\r')
                t = torch.full((b,), t, device=device, dtype=torch.long)

                # where to unmask
                changes = torch.rand(x_T.shape, device=device) < 1 / t.float().view(-1, *((1,) * (len(x_T.shape) - 1)))
                # don't unmask somewhere already unmasked
                changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # update mask with changes
                unmasked = torch.bitwise_or(unmasked, changes)

                x_0_logits = self._denoise_fn(x_T, t=t)
                # scale by temperature
                x_0_logits = [x / temp for x in x_0_logits]
                x_0_dist = [dists.Categorical(
                    logits=x) for x in x_0_logits]
                x_0_hat = torch.stack([xd.sample().long() for xd in x_0_dist], -1)
                x_T[changes] = x_0_hat[changes]

        return x_T

    def guided_sample(self, guide, eta=1, temp=1.0, sample_steps=None, x_T=None, B=None):
        b, device = self.sampling_batch_size, 'cuda'
        if B is not None:
            b = B
        if x_T is None:
            x_T = torch.ones((b, *self.shape), device=device).long() * self.mask_id
        b = x_T.shape[0]


        unmasked = torch.zeros_like(x_T, device=device, dtype=torch.bool)
        unmasked[x_T != self.mask_id] = True

        if sample_steps:
            sample_steps = min(sample_steps, (~unmasked).sum())

        sample_steps = list(range(1, sample_steps + 1))

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                x_0_logits = self._denoise_fn(x_T, t=t_tensor)
            # scale by temperature
            x_0_logits = [x / temp for x in x_0_logits]
            x_0_probs = [F.softmax(x, -1) for x in x_0_logits]

            n = min(2, len(x_0_probs))

            for i in range(n): x_0_probs[i].requires_grad = True

            guide_loss = [guide(x_0_probs[i]) for i in range(n)]
            #print(guide_loss[0].item())
            #guide_loss = [g.log() for g in guide_loss] todo: logits require log?

            for i in range(n):
                guide_loss[i].mean().backward()
                #print("loss:", guide_loss[i].mean().item())

            #grads = torch.stack([x_0_p.grad.data[0, i, x_0_hat[0, i, 0]] for i in range(x_0_hat.shape[1])])#x_0_p.grad.data[0].gather(-1, x_0_hat[:, :, 0])#todo: repair for batches

            grad = [x_0_probs[i].grad.data for i in range(n)]
            x_0_probs = [F.softmax(x_0_logits[i], -1) for i in range(n)]

            for i in range(n):
                x_0_probs[i] -= grad[i] * eta #/(grad[i].max() - grad[i].min())
                x_0_probs[i] = x_0_probs[i].clamp(0, 1)

            x_0_dist = [dists.Categorical(probs=p) for p in x_0_probs]

            x_0_hat = torch.stack([xd.sample().long() for xd in x_0_dist], -1)

            # where to unmask
            t = torch.full((b,), t, device=device, dtype=torch.long)
            unmask_rand = torch.rand(x_T.shape, device=device)
            #unmask_rand[:, :, 0] = grads
            changes = unmask_rand < 1 / t.float().view(-1, *((1,) * (len(x_T.shape) - 1)))


            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_T[changes] = x_0_hat[changes]

        return x_T

    def sample_mlm(self, temp=1.0, sample_steps=None):
        b, device = self.sampling_batch_size, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        sample_steps = np.linspace(1, self.num_timesteps, num=sample_steps).astype(np.long)

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]

        return x_0

    @torch.no_grad()
    def elbo(self, x_0):
        b, device = x_0.size(0), x_0.device
        elbo = 0.0
        for t in reversed(list(range(1, self.num_timesteps+1))):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, x_0_ignore, _ = self.q_sample(x_0=x_0, t=t)
            x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0, 2, 1)
            cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1)
            elbo += cross_entropy_loss / t
        return elbo

    def train_iter(self, x):
        loss, vb_loss, mmd_loss, fmd_loss = self._train_loss(x)
        stats = {'loss': loss, 'vb_loss': vb_loss, 'mmd_loss': mmd_loss, 'fmd_loss': fmd_loss}
        return stats

    @torch.no_grad()
    def sample_shape(self, shape, num_samples, time_steps=1000, step=1, temp=0.8):
        device = 'cuda'
        x_t = torch.ones((num_samples,) + shape, device=device).long() * self.mask_id
        x_lim = shape[0] - self.shape[0]

        unmasked = torch.zeros_like(x_t, device=device).bool()

        autoregressive_step = 0
        for t in tqdm(list(reversed(list(range(1, time_steps+1))))):
            t = torch.full((num_samples,), t, device='cuda', dtype=torch.long)

            unmasking_method = 'random'
            if unmasking_method == 'random':
                # where to unmask
                changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1).unsqueeze(-1)
                # don't unmask somewhere already unmasked
                changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
                # update mask with changes
                unmasked = torch.bitwise_or(unmasked, changes)
            elif unmasking_method == 'autoregressive':
                changes = torch.zeros(x_t.shape, device=device).bool()
                index = (int(autoregressive_step / shape[1]), autoregressive_step % shape[1])
                changes[:, index[0], index[1]] = True
                unmasked = torch.bitwise_or(unmasked, changes)
                autoregressive_step += 1

            # keep track of PoE probabilities
            x_0_probs = torch.zeros((num_samples,) + shape + self.codebook_size, device='cuda')
            # keep track of counts
            count = torch.zeros((num_samples,) + shape, device='cuda')

            # TODO: Monte carlo approximate this instead
            for i in range(0, x_lim+1, step):
                # collect local noisy area
                x_t_part = x_t[:, i:i+self.shape[0]]

                # increment count
                count[:, i:i+self.shape[0]] += 1.0

                # flatten
                #x_t_part = x_t_part.reshape(x_t_part.size(0), -1)

                # denoise
                x_0_logits_part = self._denoise_fn(x_t_part, t=t)

                # unflatten
                #x_0_logits_part = x_0_logits_part.reshape(x_t_part.size(0), self.shape[1], -1)

                # multiply probabilities
                # for mixture
                x_0_probs[:, i:i+self.shape[0], 0] += torch.softmax(x_0_logits_part[0], dim=-1)#todo: [0] list trio

            # Mixture with Temperature
            x_0_probs = x_0_probs / x_0_probs.sum(-1, keepdim=True)
            C = torch.tensor(x_0_probs.size(-1)).float()
            x_0_probs = torch.softmax((torch.log(x_0_probs) + torch.log(C)) / temp, dim=-1)

            x_0_dist = dists.Categorical(probs=x_0_probs)
            x_0_hat = x_0_dist.sample().long()

            # update x_0 where anything has been masked
            x_t[changes] = x_0_hat[changes]

        return x_t.cpu().numpy()
