import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Block

class OctupleMDLM(nn.Module):
    """
    Masked Discrete Diffusion Language Model (MDLM) specifically designed 
    for Octuple format based on Kaliakatsos-Papakostas et al. 2025.
    
    The architecture introduces stage-awareness (timestep condition t) 
    that is concatenated to the standard combination of tokens and positional embeddings,
    before being projected back to the main transformer hidden dimension.
    """

    def __init__(self, H):
        super().__init__()

        self.n_embd = H.bert_n_emb
        self.block_size = H.block_size
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size
        self.causal = H.sampler == 'autoregressive'

        use_eos = getattr(H, 'eos', False)
        if self.causal:
            # Autoregressive: no mask token, no EOS expansion
            emb_vocab_size = list(H.codebook_size)
            head_out_size = list(H.codebook_size)
        elif use_eos:
            # EOS occupies codebook_size[c]; mask_id is codebook_size[c]+1
            emb_vocab_size = [h + 2 for h in H.codebook_size]
            head_out_size = [h + 1 for h in H.codebook_size]
        else:
            emb_vocab_size = [h + 1 for h in H.codebook_size]
            head_out_size = list(H.codebook_size)
        self.vocab_size = emb_vocab_size

        # Token embeddings for each channel
        self.tok_emb = nn.ModuleList([nn.Embedding(vs, self.n_embd) for vs in emb_vocab_size])
        
        # Dimension reduction after concatenation of channel embeddings
        emb_in_dim = self.n_embd * len(self.codebook_size)
        self.emb_red = nn.Linear(emb_in_dim,  self.n_embd)
        
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        
        # Stage Embedding (E_s) - maps unmasking stage/timestep 't' to an embedding vector
        # Usually total_steps is T. We add 1 for safety.
        num_timesteps = getattr(H, 'total_steps', 1000) + 1
        self.num_timesteps = max(1, int(getattr(H, 'total_steps', 1000)))
        self.stage_emb = nn.Embedding(num_timesteps, self.n_embd)

        mdlm_cfg = getattr(H, 'octuple_mdlm', {}) or {}
        if not isinstance(mdlm_cfg, dict):
            mdlm_cfg = {}
        self.use_dual_stage_embeddings = bool(mdlm_cfg.get('use_dual_stage_embeddings', False))
        self.use_curriculum_weight = bool(mdlm_cfg.get('use_curriculum_weight', True))
        self.progress_buckets = max(2, int(mdlm_cfg.get('progress_buckets', 16)))
        self.uncertainty_buckets = max(2, int(mdlm_cfg.get('uncertainty_buckets', 8)))
        if self.use_dual_stage_embeddings:
            self.progress_bucket_emb = nn.Embedding(self.progress_buckets, self.n_embd)
            self.uncertainty_bucket_emb = nn.Embedding(self.uncertainty_buckets, self.n_embd)
        
        # Project augmented input back into transformer dimension (W_x)
        if self.use_dual_stage_embeddings:
            self.W_x = nn.Linear(self.n_embd * 4, self.n_embd)
        else:
            self.W_x = nn.Linear(self.n_embd * 2, self.n_embd)

        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(H.embd_pdrop)

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(H) for _ in range(self.n_layers)])
        
        # Decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.ModuleList([nn.Linear(self.n_embd, hs, bias=False) for hs in head_out_size])

        # Apply weight initialization
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            if hasattr(module, 'weight') and module.weight is not None and len(module.weight.shape) > 1:
                module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, t=None, progress_bucket=None, uncertainty_bucket=None, curriculum_weight=None):
        # 1. z^(t) Construction: Map each index to vector and concat
        channel_embs = [emb(idx[:, :, i]) for i, emb in enumerate(self.tok_emb)]
        z_t = torch.cat(channel_embs, -1)
        z_t = self.emb_red(z_t)
        z_t = F.relu(z_t)

        if self.causal:
            z_t = torch.cat(
                (self.start_tok.repeat(z_t.size(0), 1, 1), z_t),
                dim=1
            )

        seq_len = z_t.shape[1]
        assert seq_len <= self.block_size, f"Cannot forward, model block size ({self.block_size}) exhausted."

        # 2. Stage awareness conditioning (s_t)
        B = z_t.size(0)
        if t is not None:
            # t shape: (B,)
            t_clipped = torch.clamp(t.long(), 0, self.stage_emb.num_embeddings - 1)
            # s_t final shape: (B, seq_len, n_embd)
            s_t = self.stage_emb(t_clipped).unsqueeze(1).expand(-1, seq_len, -1)
        else:
            s_t = torch.zeros(B, seq_len, self.n_embd, device=z_t.device)

        # 3. Augmented Input
        # position_embeddings shape: (1, seq_len, n_embd)
        position_embeddings = self.pos_emb[:, :seq_len, :]
        z_p = z_t + position_embeddings
        
        if self.use_dual_stage_embeddings:
            if progress_bucket is None:
                if t is None:
                    progress_bucket = torch.zeros(B, device=z_t.device, dtype=torch.long)
                else:
                    progress = torch.clamp(t.float() / float(self.num_timesteps), 0.0, 1.0)
                    progress_bucket = torch.round(progress * float(self.progress_buckets - 1)).long()
            progress_bucket = torch.clamp(progress_bucket.long(), 0, self.progress_buckets - 1)
            s_progress = self.progress_bucket_emb(progress_bucket).unsqueeze(1).expand(-1, seq_len, -1)

            if uncertainty_bucket is None:
                uncertainty_bucket = torch.zeros(B, device=z_t.device, dtype=torch.long)
            uncertainty_bucket = torch.clamp(uncertainty_bucket.long(), 0, self.uncertainty_buckets - 1)
            s_unc = self.uncertainty_bucket_emb(uncertainty_bucket).unsqueeze(1).expand(-1, seq_len, -1)

            if self.use_curriculum_weight and curriculum_weight is not None:
                c = torch.clamp(curriculum_weight.float(), 0.0, 1.0).view(B, 1, 1)
                s_progress = s_progress * c
                s_unc = s_unc * c

            augmented_input = torch.cat([z_p, s_t, s_progress, s_unc], dim=-1)
        else:
            # Concat z_p and s_t -> shape (B, seq_len, 2 * n_embd)
            augmented_input = torch.cat([z_p, s_t], dim=-1)
        
        # W_x projection -> shape (B, seq_len, n_embd)
        x = self.W_x(augmented_input)
        
        x = self.drop(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = [h(x) for h in self.head]

        return logits
