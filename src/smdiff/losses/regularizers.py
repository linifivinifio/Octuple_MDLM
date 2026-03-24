from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDFMDRegularizer(nn.Module):
    """Differentiable minibatch MMD + Frechet-style regularizer for token logits."""

    def __init__(
        self,
        mmd_weight: float = 0.01,
        fmd_weight: float = 0.005,
        embedding_dim: int = 64,
        bandwidths: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        max_samples: int = 4096,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mmd_weight = float(mmd_weight)
        self.fmd_weight = float(fmd_weight)
        self.embedding_dim = int(embedding_dim)
        self.bandwidths = [float(b) for b in bandwidths if float(b) > 0.0]
        if not self.bandwidths:
            self.bandwidths = [1.0]
        self.max_samples = int(max_samples)
        self.eps = float(eps)

    def _build_features(
        self,
        pred_logits: List[torch.Tensor],
        target_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pred_logits[c]: (B, Vc, L)
        b, l, c = target_tokens.shape
        device = target_tokens.device

        pred_features = []
        tgt_features = []

        for ch in range(c):
            logits_ch = pred_logits[ch].permute(0, 2, 1)  # (B, L, V)
            vocab_size = logits_ch.shape[-1]
            probs = F.softmax(logits_ch, dim=-1)

            token_idx = torch.arange(vocab_size, device=device, dtype=probs.dtype)
            pred_expected = (probs * token_idx.view(1, 1, -1)).sum(dim=-1)

            denom = max(vocab_size - 1, 1)
            pred_norm = pred_expected / float(denom)
            tgt_norm = target_tokens[:, :, ch].float() / float(denom)

            pred_features.append(pred_norm)
            tgt_features.append(tgt_norm)

        pred_feat = torch.stack(pred_features, dim=-1)  # (B, L, C)
        tgt_feat = torch.stack(tgt_features, dim=-1)    # (B, L, C)

        token_valid = valid_mask.any(dim=-1)  # (B, L)
        if not token_valid.any():
            empty = torch.zeros((0, c), device=device, dtype=pred_feat.dtype)
            return empty, empty

        pred_flat = pred_feat[token_valid]
        tgt_flat = tgt_feat[token_valid]

        if self.max_samples > 0 and pred_flat.shape[0] > self.max_samples:
            idx = torch.randperm(pred_flat.shape[0], device=device)[: self.max_samples]
            pred_flat = pred_flat[idx]
            tgt_flat = tgt_flat[idx]

        return pred_flat, tgt_flat

    def _multi_kernel_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n < 2:
            return x.new_tensor(0.0)

        xx = torch.cdist(x, x, p=2).pow(2)
        yy = torch.cdist(y, y, p=2).pow(2)
        xy = torch.cdist(x, y, p=2).pow(2)

        mmd_terms = []
        for sigma in self.bandwidths:
            gamma = 1.0 / (2.0 * sigma * sigma + self.eps)
            k_xx = torch.exp(torch.clamp(-gamma * xx, min=-50.0, max=0.0))
            k_yy = torch.exp(torch.clamp(-gamma * yy, min=-50.0, max=0.0))
            k_xy = torch.exp(torch.clamp(-gamma * xy, min=-50.0, max=0.0))

            # unbiased estimate: remove diagonal terms
            k_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1) + self.eps)
            k_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1) + self.eps)
            k_xy = k_xy.mean()
            mmd = torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0)
            mmd_terms.append(mmd)

        return torch.stack(mmd_terms).mean()

    def _frechet_style(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n < 2:
            return x.new_tensor(0.0)

        mu_x = x.mean(dim=0)
        mu_y = y.mean(dim=0)
        mean_term = (mu_x - mu_y).pow(2).sum()

        x_centered = x - mu_x
        y_centered = y - mu_y

        cov_x = (x_centered.transpose(0, 1) @ x_centered) / (n - 1 + self.eps)
        cov_y = (y_centered.transpose(0, 1) @ y_centered) / (n - 1 + self.eps)

        cov_term = (cov_x - cov_y).pow(2).sum()
        return mean_term + cov_term

    def forward(
        self,
        pred_logits: List[torch.Tensor],
        target_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_feat, tgt_feat = self._build_features(pred_logits, target_tokens, valid_mask)

        if pred_feat.shape[0] < 2:
            zero = target_tokens.new_tensor(0.0, dtype=torch.float)
            return zero, zero, zero

        mmd_loss = self._multi_kernel_mmd(tgt_feat, pred_feat)
        fmd_loss = self._frechet_style(tgt_feat, pred_feat)

        total_reg = self.mmd_weight * mmd_loss + self.fmd_weight * fmd_loss
        return total_reg, mmd_loss, fmd_loss
