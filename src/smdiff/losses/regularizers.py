from typing import List, Sequence, Tuple

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
        covariance_shrinkage_alpha: float = 0.0,
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
        self.covariance_shrinkage_alpha = max(0.0, min(1.0, float(covariance_shrinkage_alpha)))
        self.eps = float(eps)

    def _build_pred_features(
        self,
        pred_logits: List[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        # pred_logits[c]: (B, Vc, L)
        c = len(pred_logits)
        device = pred_logits[0].device

        pred_features = []
        for ch in range(c):
            logits_ch = pred_logits[ch].permute(0, 2, 1)  # (B, L, V)
            vocab_size = logits_ch.shape[-1]
            probs = F.softmax(logits_ch, dim=-1)

            token_idx = torch.arange(vocab_size, device=device, dtype=probs.dtype)
            pred_expected = (probs * token_idx.view(1, 1, -1)).sum(dim=-1)

            denom = max(vocab_size - 1, 1)
            pred_norm = pred_expected / float(denom)
            pred_features.append(pred_norm)

        pred_feat = torch.stack(pred_features, dim=-1)  # (B, L, C)
        token_valid = valid_mask.any(dim=-1)  # (B, L)
        if not token_valid.any():
            return torch.zeros((0, c), device=device, dtype=pred_feat.dtype)

        pred_flat = pred_feat[token_valid]
        if self.max_samples > 0 and pred_flat.shape[0] > self.max_samples:
            idx = torch.randperm(pred_flat.shape[0], device=device)[: self.max_samples]
            pred_flat = pred_flat[idx]
        return pred_flat

    def _build_features(
        self,
        pred_logits: List[torch.Tensor],
        target_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pred_logits[c]: (B, Vc, L)
        _, _, c = target_tokens.shape
        device = target_tokens.device

        pred_flat = self._build_pred_features(pred_logits, valid_mask)

        tgt_features = []

        for ch in range(c):
            logits_ch = pred_logits[ch].permute(0, 2, 1)  # (B, L, V)
            vocab_size = logits_ch.shape[-1]
            denom = max(vocab_size - 1, 1)
            tgt_norm = target_tokens[:, :, ch].float() / float(denom)
            tgt_features.append(tgt_norm)

        tgt_feat = torch.stack(tgt_features, dim=-1)    # (B, L, C)

        token_valid = valid_mask.any(dim=-1)  # (B, L)
        if not token_valid.any():
            empty = torch.zeros((0, c), device=device, dtype=tgt_feat.dtype)
            return empty, empty

        tgt_flat = tgt_feat[token_valid]

        if self.max_samples > 0 and pred_flat.shape[0] > self.max_samples:
            idx = torch.randperm(pred_flat.shape[0], device=device)[: self.max_samples]
            pred_flat = pred_flat[idx]
            tgt_flat = tgt_flat[idx]

        return pred_flat, tgt_flat

    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        if n < 2:
            d = x.shape[1]
            return torch.eye(d, device=x.device, dtype=x.dtype) * self.eps
        mu = x.mean(dim=0)
        x_centered = x - mu
        cov = (x_centered.transpose(0, 1) @ x_centered) / (n - 1 + self.eps)
        if self.covariance_shrinkage_alpha > 0.0:
            d = cov.shape[0]
            identity = torch.eye(d, device=cov.device, dtype=cov.dtype)
            scaled_identity = (torch.trace(cov) / max(d, 1)) * identity
            cov = (1.0 - self.covariance_shrinkage_alpha) * cov + self.covariance_shrinkage_alpha * scaled_identity
        return cov

    def _matrix_sqrt_psd(self, mat: torch.Tensor) -> torch.Tensor:
        if mat.dtype not in (torch.float32, torch.float64):
            mat = mat.float()
        sym = 0.5 * (mat + mat.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(sym)
        eigvals = torch.clamp(eigvals, min=0.0)
        sqrt_eigvals = torch.sqrt(eigvals + self.eps)
        return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.transpose(-1, -2)

    def _strict_frechet_from_stats(
        self,
        mu_x: torch.Tensor,
        cov_x: torch.Tensor,
        mu_y: torch.Tensor,
        cov_y: torch.Tensor,
    ) -> torch.Tensor:
        mean_term = (mu_x - mu_y).pow(2).sum()

        eye = torch.eye(cov_x.shape[0], device=cov_x.device, dtype=cov_x.dtype)
        cov_x_eps = cov_x + self.eps * eye
        cov_y_eps = cov_y + self.eps * eye

        sqrt_cov_x = self._matrix_sqrt_psd(cov_x_eps)
        inner = sqrt_cov_x @ cov_y_eps @ sqrt_cov_x
        sqrt_inner = self._matrix_sqrt_psd(inner)

        tr_term = torch.trace(cov_x_eps) + torch.trace(cov_y_eps) - 2.0 * torch.trace(sqrt_inner)
        return torch.clamp(mean_term + tr_term, min=0.0)

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

    def strict_frechet_to_reference(
        self,
        pred_logits: List[torch.Tensor],
        valid_mask: torch.Tensor,
        ref_mean: torch.Tensor,
        ref_cov: torch.Tensor,
    ) -> torch.Tensor:
        strict, _ = self.strict_frechet_to_reference_with_memory(
            pred_logits,
            valid_mask,
            ref_mean,
            ref_cov,
            memory_features=None,
        )
        return strict

    def strict_frechet_to_reference_with_memory(
        self,
        pred_logits: List[torch.Tensor],
        valid_mask: torch.Tensor,
        ref_mean: torch.Tensor,
        ref_cov: torch.Tensor,
        memory_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_feat = self._build_pred_features(pred_logits, valid_mask)
        if pred_feat.shape[0] < 2:
            return pred_feat.new_tensor(0.0), pred_feat

        # eigh on CUDA is not implemented for float16/bfloat16.
        # Keep strict Fréchet numerics in fp32 even under autocast.
        pred_feat = pred_feat.float()

        if memory_features is not None and memory_features.numel() > 0:
            mem = memory_features.to(device=pred_feat.device, dtype=torch.float32)
            if mem.ndim == 1:
                mem = mem.unsqueeze(0)
            gen_feat = torch.cat([mem, pred_feat], dim=0)
        else:
            gen_feat = pred_feat

        if gen_feat.shape[0] < 2:
            return gen_feat.new_tensor(0.0), pred_feat

        mu_gen = gen_feat.mean(dim=0)
        cov_gen = self._covariance(gen_feat)

        ref_mean = ref_mean.to(device=gen_feat.device, dtype=torch.float32)
        ref_cov = ref_cov.to(device=gen_feat.device, dtype=torch.float32)
        strict = self._strict_frechet_from_stats(mu_gen, cov_gen, ref_mean, ref_cov)
        return strict, pred_feat

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
