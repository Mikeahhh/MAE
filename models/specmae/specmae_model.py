"""
SpecMAE — Spectrogram Masked Autoencoder for UAV acoustic anomaly detection.

Architecture overview
---------------------
  Input  : (B, 1, F, T)  single-channel log-Mel spectrogram
  ↓ AudioPatchEmbed       patch_size=16 non-overlapping patches, pad T→T_pad
  ↓ SpecMAEEncoder        CLS token + 2-D sin-cos pos embed + random masking
                          → only visible tokens (25 %) pass through 12 blocks
  ↓ SpecMAEDecoder        project to 512-d; insert mask tokens; 4 Transformer
                          blocks; linear head → patch_size² pixel predictions
  Loss   : MSE on masked patches only (norm_pix_loss=True by default)

Training   : mask_ratio=0.75  (tunable via cross-validation)
Inference  : compute_anomaly_score() — MSE over ALL patches (no masking)
             → high score signals an anomaly (e.g., human distress call)

Cross-validation usage
----------------------
  Use mask_ratio as the primary hyperparameter to sweep:
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        model = specmae_vit_base_patch16(mask_ratio=mask_ratio)
        ...
  After finding optimal mask_ratio, retrain on ALL data.

References
----------
  MAE     : He et al., 2022  https://arxiv.org/abs/2111.06377
  AudioMAE: Huang et al., 2022  https://arxiv.org/abs/2207.06405
  AST     : Gong et al., 2021  https://arxiv.org/abs/2104.01778
  DCASE   : Anomaly detection baseline — reconstruction-error scoring
"""
from __future__ import annotations

import math
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SpecMAEEncoder
from .decoder import SpecMAEDecoder


# ═══════════════════════════════════════════════════════════════════════════
#  Main model
# ═══════════════════════════════════════════════════════════════════════════

class SpecMAE(nn.Module):
    """
    Spectrogram Masked Autoencoder (SpecMAE).

    Trained exclusively on normal UAV background noise (drone + environment).
    During inference, human distress signals cannot be reconstructed, causing
    a step-like spike in reconstruction error that triggers the responder.

    Args:
        n_mels:             Mel frequency bins.
        n_time_frames:      Padded time frames (must be divisible by patch_size).
                            Default: 112 → 100 frames padded to next multiple of 16.
        patch_size:         Square patch side length in spectrogram pixels.
        embed_dim:          Encoder token embedding dimension.
        depth:              Encoder Transformer depth.
        num_heads:          Encoder attention heads.
        mlp_ratio:          MLP expansion ratio (encoder and decoder).
        decoder_embed_dim:  Decoder embedding dimension.
        decoder_depth:      Decoder Transformer depth.
        decoder_num_heads:  Decoder attention heads.
        mask_ratio:         Default masking fraction for training.
                            Primary hyperparameter for cross-validation.
        norm_pix_loss:      Normalise each patch to zero mean/unit var before
                            computing MSE (recommended — improves training).
        drop_path_rate:     Peak stochastic-depth drop probability (encoder).
    """

    def __init__(
        self,
        n_mels:            int   = 128,
        n_time_frames:     int   = 112,
        patch_size:        int   = 16,
        embed_dim:         int   = 768,
        depth:             int   = 12,
        num_heads:         int   = 12,
        mlp_ratio:         float = 4.0,
        decoder_embed_dim: int   = 512,
        decoder_depth:     int   = 4,
        decoder_num_heads: int   = 8,
        mask_ratio:        float = 0.75,
        norm_pix_loss:     bool  = True,
        drop_path_rate:    float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_size    = patch_size
        self.mask_ratio    = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        _norm = partial(nn.LayerNorm, eps=1e-6)

        n_freq_patches = n_mels        // patch_size   # 8
        n_time_patches = n_time_frames // patch_size   # 7
        num_patches    = n_freq_patches * n_time_patches  # 56

        # ── Encoder ─────────────────────────────────────────────────────
        self.encoder = SpecMAEEncoder(
            n_mels=n_mels,
            n_time_frames=n_time_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=_norm,
        )

        # ── Decoder ─────────────────────────────────────────────────────
        self.decoder = SpecMAEDecoder(
            num_patches=num_patches,
            patch_size=patch_size,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=_norm,
            n_freq_patches=n_freq_patches,
            n_time_patches=n_time_patches,
        )

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Decompose a spectrogram into non-overlapping patches.

        The time axis is right-padded (with zeros) to the nearest multiple
        of patch_size to match the encoder's AudioPatchEmbed behaviour.

        Args:
            imgs:  (B, 1, F, T) float spectrogram.

        Returns:
            patches: (B, N, patch_size²)  flattened patch pixel values.
        """
        p = self.patch_size
        B, C, n_freq, n_time = imgs.shape

        # Align time dimension
        n_time_pad = math.ceil(n_time / p) * p
        if n_time_pad != n_time:
            imgs = F.pad(imgs, (0, n_time_pad - n_time))

        h = n_freq   // p    # 8
        w = n_time_pad // p  # 7

        # (B, 1, h, p, w, p) → (B, h, w, p, p, 1) → (B, N, p²)
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)       # (B, h, w, p_f, p_t, C)
        x = x.reshape(B, h * w, p * p * C)    # C=1, so p²*1 = p²
        return x

    # ------------------------------------------------------------------

    def unpatchify(
        self,
        patches: torch.Tensor,
        n_f:     int,
        n_t:     int,
    ) -> torch.Tensor:
        """
        Reconstruct a spectrogram from patch tokens.

        Args:
            patches: (B, N, patch_size²) predicted patch pixels.
            n_f:     Frequency-axis patch count.
            n_t:     Time-axis patch count.

        Returns:
            imgs: (B, 1, n_f × patch_size, n_t × patch_size)
        """
        p = self.patch_size
        B = patches.shape[0]

        x = patches.reshape(B, n_f, n_t, p, p, 1)  # (B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)             # (B, 1, h, p_f, w, p_t)
        imgs = x.reshape(B, 1, n_f * p, n_t * p)    # (B, 1, F, T_pad)
        return imgs

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruction MSE, computed only on masked patches.

        Normalising each patch to zero mean / unit variance before the MSE
        (norm_pix_loss=True) is recommended: it prevents the model from
        concentrating capacity on high-energy patches and leads to more
        uniform feature learning across the spectrogram.

        Args:
            imgs:  (B, 1, F, T)  input spectrograms.
            pred:  (B, N, p²)    decoder predictions.
            mask:  (B, N)        binary mask — 1 = this patch was masked.

        Returns:
            loss:  scalar.
        """
        target = self.patchify(imgs)       # (B, N, p²)

        if self.norm_pix_loss:
            mean   = target.mean(dim=-1, keepdim=True)
            var    = target.var(dim=-1,  keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2        # (B, N, p²)
        loss = loss.mean(dim=-1)           # (B, N)  mean per patch

        # Average over masked patches only; guard against empty mask
        if mask.sum() == 0:
            # mask_ratio=0: autoencoder mode — loss on all patches
            loss = loss.mean()
        else:
            denom = mask.sum().clamp(min=1.0)
            loss  = (loss * mask).sum() / denom
        return loss

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        imgs:        torch.Tensor,
        mask_ratio:  Optional[float] = None,
        mask_2d:     bool            = False,
        mask_t_prob: float           = 0.6,
        mask_f_prob: float           = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            imgs:        (B, 1, F, T) log-Mel spectrogram batch.
            mask_ratio:  Override model default mask ratio.
            mask_2d:     Use structured 2-D masking instead of 1-D random.
            mask_t_prob: Time-axis mask fraction (2-D masking only).
            mask_f_prob: Freq-axis mask fraction (2-D masking only).

        Returns:
            loss:  Scalar training loss (MSE on masked patches).
            pred:  (B, N, p²) decoder predictions.
            mask:  (B, N) binary mask used in this forward pass.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        latent, mask, ids_restore, n_f, n_t = self.encoder(
            imgs, mask_ratio=mask_ratio,
            mask_2d=mask_2d, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob,
        )
        pred = self.decoder(latent, ids_restore, n_f, n_t)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ------------------------------------------------------------------
    # Anomaly scoring (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        imgs:       torch.Tensor,
        mask_ratio: Optional[float] = None,
        n_passes:   int             = 100,
        score_mode: str             = "top_k",
        top_k_ratio: float          = 0.30,
    ) -> torch.Tensor:
        """
        Per-sample anomaly score for sentinel-mode inference.

        Runs the model with the training mask ratio and computes the
        reconstruction error, aggregated via *score_mode*.
        Averaged over `n_passes` independent random masks for stability.

        A high score means the model cannot reconstruct the input → anomaly.
        A low score means the content matches the learned normal manifold.

        Args:
            imgs:        (B, 1, F, T) spectrogram batch.
            mask_ratio:  Override default mask ratio (None = use training ratio).
            n_passes:    Number of random-mask passes to average.
            score_mode:  Patch aggregation strategy:
                         "mean"  — mean MSE over all patches (baseline).
                         "max"   — max single-patch MSE (most sensitive to
                                   localised anomalies, but noisy).
                         "top_k" — mean of top-k% worst-reconstructed patches
                                   (best trade-off for low-SNR detection).
            top_k_ratio: Fraction of patches used when score_mode="top_k"
                         (default 0.2 = top 20%).

        Returns:
            scores: (B,) per-sample anomaly scores (MSE, non-negative).
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        was_training = self.training
        self.eval()

        target = self.patchify(imgs)   # (B, N, p²) — computed once
        if self.norm_pix_loss:
            mean   = target.mean(dim=-1, keepdim=True)
            var    = target.var(dim=-1,  keepdim=True)
            target_norm = (target - mean) / (var + 1e-6).sqrt()
        else:
            target_norm = target

        accumulated = torch.zeros(imgs.shape[0], device=imgs.device)

        for _ in range(n_passes):
            latent, _, ids_restore, n_f, n_t = self.encoder(
                imgs, mask_ratio=mask_ratio,
            )
            pred = self.decoder(latent, ids_restore, n_f, n_t)   # (B, N, p²)

            mse = (pred - target_norm) ** 2                      # (B, N, p²)
            patch_mse = mse.mean(dim=-1)                         # (B, N)

            if score_mode == "max":
                accumulated += patch_mse.max(dim=-1).values      # (B,)
            elif score_mode == "top_k":
                k = max(1, int(patch_mse.shape[-1] * top_k_ratio))
                topk_vals, _ = patch_mse.topk(k, dim=-1)         # (B, k)
                accumulated += topk_vals.mean(dim=-1)            # (B,)
            else:  # "mean"
                accumulated += patch_mse.mean(dim=-1)            # (B,)

        scores = accumulated / n_passes

        if was_training:
            self.train()

        return scores

    # ------------------------------------------------------------------
    # Multi-scale anomaly scoring (MC-smoothed)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_multiscale_anomaly_score(
        self,
        imgs:         torch.Tensor,
        mask_ratios:  Tuple[float, ...] = (0.3, 0.5, 0.7),
        weights:      Optional[Tuple[float, ...]] = None,
        n_passes:     int   = 50,
        score_mode:   str   = "top_k",
        top_k_ratio:  float = 0.15,
        patch_znorm:  bool  = True,
    ) -> torch.Tensor:
        """
        Multi-scale reconstruction anomaly score with MC smoothing.

        Runs reconstruction at multiple mask ratios and fuses the scores.
        Each mask ratio is averaged over `n_passes` independent random masks
        for stability (Monte Carlo smoothing).

        Patch-level z-score normalisation (`patch_znorm=True`) equalises
        contributions from high-energy and low-energy patches.

        Args:
            imgs:         (B, 1, F, T) spectrogram batch.
            mask_ratios:  Tuple of mask ratios to evaluate.
            weights:      Per-ratio weights (default: equal).
            n_passes:     MC passes per mask ratio.
            score_mode:   Patch aggregation: "mean", "max", "top_k".
            top_k_ratio:  Fraction for top_k mode (default 0.15).
            patch_znorm:  Z-score normalise per-patch MSE before aggregation.

        Returns:
            scores: (B,) per-sample anomaly scores.
        """
        if weights is None:
            weights = tuple(1.0 / len(mask_ratios) for _ in mask_ratios)
        assert len(weights) == len(mask_ratios)

        was_training = self.training
        self.eval()

        target = self.patchify(imgs)  # (B, N, p²)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1,  keepdim=True)
            target_norm = (target - mean) / (var + 1e-6).sqrt()
        else:
            target_norm = target

        B = imgs.shape[0]
        fused = torch.zeros(B, device=imgs.device)

        for mr, w in zip(mask_ratios, weights):
            accumulated = torch.zeros(B, device=imgs.device)
            for _ in range(n_passes):
                latent, _, ids_restore, n_f, n_t = self.encoder(
                    imgs, mask_ratio=mr,
                )
                pred = self.decoder(latent, ids_restore, n_f, n_t)

                mse = (pred - target_norm) ** 2          # (B, N, p²)
                patch_mse = mse.mean(dim=-1)             # (B, N)

                # Optional per-patch z-score normalisation
                if patch_znorm:
                    pm_mean = patch_mse.mean(dim=-1, keepdim=True)
                    pm_std  = patch_mse.std(dim=-1, keepdim=True).clamp(min=1e-6)
                    patch_mse = (patch_mse - pm_mean) / pm_std

                if score_mode == "max":
                    accumulated += patch_mse.max(dim=-1).values
                elif score_mode == "top_k":
                    k = max(1, int(patch_mse.shape[-1] * top_k_ratio))
                    topk_vals, _ = patch_mse.topk(k, dim=-1)
                    accumulated += topk_vals.mean(dim=-1)
                else:
                    accumulated += patch_mse.mean(dim=-1)

            fused += w * (accumulated / n_passes)

        if was_training:
            self.train()

        return fused

    # ------------------------------------------------------------------
    # Feature extraction (for latent-space anomaly detection)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(
        self,
        imgs:       torch.Tensor,
        mask_ratio: float = 0.0,
        n_passes:   int   = 1,
    ) -> dict[str, torch.Tensor]:
        """
        Extract encoder features for downstream anomaly detection.

        With mask_ratio=0 all patches are visible and the CLS token
        aggregates global information via self-attention.

        Args:
            imgs:       (B, 1, F, T) spectrogram batch.
            mask_ratio: 0.0 = no masking (recommended).
            n_passes:   >1 to average features over random masks.

        Returns:
            dict with keys:
                "cls":        (B, D) CLS token output
                "patch_mean": (B, D) mean-pooled patch tokens
                "concat":     (B, 2D) concat of cls + patch_mean
        """
        was_training = self.training
        self.eval()

        D = self.encoder.cls_token.shape[-1]
        B = imgs.shape[0]
        cls_acc = torch.zeros(B, D, device=imgs.device)
        patch_acc = torch.zeros(B, D, device=imgs.device)

        for _ in range(n_passes):
            latent, mask, ids_restore, n_f, n_t = self.encoder(
                imgs, mask_ratio=mask_ratio,
            )
            # latent: (B, N_vis+1, D) — first token is CLS
            cls_acc += latent[:, 0, :]            # (B, D)
            patch_acc += latent[:, 1:, :].mean(dim=1)  # (B, D)

        cls_feat = cls_acc / n_passes
        patch_feat = patch_acc / n_passes

        if was_training:
            self.train()

        return {
            "cls":        cls_feat,
            "patch_mean": patch_feat,
            "concat":     torch.cat([cls_feat, patch_feat], dim=-1),
        }

    # ------------------------------------------------------------------
    # Convenience: full reconstruction (for visualisation / debugging)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(
        self,
        imgs:       torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct spectrograms and return visible-region overlay.

        Args:
            imgs:       (B, 1, F, T) input spectrograms.
            mask_ratio: Masking fraction (default = training ratio).

        Returns:
            recon:    (B, 1, F, T_pad) full reconstruction.
            masked:   (B, 1, F, T_pad) input with masked regions zeroed.
            mask_map: (B, 1, F, T_pad) binary mask in spectrogram space.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        was_training = self.training
        self.eval()

        latent, mask, ids_restore, n_f, n_t = self.encoder(
            imgs, mask_ratio=mask_ratio,
        )
        pred  = self.decoder(latent, ids_restore, n_f, n_t)    # (B, N, p²)
        recon = self.unpatchify(pred, n_f, n_t)                # (B, 1, F, T_pad)

        # Build mask map in spectrogram pixel space
        B, N = mask.shape
        p    = self.patch_size
        mask_patches = mask.unsqueeze(-1).expand(-1, -1, p * p)    # (B, N, p²)
        mask_map     = self.unpatchify(mask_patches.float(), n_f, n_t)  # (B,1,F,T)

        # Pad original input to match reconstruction shape
        n_time_pad = n_t * p
        n_time     = imgs.shape[-1]
        if n_time_pad != n_time:
            imgs_pad = F.pad(imgs, (0, n_time_pad - n_time))
        else:
            imgs_pad = imgs

        masked = imgs_pad * (1.0 - mask_map)

        if was_training:
            self.train()

        return recon, masked, mask_map

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        enc = self.encoder
        dec = self.decoder
        N   = enc.num_patches
        return (
            f"patches={N} ({enc.n_freq_patches}×{enc.n_time_patches}), "
            f"mask_ratio={self.mask_ratio}, "
            f"encoder=[{enc.blocks[0].attn.qkv.in_features}d, "
            f"{len(enc.blocks)}L, {enc.blocks[0].attn.num_heads}H], "
            f"decoder=[{dec.decoder_embed.out_features}d, "
            f"{len(dec.decoder_blocks)}L, {dec.decoder_blocks[0].attn.num_heads}H]"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Factory functions
# ═══════════════════════════════════════════════════════════════════════════

def specmae_vit_tiny_patch16(**kwargs) -> SpecMAE:
    """
    SpecMAE-Tiny: embed_dim=192, depth=12, num_heads=3.
    ~5-6M params — suitable for edge deployment and real-time UAV inference.
    """
    return SpecMAE(
        patch_size=16,
        embed_dim=192,      depth=12,  num_heads=3,
        decoder_embed_dim=96, decoder_depth=2, decoder_num_heads=3,
        mlp_ratio=4.0,
        **kwargs,
    )


def specmae_vit_small_patch16(**kwargs) -> SpecMAE:
    """
    SpecMAE-Small: embed_dim=384, depth=12, num_heads=6.
    Suitable for fast experimentation and low-memory devices.
    """
    return SpecMAE(
        patch_size=16,
        embed_dim=384,      depth=12,  num_heads=6,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4.0,
        **kwargs,
    )


def specmae_vit_base_patch16(**kwargs) -> SpecMAE:
    """
    SpecMAE-Base: embed_dim=768, depth=12, num_heads=12.
    Default model — matches the configuration in the SPAWC paper.
    """
    return SpecMAE(
        patch_size=16,
        embed_dim=768,      depth=12,  num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=8,
        mlp_ratio=4.0,
        **kwargs,
    )


def specmae_vit_large_patch16(**kwargs) -> SpecMAE:
    """
    SpecMAE-Large: embed_dim=1024, depth=24, num_heads=16.
    Use for multi-scenario pre-training with large compute budgets.
    """
    return SpecMAE(
        patch_size=16,
        embed_dim=1024,     depth=24,  num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0,
        **kwargs,
    )


# ── Aliases ────────────────────────────────────────────────────────────────
specmae_tiny  = specmae_vit_tiny_patch16
specmae_small = specmae_vit_small_patch16
specmae_base  = specmae_vit_base_patch16
specmae_large = specmae_vit_large_patch16


# ── Model registry (for --model_size CLI arg) ────────────────────────────
MODEL_REGISTRY: Dict[str, Callable] = {
    "tiny":  specmae_vit_tiny_patch16,
    "small": specmae_vit_small_patch16,
    "base":  specmae_vit_base_patch16,
    "large": specmae_vit_large_patch16,
}


def get_model_factory(model_size: str = "base") -> Callable:
    """Look up factory function by size name."""
    if model_size not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_size={model_size!r}. "
            f"Choose from {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_size]
