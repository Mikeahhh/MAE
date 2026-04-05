"""
SpecMAE Decoder.

A lightweight Transformer decoder that:
  1. Projects encoder tokens to a lower-dimensional decoder space.
  2. Inserts learnable mask tokens to reconstruct the full patch sequence.
  3. Adds 2-D sin-cos positional embeddings.
  4. Processes all tokens (visible + mask) through shallow Transformer blocks.
  5. Projects each token back to patch-pixel space (patch_size²) via a
     linear head.

Design follows MAE (He et al., 2022).  The decoder is intentionally
shallow (4 blocks, 512-dim) to keep sentinel-mode inference cost low.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
import torch.nn as nn

from .encoder  import TransformerBlock, _init_module_weights
from .pos_embed import get_2d_sincos_pos_embed, interpolate_pos_embed_2d


class SpecMAEDecoder(nn.Module):
    """
    SpecMAE Decoder.

    Args:
        num_patches:        Total number of patches (N = n_f × n_t).
        patch_size:         Square patch side length (pixels).
        encoder_embed_dim:  Encoder output dimension.
        decoder_embed_dim:  Decoder internal dimension (default 512).
        decoder_depth:      Number of Transformer blocks (default 4).
        decoder_num_heads:  Attention heads per block (default 8).
        mlp_ratio:          MLP hidden-width ratio.
        norm_layer:         Normalisation layer constructor.
        n_freq_patches:     Patch count on frequency axis (for pos embed).
        n_time_patches:     Patch count on time axis (for pos embed).
    """

    def __init__(
        self,
        num_patches:        int,
        patch_size:         int,
        encoder_embed_dim:  int,
        decoder_embed_dim:  int   = 512,
        decoder_depth:      int   = 4,
        decoder_num_heads:  int   = 8,
        mlp_ratio:          float = 4.0,
        norm_layer                = partial(nn.LayerNorm, eps=1e-6),
        n_freq_patches:     int   = 8,
        n_time_patches:     int   = 7,
    ) -> None:
        super().__init__()

        self.patch_size      = patch_size
        self.n_freq_patches  = n_freq_patches
        self.n_time_patches  = n_time_patches
        self.num_patches     = num_patches

        # ── Latent-space projection: Z = LN(WH + b)  [Paper III.B.2] ──
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.latent_norm   = norm_layer(decoder_embed_dim)

        # ── Learnable mask token ────────────────────────────────────────
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # ── Fixed sin-cos positional embedding (decoder space) ──────────
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )

        # ── Transformer blocks (no stochastic depth — decoder is shallow) ─
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim, num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # ── Prediction head: decoder_dim → patch_size² ─────────────────
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * patch_size, bias=True,
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Sin-cos pos embed for decoder
        pe = get_2d_sincos_pos_embed(
            embed_dim=self.decoder_pos_embed.shape[-1],
            grid_h=self.n_freq_patches,
            grid_w=self.n_time_patches,
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pe).float().unsqueeze(0)
        )

        # mask_token — truncated normal, consistent with encoder cls_token
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_module_weights)

    # ------------------------------------------------------------------

    def _get_pos_embed(self, n_f: int, n_t: int) -> torch.Tensor:
        """Return decoder pos_embed, interpolating if grid size changed."""
        if n_f == self.n_freq_patches and n_t == self.n_time_patches:
            return self.decoder_pos_embed
        return interpolate_pos_embed_2d(
            self.decoder_pos_embed,
            self.n_freq_patches, self.n_time_patches,
            n_f, n_t,
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        latent:      torch.Tensor,
        ids_restore: torch.Tensor,
        n_f:         int,
        n_t:         int,
    ) -> torch.Tensor:
        """
        Reconstruct all patches from encoder output.

        Args:
            latent:      (B, N_vis + 1, encoder_embed_dim)
                         CLS token followed by visible patch tokens.
            ids_restore: (B, N)  permutation indices from encoder masking.
            n_f:         Number of frequency patches in this batch.
            n_t:         Number of time patches in this batch.

        Returns:
            pred:  (B, N, patch_size²)  predicted pixel values per patch.
        """
        B  = latent.shape[0]
        N  = n_f * n_t

        # 1. Latent-space projection: Z = LN(WH + b)  [Paper III.B.2]
        x = self.decoder_embed(latent)                      # (B, N_vis+1, D_dec)
        x = self.latent_norm(x)                             # LayerNorm on projection

        # 2. Build full sequence: visible tokens + mask tokens
        N_vis     = x.shape[1] - 1                          # exclude CLS
        n_masked  = N - N_vis
        mask_toks = self.mask_token.expand(B, n_masked, -1) # (B, n_masked, D_dec)

        # Concatenate (no CLS), then restore original patch order
        x_ = torch.cat([x[:, 1:, :], mask_toks], dim=1)    # (B, N, D_dec)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]),
        )                                                    # (B, N, D_dec) unshuffled

        # Re-prepend CLS token
        x = torch.cat([x[:, :1, :], x_], dim=1)            # (B, N+1, D_dec)

        # 3. Add positional embedding (interpolate if grid size differs)
        x = x + self._get_pos_embed(n_f, n_t)

        # 4. Shallow Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 5. Predict patch pixels; remove CLS token
        pred = self.decoder_pred(x)                         # (B, N+1, p²)
        pred = pred[:, 1:, :]                               # (B, N,   p²)

        return pred
