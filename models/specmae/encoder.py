"""
SpecMAE Encoder.

Contains:
  - DropPath          stochastic depth regularisation
  - Attention         multi-head scaled dot-product self-attention
  - Mlp               two-layer GELU MLP
  - TransformerBlock  pre-LN Transformer block  (MHSA + MLP)
  - SpecMAEEncoder    full encoder with patch embedding, CLS token,
                      fixed 2-D sin-cos positional encoding, and
                      both 1-D random masking and 2-D structured masking

Design follows MAE (He et al., 2022) and AudioMAE (Huang et al., 2022)
without depending on timm, ensuring version-independent reproducibility.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .patch_embed import AudioPatchEmbed
from .pos_embed   import get_2d_sincos_pos_embed


# ═══════════════════════════════════════════════════════════════════════════
#  Primitive building blocks
# ═══════════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """
    Per-sample stochastic depth (Huang et al., 2016).

    During training, each sample independently has its residual branch
    dropped with probability `drop_prob`.  At test time this is a no-op.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob   = 1.0 - self.drop_prob
        shape       = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor = torch.floor(rand_tensor + keep_prob)      # binarise
        return x * rand_tensor.div(keep_prob)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with fused QKV projection.

    Args:
        dim:       Token embedding dimension.
        num_heads: Number of attention heads (dim must be divisible by this).
        qkv_bias:  Learnable bias on Q/K/V projections (default True).
        attn_drop: Dropout applied to attention weights.
        proj_drop: Dropout applied after the output projection.
    """

    def __init__(
        self,
        dim:       int,
        num_heads: int   = 8,
        qkv_bias:  bool  = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Fused QKV → split into Q, K, V
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )                                               # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)                        # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    """
    Two-layer MLP with GELU activation and optional dropout.

    Args:
        in_features:     Input dimension.
        hidden_features: Hidden layer width (default = in_features).
        out_features:    Output dimension (default = in_features).
        act_layer:       Activation class.
        drop:            Dropout probability (applied after both linear layers).
    """

    def __init__(
        self,
        in_features:     int,
        hidden_features: int | None = None,
        out_features:    int | None = None,
        act_layer                   = nn.GELU,
        drop:            float      = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features

        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block:
        x ← x + DropPath(MHSA(LN(x)))
        x ← x + DropPath(MLP(LN(x)))

    Args:
        dim:        Token embedding dimension.
        num_heads:  Attention heads.
        mlp_ratio:  MLP hidden-width expansion ratio.
        qkv_bias:   Bias in QKV projection.
        drop:       Dropout in MLP and attention output projection.
        attn_drop:  Dropout on attention weights.
        drop_path:  Stochastic depth probability.
        norm_layer: Normalisation layer constructor.
    """

    def __init__(
        self,
        dim:        int,
        num_heads:  int,
        mlp_ratio:  float = 4.0,
        qkv_bias:   bool  = True,
        drop:       float = 0.0,
        attn_drop:  float = 0.0,
        drop_path:  float = 0.0,
        norm_layer        = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU, drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  SpecMAE Encoder
# ═══════════════════════════════════════════════════════════════════════════

class SpecMAEEncoder(nn.Module):
    """
    SpecMAE Encoder.

    Implements the asymmetric masking strategy from MAE:
      1. Embed all patches via Conv2d.
      2. Mask a large fraction (default 75 %) of tokens.
      3. Pass only the visible tokens through the Transformer.

    This makes the encoder ~4× cheaper than processing all tokens,
    which is critical for the low-power sentinel DSP target.

    Masking modes
    -------------
    mask_2d=False  (default):
        Standard 1-D random masking — each token is independently
        retained or discarded.  Used for single-scenario pre-training.

    mask_2d=True:
        Structured 2-D masking — independently sample time columns and
        frequency rows to mask, then take their union.  Encourages the
        model to learn both spectral and temporal correlations, and
        produces a variable effective mask ratio ≈ 1-(1-t)*(1-f).

    Args:
        n_mels:          Mel frequency bins (must be divisible by patch_size).
        n_time_frames:   Expected padded time-frame count (divisible by patch_size).
                         Default 112 = 7 × 16, from 100 frames padded to ×16.
        patch_size:      Square patch side length in pixels.
        embed_dim:       Token embedding dimension (ViT-Base = 768).
        depth:           Number of Transformer blocks.
        num_heads:       Attention heads per block.
        mlp_ratio:       MLP hidden-width expansion ratio.
        drop_path_rate:  Peak stochastic-depth probability (linearly scheduled).
        norm_layer:      Normalisation layer constructor.
    """

    def __init__(
        self,
        n_mels:         int   = 128,
        n_time_frames:  int   = 112,
        patch_size:     int   = 16,
        embed_dim:      int   = 768,
        depth:          int   = 12,
        num_heads:      int   = 12,
        mlp_ratio:      float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer            = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()

        if n_time_frames % patch_size != 0:
            raise ValueError(
                f"n_time_frames ({n_time_frames}) must be divisible by "
                f"patch_size ({patch_size})."
            )

        # ── Patch grid metadata ─────────────────────────────────────────
        self.patch_size      = patch_size
        self.n_freq_patches  = n_mels        // patch_size   # 8
        self.n_time_patches  = n_time_frames // patch_size   # 7
        self.num_patches     = self.n_freq_patches * self.n_time_patches  # 56

        # ── Patch embedding ─────────────────────────────────────────────
        self.patch_embed = AudioPatchEmbed(
            n_mels=n_mels, patch_size=patch_size, embed_dim=embed_dim,
        )

        # ── CLS token ───────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── Fixed sin-cos positional embedding (not learnable) ──────────
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False,
        )

        # ── Transformer blocks (linear drop-path schedule) ──────────────
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=0.0, attn_drop=0.0,
                drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # 2-D sin-cos pos embed (frozen)
        pe = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_h=self.n_freq_patches,
            grid_w=self.n_time_patches,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # Patch projection: Xavier uniform (following JAX ViT)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

        # CLS token — truncated normal, consistent with mask_token and pos perturbation
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Linear and LayerNorm layers
        self.apply(_init_module_weights)

    # ------------------------------------------------------------------
    # Masking strategies
    # ------------------------------------------------------------------

    def random_masking(
        self,
        x:          torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1-D random masking (uniform over all tokens).

        Tokens are randomly shuffled; the first `n_keep` are retained
        and the rest are discarded.  This exactly follows the original
        MAE implementation.

        Args:
            x:          (B, N, D) embedded patches.
            mask_ratio: Fraction of tokens to discard.

        Returns:
            x_visible:   (B, N_vis, D)  visible tokens.
            mask:        (B, N)         binary mask — 1 = discarded.
            ids_restore: (B, N)         argsort to recover original order.
        """
        B, N, D = x.shape
        n_keep  = int(N * (1.0 - mask_ratio))

        noise       = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)           # ascending
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep  = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    # ------------------------------------------------------------------

    def random_masking_2d(
        self,
        x:             torch.Tensor,
        mask_t_prob:   float,
        mask_f_prob:   float,
        n_freq_patches: int,
        n_time_patches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Structured 2-D masking: independently mask time columns and
        frequency rows, then take their union.

        Effective mask ratio ≈ 1 - (1 - mask_t_prob) × (1 - mask_f_prob).
        Useful for learning both temporal and spectral correlations.

        Args:
            x:               (B, N, D) where N = n_freq × n_time.
            mask_t_prob:     Fraction of time-axis patches to mask.
            mask_f_prob:     Fraction of freq-axis patches to mask.
            n_freq_patches:  Patch count on frequency axis.
            n_time_patches:  Patch count on time axis.

        Returns:
            Same three tensors as random_masking.
        """
        B, N, D = x.shape
        assert N == n_freq_patches * n_time_patches

        def _axis_mask(length: int, mask_prob: float) -> torch.Tensor:
            """(B, length) binary mask — 0 = keep, 1 = remove."""
            n_keep      = int(length * (1.0 - mask_prob))
            noise       = torch.rand(B, length, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            m = torch.ones(B, length, device=x.device)
            m[:, :n_keep] = 0.0
            return torch.gather(m, dim=1, index=ids_restore)

        mask_t = _axis_mask(n_time_patches, mask_t_prob)  # (B, T)
        mask_f = _axis_mask(n_freq_patches, mask_f_prob)  # (B, F)

        # Broadcast to 2-D grid: union of row-mask and col-mask
        mask_t_2d = mask_t.unsqueeze(1).expand(-1, n_freq_patches, -1)  # (B,F,T)
        mask_f_2d = mask_f.unsqueeze(2).expand(-1, -1, n_time_patches)  # (B,F,T)
        mask_2d   = 1.0 - (1.0 - mask_t_2d) * (1.0 - mask_f_2d)        # (B,F,T)
        mask_2d   = mask_2d.reshape(B, N)                                # (B, N)

        # Sort: 0-masked (visible) first, tie-break with small noise
        ids_shuffle = torch.argsort(mask_2d + torch.rand_like(mask_2d) * 1e-6, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        n_keep   = int((mask_2d[0] < 0.5).sum().item())
        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = (mask_2d >= 0.5).float()
        return x_visible, mask, ids_restore

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:           torch.Tensor,
        mask_ratio:  float = 0.75,
        mask_2d:     bool  = False,
        mask_t_prob: float = 0.6,
        mask_f_prob: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Encode a batch of log-Mel spectrograms with masking.

        Args:
            x:           (B, 1, F, T) float32 spectrogram.
            mask_ratio:  Fraction of tokens to mask (1-D masking).
            mask_2d:     Use structured 2-D time/frequency masking.
            mask_t_prob: Time-axis mask fraction (2-D only).
            mask_f_prob: Freq-axis mask fraction (2-D only).

        Returns:
            latent:         (B, N_vis + 1, embed_dim)  CLS + visible tokens.
            mask:           (B, N)                     1 = masked token.
            ids_restore:    (B, N)                     permutation to unshuffle.
            n_freq_patches: int
            n_time_patches: int
        """
        # 1. Patch embedding (time axis padded inside AudioPatchEmbed)
        tokens, n_f, n_t = self.patch_embed(x)    # (B, N, D)

        # 2. Add positional embedding to patch tokens (skip CLS slot)
        pos = self._get_pos_embed(n_f, n_t)
        tokens = tokens + pos[:, 1:, :]

        # 3. Mask
        if mask_2d:
            tokens, mask, ids_restore = self.random_masking_2d(
                tokens, mask_t_prob, mask_f_prob, n_f, n_t,
            )
        else:
            tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio)

        # 4. Prepend CLS token (with its positional embedding)
        cls = (self.cls_token + pos[:, :1, :]).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # 5. Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens, mask, ids_restore, n_f, n_t

    # ------------------------------------------------------------------

    def _get_pos_embed(self, n_f: int, n_t: int) -> torch.Tensor:
        """Return pos_embed, interpolating if the current grid differs."""
        if n_f == self.n_freq_patches and n_t == self.n_time_patches:
            return self.pos_embed
        # Rare case: input padded to a different time length
        from .pos_embed import interpolate_pos_embed_2d
        return interpolate_pos_embed_2d(
            self.pos_embed, self.n_freq_patches, self.n_time_patches, n_f, n_t,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Shared weight-initialisation helper
# ═══════════════════════════════════════════════════════════════════════════

def _init_module_weights(m: nn.Module) -> None:
    """
    trunc_normal_(std=0.02) for Linear weights; ones/zeros for LayerNorm.

    Follows BERT / DeiT / BEiT convention: a truncated normal with a fixed
    small std gives consistent variance across all layers regardless of fan-in
    or fan-out, which is preferable to xavier_uniform_ for deep pre-LN
    Transformers where each block already provides its own scale via LN.
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
