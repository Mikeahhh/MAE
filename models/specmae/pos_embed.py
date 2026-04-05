"""
2-D sin-cos positional embedding utilities for SpecMAE.

Reference: MAE (He et al., 2022) https://github.com/facebookresearch/mae
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_h: int,
    grid_w: int,
    cls_token: bool = False,
) -> np.ndarray:
    """
    Build a fixed 2-D sin-cos positional embedding.

    The frequency axis occupies the first half of the embedding dimensions;
    the time axis occupies the second half.  This mirrors the MAE convention
    and works correctly for non-square patch grids.

    Args:
        embed_dim:  Total embedding dimension (must be even).
        grid_h:     Patch count along the frequency (rows) axis.
        grid_w:     Patch count along the time (columns) axis.
        cls_token:  Prepend a zero-vector for the CLS token if True.

    Returns:
        pos_embed:  float32 ndarray of shape
                    [grid_h * grid_w, embed_dim]         (cls_token=False)
                    [1 + grid_h * grid_w, embed_dim]     (cls_token=True)
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}.")

    # grid_F[i, j] = i  (frequency index)
    # grid_T[i, j] = j  (time index)
    grid_f = np.arange(grid_h, dtype=np.float32)
    grid_t = np.arange(grid_w, dtype=np.float32)
    grid_T, grid_F = np.meshgrid(grid_t, grid_f)       # both (grid_h, grid_w)

    emb_f = _sincos_1d(embed_dim // 2, grid_F.ravel())  # (N, D/2)
    emb_t = _sincos_1d(embed_dim // 2, grid_T.ravel())  # (N, D/2)
    pos_embed = np.concatenate([emb_f, emb_t], axis=1)  # (N, D)

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0
        )
    return pos_embed.astype(np.float32)


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """
    1-D sin-cos positional encoding.

    Args:
        embed_dim:  Must be even.
        positions:  1-D array of N scalar positions.

    Returns:
        emb:  float32 array of shape (N, embed_dim).
    """
    half  = embed_dim // 2
    omega = 1.0 / (10_000.0 ** (np.arange(half, dtype=np.float64) / half))
    out   = np.outer(positions.astype(np.float64), omega)          # (N, half)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


def interpolate_pos_embed_2d(
    stored:      torch.Tensor,
    grid_h_old:  int,
    grid_w_old:  int,
    grid_h_new:  int,
    grid_w_new:  int,
) -> torch.Tensor:
    """
    Bicubic interpolation of a stored 2-D positional embedding to a new
    patch-grid size.  The CLS token slot (index 0) is preserved unchanged.

    Args:
        stored:      Tensor of shape (1, 1 + N_old, D).
        grid_h_old:  Original frequency-axis patch count.
        grid_w_old:  Original time-axis patch count.
        grid_h_new:  Target frequency-axis patch count.
        grid_w_new:  Target time-axis patch count.

    Returns:
        Interpolated tensor of shape (1, 1 + N_new, D).
    """
    cls_embed   = stored[:, :1, :]                                     # (1, 1, D)
    patch_embed = stored[:, 1:, :]                                     # (1, N_old, D)
    D           = patch_embed.shape[-1]

    patch_embed = patch_embed.reshape(1, grid_h_old, grid_w_old, D)
    patch_embed = patch_embed.permute(0, 3, 1, 2)                      # (1, D, H, W)
    patch_embed = F.interpolate(
        patch_embed, size=(grid_h_new, grid_w_new),
        mode="bicubic", align_corners=False,
    )
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, -1, D)   # (1, N_new, D)
    return torch.cat([cls_embed, patch_embed], dim=1)
