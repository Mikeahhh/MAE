"""
Audio Spectrogram Patch Embedding for SpecMAE.

Converts a single-channel log-Mel spectrogram (B, 1, F, T) into a flat
sequence of D-dimensional patch tokens (B, N, D), dynamically padding
the time axis to the nearest multiple of patch_size.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioPatchEmbed(nn.Module):
    """
    Patch embedding for single-channel audio spectrograms.

    A single Conv2d with kernel_size = stride = patch_size acts as the
    linear projection over each non-overlapping patch (identical to the
    standard ViT PatchEmbed but restricted to 1 input channel).
    A post-projection LayerNorm improves training stability.

    Args:
        n_mels:     Mel frequency bins (must be divisible by patch_size).
        patch_size: Side length of each square patch in spectrogram pixels.
        embed_dim:  Output embedding dimension.
    """

    def __init__(
        self,
        n_mels:     int = 128,
        patch_size: int = 16,
        embed_dim:  int = 768,
    ) -> None:
        super().__init__()
        if n_mels % patch_size != 0:
            raise ValueError(
                f"n_mels ({n_mels}) must be divisible by patch_size ({patch_size})."
            )
        self.patch_size     = patch_size
        self.n_freq_patches = n_mels // patch_size   # static: always 8 for n_mels=128, p=16

        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------

    def _pad_time(self, x: torch.Tensor) -> torch.Tensor:
        """Right-pad time axis so length is divisible by patch_size."""
        remainder = x.shape[-1] % self.patch_size
        if remainder != 0:
            x = F.pad(x, (0, self.patch_size - remainder))
        return x

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: Float tensor of shape (B, 1, F, T).

        Returns:
            tokens:         (B, N, embed_dim) patch token sequence.
            n_freq_patches: Patches along frequency axis (= F // patch_size).
            n_time_patches: Patches along the padded time axis.
        """
        x = self._pad_time(x)                          # (B, 1, F, T_pad)
        n_time_patches = x.shape[-1] // self.patch_size

        x = self.proj(x)                               # (B, D, n_f, n_t)
        x = x.flatten(2).transpose(1, 2)               # (B, N, D)
        x = self.norm(x)
        return x, self.n_freq_patches, n_time_patches
