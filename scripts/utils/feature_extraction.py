"""
Log-Mel spectrogram extraction for SpecMAE.

All audio is assumed to be already at 48 kHz, 1 s, mono (PCM-16).
The extractor returns a (1, n_mels, T) float32 tensor ready for the model.

Default parameters match base_config.yaml:
    sample_rate : 48 000 Hz
    duration    : 1.0 s  → 48 000 samples
    n_mels      : 128
    n_fft       : 1 024
    hop_length  : 480   → T ≈ 100 frames (padded to 112 inside the model)
    f_min       : 0 Hz
    f_max       : 24 000 Hz (Nyquist @ 48 kHz)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch


# ── Configuration dataclass ────────────────────────────────────────────────

@dataclass
class AudioConfig:
    """All audio / feature hyperparameters in one place."""
    sample_rate: int   = 48_000
    duration:    float = 1.0
    n_mels:      int   = 128
    n_fft:       int   = 1_024
    hop_length:  int   = 480     # 48000/480 + 1 = 101 frames (center=True)
    f_min:       float = 0.0
    f_max:       float = 24_000.0
    # Normalisation (computed over training set; update after compute_stats.py)
    norm_mean:   float = -6.0
    norm_std:    float = 5.0

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration)


# ── Core extractor ─────────────────────────────────────────────────────────

class LogMelExtractor:
    """
    Stateless log-Mel spectrogram extractor.

    Args:
        cfg:         AudioConfig instance.
        normalize:   Apply (x - mean) / std normalisation if True.
        use_db:      Convert power spectrogram to dB scale (recommended).
        top_db:      Dynamic range clip for dB conversion (default: 80 dB).
    """

    def __init__(
        self,
        cfg:       Optional[AudioConfig] = None,
        normalize: bool = True,
        use_db:    bool = True,
        top_db:    float = 80.0,
    ) -> None:
        self.cfg       = cfg or AudioConfig()
        self.normalize = normalize
        self.use_db    = use_db
        self.top_db    = top_db

    # ------------------------------------------------------------------

    def load_wav(self, path: str) -> np.ndarray:
        """
        Load a WAV file to a mono float32 array at cfg.sample_rate.
        Resamples if needed; truncates or zero-pads to cfg.n_samples.
        """
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)   # stereo → mono

        if sr != self.cfg.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.cfg.sample_rate)

        n = self.cfg.n_samples
        if len(audio) >= n:
            audio = audio[:n]
        else:
            audio = np.pad(audio, (0, n - len(audio)))

        return audio.astype(np.float32)

    # ------------------------------------------------------------------

    def extract(self, audio: np.ndarray) -> torch.Tensor:
        """
        Compute log-Mel spectrogram from a raw waveform array.

        Args:
            audio: float32 array of shape (n_samples,).

        Returns:
            spec: float32 tensor of shape (1, n_mels, T).
        """
        c = self.cfg
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=c.sample_rate,
            n_mels=c.n_mels,
            n_fft=c.n_fft,
            hop_length=c.hop_length,
            fmin=c.f_min,
            fmax=c.f_max,
        )                                    # (n_mels, T)  power spectrum

        if self.use_db:
            mel = librosa.power_to_db(mel, top_db=self.top_db)

        if self.normalize:
            mel = (mel - c.norm_mean) / (c.norm_std + 1e-8)

        return torch.from_numpy(mel).float().unsqueeze(0)   # (1, n_mels, T)

    # ------------------------------------------------------------------

    def __call__(self, path: str) -> torch.Tensor:
        """Load a WAV file and return its log-Mel spectrogram tensor."""
        return self.extract(self.load_wav(path))


# ── Dataset-level statistics ───────────────────────────────────────────────

def compute_dataset_stats(
    file_paths: list[str],
    cfg:        Optional[AudioConfig] = None,
    n_samples:  int = 500,
) -> tuple[float, float]:
    """
    Estimate dataset mean and std of raw log-Mel values (before normalisation).

    Call this once on the training set and store the results in AudioConfig.

    Args:
        file_paths: List of WAV file paths.
        cfg:        AudioConfig (normalise=False inside to get raw values).
        n_samples:  Number of files to sample for efficiency.

    Returns:
        (mean, std) as Python floats.
    """
    import random
    cfg    = cfg or AudioConfig()
    ext    = LogMelExtractor(cfg=cfg, normalize=False, use_db=True)
    paths  = random.sample(file_paths, min(n_samples, len(file_paths)))

    all_vals = []
    for p in paths:
        spec = ext(p).numpy().ravel()
        all_vals.append(spec)

    all_vals = np.concatenate(all_vals)
    return float(all_vals.mean()), float(all_vals.std())
