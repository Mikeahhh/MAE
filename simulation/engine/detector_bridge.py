"""
DetectorBridge — Pure reconstruction-based anomaly detection for simulation.

Bridges the trained SpecMAE model to the UAV localization simulation pipeline.
PatchKNN has been removed; detection is based entirely on MC-averaged
reconstruction error (masked autoencoder anomaly scoring).

Usage:
    bridge = DetectorBridge(
        checkpoint_path="checkpoints/sweep_desert/mr_0.50/model.pth",
        device="cpu",
    )
    is_anomaly, score, method = bridge.detect(audio_signal)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Ensure SpecMae is importable
_SPECMAE = Path(__file__).resolve().parent.parent.parent
_PROJECT = _SPECMAE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from SpecMae.models.specmae import SpecMAE, get_model_factory
from SpecMae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor


class DetectorBridge:
    """
    Bridges SpecMAE reconstruction-based anomaly detection into the simulation loop.

    Detection is performed via Monte Carlo (MC) averaged reconstruction error:
      1. Run the SpecMAE forward pass `recon_n_passes` times with random masks
      2. Average the per-patch reconstruction errors
      3. Score using top-k worst-reconstructed patches

    Args:
        checkpoint_path: Path to trained SpecMAE model checkpoint
        device: Torch device ("cpu", "mps", "cuda")
        model_size: SpecMAE variant ("tiny", "small", "base", "large")
        recon_mask_ratio: Mask ratio for reconstruction scoring
        recon_n_passes: Number of MC forward passes (higher = smoother)
        score_mode: Scoring mode ("top_k", "mean", "max")
        top_k_ratio: Fraction of worst patches to average for top_k mode
        anomaly_threshold: Score threshold for anomaly decision
        verbose: Print loading info
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        model_size: str = "base",
        recon_mask_ratio: float = 0.75,
        recon_n_passes: int = 100,
        score_mode: str = "top_k",
        top_k_ratio: float = 0.15,
        anomaly_threshold: Optional[float] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.recon_mask_ratio = recon_mask_ratio
        self.recon_n_passes = recon_n_passes
        self.score_mode = score_mode
        self.top_k_ratio = top_k_ratio

        # Device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load SpecMAE model + AudioConfig from checkpoint
        self.model, self.mask_ratio, self.audio_cfg = self._load_model(
            checkpoint_path, model_size
        )
        self.extractor = LogMelExtractor(cfg=self.audio_cfg)

        # Anomaly threshold
        if anomaly_threshold is not None:
            self.threshold = anomaly_threshold
        else:
            self.threshold = 0.5
            if self.verbose:
                print(f"  [WARN] Using default threshold={self.threshold}. "
                      f"Set from eval results for best performance.")

        self.detection_method = "reconstruction"

        if self.verbose:
            print(f"  DetectorBridge ready: method={self.detection_method}, "
                  f"n_passes={self.recon_n_passes}, "
                  f"threshold={self.threshold:.4f}, device={self.device}")

    def _load_model(
        self, checkpoint_path: str, model_size: str
    ) -> Tuple[SpecMAE, float, AudioConfig]:
        """Load SpecMAE model from checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        mask_ratio = float(ckpt.get("mask_ratio", 0.75))
        cfg_dict = ckpt.get("audio_cfg", {})
        cfg = AudioConfig(
            sample_rate=cfg_dict.get("sample_rate", 48_000),
            n_mels=cfg_dict.get("n_mels", 128),
            n_fft=cfg_dict.get("n_fft", 1_024),
            hop_length=cfg_dict.get("hop_length", 480),
            f_min=cfg_dict.get("f_min", 0.0),
            f_max=cfg_dict.get("f_max", 24_000.0),
            norm_mean=cfg_dict.get("norm_mean", -6.0),
            norm_std=cfg_dict.get("norm_std", 5.0),
        )
        factory = get_model_factory(model_size)
        model = factory(mask_ratio=mask_ratio, norm_pix_loss=True, n_mels=cfg.n_mels)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing and self.verbose:
            print(f"  [INFO] Missing keys (using defaults): {missing}")
        if unexpected and self.verbose:
            print(f"  [WARN] Unexpected keys in checkpoint: {unexpected}")
        model.to(self.device).eval()

        if self.verbose:
            print(f"  SpecMAE loaded: mask_ratio={mask_ratio}, "
                  f"model_size={model_size}, device={self.device}")
            print(f"  AudioConfig: sr={cfg.sample_rate}, n_mels={cfg.n_mels}, "
                  f"hop={cfg.hop_length}, norm=({cfg.norm_mean:.1f}, {cfg.norm_std:.1f})")

        return model, mask_ratio, cfg

    def detect(self, audio_signal: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect anomaly from raw audio signal.

        Args:
            audio_signal: (n_samples,) raw waveform at model sample rate

        Returns:
            (is_anomaly, score, method)
            is_anomaly: True if anomaly detected
            score: Anomaly score (higher = more anomalous)
            method: always "reconstruction"
        """
        # Ensure 1.0s duration — zero-pad if shorter
        target_samples = self.audio_cfg.n_samples
        if len(audio_signal) < target_samples:
            audio_signal = np.pad(audio_signal, (0, target_samples - len(audio_signal)))
        elif len(audio_signal) > target_samples:
            audio_signal = audio_signal[:target_samples]

        # Audio -> spectrogram
        spec = self.extractor.extract(audio_signal.astype(np.float32))  # (1, n_mels, T)
        spec_batch = spec.unsqueeze(0).to(self.device)  # (1, 1, n_mels, T)

        score = self._score_reconstruction(spec_batch)
        is_anomaly = score > self.threshold
        return is_anomaly, float(score), "reconstruction"

    @torch.no_grad()
    def _score_reconstruction(self, spec_batch: torch.Tensor) -> float:
        """Compute MC-averaged reconstruction anomaly score."""
        scores = self.model.compute_anomaly_score(
            spec_batch,
            mask_ratio=self.recon_mask_ratio,
            n_passes=self.recon_n_passes,
            score_mode=self.score_mode,
            top_k_ratio=self.top_k_ratio,
        )
        return float(scores[0])

    def score_batch(self, audio_batch: List[np.ndarray]) -> np.ndarray:
        """
        Score a batch of audio signals.

        Args:
            audio_batch: list of (n_samples,) waveforms

        Returns:
            scores: (N,) anomaly scores
        """
        target_samples = self.audio_cfg.n_samples
        specs = []
        for audio in audio_batch:
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            elif len(audio) > target_samples:
                audio = audio[:target_samples]
            spec = self.extractor.extract(audio.astype(np.float32))
            specs.append(spec)

        spec_batch = torch.stack(specs, dim=0).to(self.device)

        with torch.no_grad():
            scores = self.model.compute_anomaly_score(
                spec_batch,
                mask_ratio=self.recon_mask_ratio,
                n_passes=self.recon_n_passes,
                score_mode=self.score_mode,
                top_k_ratio=self.top_k_ratio,
            )
        return scores.cpu().numpy()

    def detect_from_spectrogram(self, spec: torch.Tensor) -> Tuple[bool, float, str]:
        """
        Detect anomaly from a pre-computed spectrogram tensor.

        Args:
            spec: (1, n_mels, T) spectrogram tensor

        Returns:
            (is_anomaly, score, method)
        """
        spec_batch = spec.unsqueeze(0).to(self.device) if spec.dim() == 3 else spec.to(self.device)
        score = self._score_reconstruction(spec_batch)
        return score > self.threshold, float(score), "reconstruction"
