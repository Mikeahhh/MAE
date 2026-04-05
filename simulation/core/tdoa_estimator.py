"""
TDOA (Time Difference of Arrival) estimation module.

Uses GCC-PHAT (Generalized Cross-Correlation with Phase Transform) to
estimate time delays between microphone pairs.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class TDOAResult:
    """TDOA estimation result."""
    tau: float         # Time delay [s]
    confidence: float  # Confidence (peak/secondary-peak ratio)
    peak_value: float  # Peak magnitude
    snr: float         # SNR estimate


class TDOAEstimator:
    """TDOA estimator with configurable parameters."""

    def __init__(
        self,
        fs: int = 48000,
        window_samples: int = 2048,
        max_delay_samples: int = 100,
        confidence_threshold: float = 1.5
    ):
        self.fs = fs
        self.window_samples = window_samples
        self.max_delay_samples = max_delay_samples
        self.confidence_threshold = confidence_threshold

    def estimate(self, signal1: np.ndarray, signal2: np.ndarray) -> TDOAResult:
        """Estimate TDOA between two signals."""
        tau, confidence, peak_value, snr = gcc_phat(
            signal1, signal2, self.fs, self.max_delay_samples
        )
        return TDOAResult(tau, confidence, peak_value, snr)

    def estimate_array(
        self,
        mic_signals: np.ndarray,
        reference_mic: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate TDOA for all mics relative to reference.

        Returns (tdoa_array, peak_values_array), both (n_mics-1,).
        """
        return estimate_tdoa_array(mic_signals, self.fs, reference_mic, self.max_delay_samples)


def gcc_phat(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: int = 48000,
    max_delay_samples: int = 100
) -> Tuple[float, float, float, float]:
    """
    GCC-PHAT time delay estimation.

    Math:
        1. FFT: X1(f), X2(f)
        2. Cross-spectrum: R12(f) = X1(f) * conj(X2(f))
        3. PHAT weighting: R12(f) / |R12(f)|
        4. IFFT -> r12(tau)
        5. Peak search: tau = argmax |r12(tau)|

    Returns:
        (tau_seconds, confidence, peak_value, snr)
    """
    n = min(len(signal1), len(signal2))
    signal1 = signal1[:n]
    signal2 = signal2[:n]

    X1 = np.fft.rfft(signal1, n=n)
    X2 = np.fft.rfft(signal2, n=n)

    R12 = X1 * np.conj(X2)

    epsilon = 1e-10
    R12_phat = R12 / (np.abs(R12) + epsilon)

    r12 = np.fft.irfft(R12_phat, n=n)

    # Search within max_delay window
    r12_centered = np.concatenate([r12[-max_delay_samples:],
                                   r12[:max_delay_samples + 1]])

    peak_idx = np.argmax(np.abs(r12_centered))
    tau_samples = peak_idx - max_delay_samples
    tau_seconds = tau_samples / fs

    # Confidence: peak / second-peak ratio
    abs_r12 = np.abs(r12_centered)
    sorted_peaks = np.sort(abs_r12)
    peak_value = sorted_peaks[-1]
    second_peak = sorted_peaks[-2]
    confidence = peak_value / (second_peak + epsilon)

    # SNR: peak / mean ratio
    mean_value = np.mean(abs_r12)
    snr = peak_value / (mean_value + epsilon)

    return tau_seconds, confidence, peak_value, snr


def estimate_tdoa_array(
    mic_signals: np.ndarray,
    fs: int = 48000,
    reference_mic: int = 0,
    max_delay_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute TDOA for all mics relative to reference mic.

    Returns:
        (tdoa_array, peak_values_array) — both shape (n_mics-1,).
        peak_values are the GCC-PHAT correlation peaks, usable as
        DOA confidence weights for weighted triangulation.
    """
    n_mics = mic_signals.shape[0]
    ref_signal = mic_signals[reference_mic]

    tdoa_list = []
    peak_list = []
    for i in range(n_mics):
        if i == reference_mic:
            continue
        tau, _, peak_value, _ = gcc_phat(ref_signal, mic_signals[i], fs, max_delay_samples)
        tdoa_list.append(tau)
        peak_list.append(peak_value)

    return np.array(tdoa_list), np.array(peak_list)


def estimate_tdoa_with_confidence(
    mic_signals: np.ndarray,
    fs: int = 48000,
    reference_mic: int = 0,
    max_delay_samples: int = 100,
    confidence_threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute TDOA with confidence and validity mask.

    Returns:
        (tdoa, confidence, valid_mask)
    """
    n_mics = mic_signals.shape[0]
    ref_signal = mic_signals[reference_mic]

    tdoa_list = []
    confidence_list = []

    for i in range(n_mics):
        if i == reference_mic:
            continue
        tau, conf, _, _ = gcc_phat(ref_signal, mic_signals[i], fs, max_delay_samples)
        tdoa_list.append(tau)
        confidence_list.append(conf)

    tdoa = np.array(tdoa_list)
    confidence = np.array(confidence_list)
    valid_mask = confidence >= confidence_threshold

    return tdoa, confidence, valid_mask
