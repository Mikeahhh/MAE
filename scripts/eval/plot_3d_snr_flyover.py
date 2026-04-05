"""
3D SNR flyover geometry figure for SPAWC paper.

Visualizes **why** the SNR follows a U-curve: a UAV flies a straight-line
path at altitude while a person (source "P") sits on the ground offset
from the flight line.  3D perspective shows flight path, distance lines,
and height/offset geometry.

With ``--with_error``, adds a right-side dual-panel showing:
  - **Top**: SNR U-curve + detection probability (from V4 height-sweep data)
  - **Bottom**: localization error (sliding-window U-curve) + DOA angular error

Usage:
    python -m SpecMae.scripts.eval.plot_3d_snr_flyover --terrain desert --show
    python -m SpecMae.scripts.eval.plot_3d_snr_flyover --terrain desert --with_error --show
    python -m SpecMae.scripts.eval.plot_3d_snr_flyover --combined --with_error
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ── Path setup (same pattern as eval_dynamic_snr.py) ────────────────────
_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent          # SpecMae/
_PROJECT = _SPEC.parent                 # model_train_example/
sys.path.insert(0, str(_PROJECT))

from SpecMae.simulation.core.propagation_model import (
    get_propagation_model,
    PropagationModel,
    DEFAULT_SOURCE_SPL,
    DEFAULT_DRONE_NOISE_SPL,
    DEFAULT_HORIZONTAL_OFFSET,
)
from SpecMae.simulation.core.triangulation import triangulate_source
from SpecMae.simulation.core.tdoa_estimator import estimate_tdoa_array
from SpecMae.scripts.eval.eval_dynamic_snr import estimate_doa_error
from SpecMae.simulation.visualization.scene_3d import set_publication_style

RESULTS_ROOT = _SPEC / "results"
FIGURES_DIR  = RESULTS_ROOT / "figures"


# ═══════════════════════════════════════════════════════════════════════════
#  Terrain-specific flyover parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FlyoverParams:
    flight_height: float      # [m] UAV altitude AGL
    person_offset_y: float    # [m] main flight line Y offset from P
    path_start: float         # [m] start X of main flight path
    path_end: float           # [m] end X of main flight path
    sentinel_y_start: float = -4.0  # [m] sentinel takeoff Y coordinate
    source_spl: float = DEFAULT_SOURCE_SPL       # 120 dB SPL @ 1m (whiteboard)
    noise_spl: float = DEFAULT_DRONE_NOISE_SPL   # 75 dB drone noise

    @property
    def mic_height(self) -> float:
        """Mic = UAV height (no suspension cable)."""
        return self.flight_height

    @property
    def path_half_length(self) -> float:
        """Backward compat: max absolute extent."""
        return max(abs(self.path_start), abs(self.path_end))


FLYOVER_PARAMS = {
    "desert": FlyoverParams(
        flight_height=5, person_offset_y=DEFAULT_HORIZONTAL_OFFSET,
        path_start=-500, path_end=200, sentinel_y_start=-4.0,
    ),
    "forest": FlyoverParams(
        flight_height=15, person_offset_y=DEFAULT_HORIZONTAL_OFFSET,
        path_start=-500, path_end=200, sentinel_y_start=-4.0,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry computation (pure data)
# ═══════════════════════════════════════════════════════════════════════════

def compute_flyover_geometry(
    terrain: str = "desert",
    n_points: int = 200,
    n_distance_lines: int = 7,
) -> dict:
    """
    Compute all positions, distances, and SNR values for the flyover figure.

    Returns dict with:
        path_x      : (N,)   X coords along flight path
        uav_pos     : (N, 3) UAV positions
        mic_pos     : (N, 3) mic positions (below UAV)
        person_pos  : (3,)   person on ground
        distances_3d: (N,)   3D distance from each mic pos to person
        snr_values  : (N,)   SNR at each position [dB]
        line_indices: list[int]  indices of the N_lines sample positions
        params      : FlyoverParams
    """
    params = FLYOVER_PARAMS[terrain]
    model = get_propagation_model(terrain)

    # Person at origin (ground level)
    person_pos = np.array([0.0, 0.0, 0.0])

    # Flight path along X axis, offset in Y by person_offset_y, at flight_height
    path_x = np.linspace(params.path_start, params.path_end, n_points)
    uav_pos = np.column_stack([path_x, np.full(n_points, params.person_offset_y), np.full(n_points, params.flight_height)])
    mic_pos = np.column_stack([path_x, np.full(n_points, params.person_offset_y), np.full(n_points, params.mic_height)])

    # 3D distances from mic to person
    diff = mic_pos - person_pos[np.newaxis, :]
    distances_3d = np.linalg.norm(diff, axis=1)

    # SNR via propagation model (uses horizontal distance internally)
    horiz_dist = np.sqrt(path_x**2 + params.person_offset_y**2)
    snr_values = np.array([
        model.snr_at_distance(
            float(d), params.source_spl, params.noise_spl,
            params.flight_height, source_height=0.0,
        )
        for d in horiz_dist
    ])

    # Pick indices for distance lines — evenly spaced along path
    line_indices = np.linspace(10, n_points - 11, n_distance_lines, dtype=int).tolist()

    return {
        "path_x": path_x,
        "uav_pos": uav_pos,
        "mic_pos": mic_pos,
        "person_pos": person_pos,
        "distances_3d": distances_3d,
        "snr_values": snr_values,
        "line_indices": line_indices,
        "params": params,
        "terrain": terrain,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  V4 height-sweep detection data loader
# ═══════════════════════════════════════════════════════════════════════════

def load_height_sweep_detection_data(
    scenario: str,
    mask_ratio: float = 0.15,
) -> dict | None:
    """
    Load detection probability vs SNR from height_sweep_{scenario}.json.

    Extracts (snr_db, detection_accuracy/100) pairs for the given mask_ratio,
    returning a dict suitable for interpolation:
        {"snr_levels": [float, ...], "det_rates": [float, ...]}
    sorted by snr_levels ascending.

    Returns None if file not found or mask_ratio not present.
    """
    # Try both naming patterns
    candidates = [
        RESULTS_ROOT / f"height_sweep_{scenario}" / f"height_sweep_{scenario}.json",
        RESULTS_ROOT / f"height_sweep_{scenario}.json",
    ]
    data_path = None
    for c in candidates:
        if c.exists():
            data_path = c
            break
    if data_path is None:
        return None

    with open(data_path) as f:
        data = json.load(f)

    # Find the result block for the requested mask_ratio
    target_result = None
    for r in data.get("results", []):
        if abs(r.get("mask_ratio", -1) - mask_ratio) < 0.005:
            target_result = r
            break
    if target_result is None:
        return None

    snr_levels = []
    det_rates = []
    for height_key, height_data in target_result.get("per_height", {}).items():
        snr_db = height_data.get("snr_db")
        # Use detection_accuracy (raw MC values — NOT monotonicity-enforced)
        det_acc = height_data.get("detection_accuracy", 0.0)
        if snr_db is not None:
            snr_levels.append(float(snr_db))
            det_rates.append(float(det_acc) / 100.0)

    if not snr_levels:
        return None

    # Sort by SNR ascending
    order = np.argsort(snr_levels)
    snr_levels = [snr_levels[i] for i in order]
    det_rates = [det_rates[i] for i in order]

    # Keep raw MC data — natural micro-fluctuations are expected and required
    return {"snr_levels": snr_levels, "det_rates": det_rates}


def interpolate_detection_from_height_sweep(
    snr_db: float,
    sweep_data: dict,
) -> float:
    """
    Piecewise linear interpolation of detection probability from height-sweep data.

    Below lowest measured SNR: linearly extrapolate toward 0 (clamped at 0).
    Above highest measured SNR: cap at highest measured value.
    """
    snr_levels = sweep_data["snr_levels"]
    det_rates = sweep_data["det_rates"]

    if snr_db <= snr_levels[0]:
        # Linearly decrease toward 0 at low SNR
        # Physical: lower SNR → lower detection probability
        falloff_db = 40.0
        zero_snr = snr_levels[0] - falloff_db
        if snr_db <= zero_snr:
            return 0.0
        t = (snr_db - zero_snr) / falloff_db
        return det_rates[0] * t

    if snr_db >= snr_levels[-1]:
        return min(1.0, det_rates[-1])

    # Piecewise linear interpolation
    for i in range(len(snr_levels) - 1):
        if snr_levels[i] <= snr_db <= snr_levels[i + 1]:
            t = (snr_db - snr_levels[i]) / (snr_levels[i + 1] - snr_levels[i])
            return det_rates[i] + t * (det_rates[i + 1] - det_rates[i])

    return det_rates[-1]


# ═══════════════════════════════════════════════════════════════════════════
#  Localization simulation along flight path
# ═══════════════════════════════════════════════════════════════════════════

def simulate_localization_along_path(
    geo: dict,
    snr_threshold: float = -10.0,
    n_mc: int = 50,
) -> dict:
    """
    Monte Carlo incremental triangulation along the flight path.

    At each position where SNR > threshold, accumulates DOA rays with
    angular noise proportional to DOA estimation error at that SNR.
    After ≥3 rays, triangulates and records localization error.

    Returns dict with:
        det_x       : (K,) X positions where detections occurred
        det_indices : (K,) indices into geo arrays
        mean_error  : (K,) mean localization error [m] across MC runs
        std_error   : (K,) std of localization error [m]
        doa_errors  : (K,) DOA angular error at each detection position [deg]
        true_doa_vectors : list of (K,) true DOA unit vectors (mic→person)
    """
    mic_pos = geo["mic_pos"]
    person = geo["person_pos"]
    snr = geo["snr_values"]
    path_x = geo["path_x"]

    # Find detection positions (SNR > threshold)
    # If no positions exceed the threshold, fall back to peak_snr - 5 dB
    det_mask = snr > snr_threshold
    if not np.any(det_mask):
        fallback = float(snr.max()) - 5.0
        det_mask = snr > fallback
    det_indices = np.where(det_mask)[0]
    if len(det_indices) == 0:
        return {
            "det_x": np.array([]), "det_indices": np.array([]),
            "mean_error": np.array([]), "std_error": np.array([]),
            "doa_errors": np.array([]), "true_doa_vectors": [],
        }

    det_x = path_x[det_indices]

    # True DOA vectors (mic → person, normalized)
    true_doa_vectors = []
    doa_errors_deg = []
    for idx in det_indices:
        vec = person - mic_pos[idx]
        true_doa_vectors.append(vec / np.linalg.norm(vec))
        doa_errors_deg.append(estimate_doa_error(float(snr[idx])))
    doa_errors_deg = np.array(doa_errors_deg)

    # Monte Carlo: incremental triangulation
    n_det = len(det_indices)
    all_errors = np.full((n_mc, n_det), np.nan)

    rng = np.random.default_rng(42)

    for mc in range(n_mc):
        acc_points = []
        acc_vectors = []
        for k, idx in enumerate(det_indices):
            mp = mic_pos[idx]
            true_dir = true_doa_vectors[k]
            sigma_rad = np.radians(doa_errors_deg[k])

            # Add angular noise: random rotation in a cone around true_dir
            noise_az = rng.normal(0, sigma_rad)
            noise_el = rng.normal(0, sigma_rad)
            # Rotation via small-angle: perturb in local tangent plane
            # Build orthonormal basis around true_dir
            if abs(true_dir[2]) < 0.9:
                up = np.array([0, 0, 1.0])
            else:
                up = np.array([1.0, 0, 0])
            e1 = np.cross(true_dir, up)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(true_dir, e1)
            e2 /= np.linalg.norm(e2)

            noisy_dir = true_dir + noise_az * e1 + noise_el * e2
            noisy_dir /= np.linalg.norm(noisy_dir)

            acc_points.append(mp.copy())
            acc_vectors.append(noisy_dir)

            if len(acc_points) >= 3:
                try:
                    est_pos, residual, cond = triangulate_source(
                        acc_points, acc_vectors
                    )
                    err = np.linalg.norm(est_pos - person)
                    # Clamp extreme outliers (degenerate geometry)
                    all_errors[mc, k] = min(err, 500.0)
                except Exception:
                    all_errors[mc, k] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_error = np.nanmean(all_errors, axis=0)
        std_error = np.nanstd(all_errors, axis=0)

    return {
        "det_x": det_x,
        "det_indices": det_indices,
        "mean_error": mean_error,
        "std_error": std_error,
        "doa_errors": doa_errors_deg,
        "true_doa_vectors": true_doa_vectors,
    }


def simulate_localization_sliding_window(
    geo: dict,
    window_size: int = 10,
    snr_threshold: float = -10.0,
    n_mc: int = 50,
    height_sweep_data: dict | None = None,
) -> dict:
    """
    Monte Carlo **sliding-window** triangulation along the flight path.

    Unlike ``simulate_localization_along_path`` which accumulates all rays,
    this keeps only the most recent *window_size* DOA rays for triangulation.
    When the UAV flies past the source, old high-quality rays drop out of the
    window and are replaced by far-away low-quality rays, producing the
    expected U-shaped error curve.

    If *height_sweep_data* is provided, uses it for probabilistic detection
    (stochastic per-position, per-MC). Otherwise uses a hard SNR threshold.

    Returns dict with:
        path_x      : (N,)  all X positions along flight path
        all_indices : (N,)  all indices
        mean_error  : (N,)  mean localization error (NaN where < 3 rays)
        std_error   : (N,)  std of localization error
        doa_errors  : (N,)  DOA angular error at each position [deg]
        det_probs   : (N,)  detection probability at each position
        snr_values  : (N,)  SNR at each position [dB]
    """
    mic_pos = geo["mic_pos"]
    person = geo["person_pos"]
    snr = geo["snr_values"]
    path_x = geo["path_x"]
    n_pos = len(path_x)

    # Precompute DOA vectors and errors for all positions
    true_doa_vectors = []
    doa_errors_deg = np.array([estimate_doa_error(float(s)) for s in snr])
    for i in range(n_pos):
        vec = person - mic_pos[i]
        true_doa_vectors.append(vec / np.linalg.norm(vec))

    # Detection probability at each position
    if height_sweep_data is not None:
        det_probs = np.array([
            interpolate_detection_from_height_sweep(float(s), height_sweep_data)
            for s in snr
        ])
    else:
        # Hard threshold → step function
        det_probs = np.where(snr > snr_threshold, 1.0, 0.0)

    # Monte Carlo sliding-window triangulation
    all_errors = np.full((n_mc, n_pos), np.nan)
    rng = np.random.default_rng(42)

    for mc in range(n_mc):
        # Ring buffer of (point, vector) for the sliding window
        window_points = []
        window_vectors = []

        for i in range(n_pos):
            # Decide detection stochastically
            if height_sweep_data is not None:
                detected = rng.random() < det_probs[i]
            else:
                detected = snr[i] > snr_threshold

            if not detected:
                continue

            mp = mic_pos[i]
            true_dir = true_doa_vectors[i]
            sigma_rad = np.radians(doa_errors_deg[i])

            # Angular noise in local tangent plane
            noise_az = rng.normal(0, sigma_rad)
            noise_el = rng.normal(0, sigma_rad)
            if abs(true_dir[2]) < 0.9:
                up = np.array([0, 0, 1.0])
            else:
                up = np.array([1.0, 0, 0])
            e1 = np.cross(true_dir, up)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(true_dir, e1)
            e2 /= np.linalg.norm(e2)

            noisy_dir = true_dir + noise_az * e1 + noise_el * e2
            noisy_dir /= np.linalg.norm(noisy_dir)

            # Add to sliding window
            window_points.append(mp.copy())
            window_vectors.append(noisy_dir)

            # Enforce window size limit — drop oldest
            if len(window_points) > window_size:
                window_points.pop(0)
                window_vectors.pop(0)

            if len(window_points) >= 3:
                try:
                    est_pos, _, _ = triangulate_source(
                        list(window_points), list(window_vectors)
                    )
                    err = np.linalg.norm(est_pos - person)
                    all_errors[mc, i] = min(err, 500.0)
                except Exception:
                    all_errors[mc, i] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_error = np.nanmean(all_errors, axis=0)
        std_error = np.nanstd(all_errors, axis=0)

    return {
        "path_x": path_x,
        "all_indices": np.arange(n_pos),
        "mean_error": mean_error,
        "std_error": std_error,
        "doa_errors": doa_errors_deg,
        "det_probs": det_probs,
        "snr_values": snr,
        "true_doa_vectors": true_doa_vectors,
        # Compatibility fields for 3D overlay
        "det_x": path_x[det_probs > 0.01],
        "det_indices": np.where(det_probs > 0.01)[0],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  End-to-end real simulation (need.txt compliant)
# ═══════════════════════════════════════════════════════════════════════════

# Best model checkpoints from 100-MC evaluation
_BEST_CHECKPOINTS = {
    "desert": _SPEC / "results" / "sweep_desert" / "mr_0.10" / "model.pth",
    "forest": _SPEC / "results" / "sweep_forest" / "mr_0.10" / "model.pth",
}


def _load_raw_audio(terrain: str):
    """Load raw audio sources for physics-based mixing."""
    import soundfile as sf
    import librosa

    data_dir = _SPEC / "data"

    # DJI drone noise (48kHz, constant)
    drone_audio, sr = sf.read(data_dir / "drone" / "dji_sound.wav")
    if drone_audio.ndim > 1:
        drone_audio = drone_audio[:, 0]

    # Ambient noise
    ambient_dir = data_dir / "ambient" / terrain
    ambient_audios = []
    for f in sorted(ambient_dir.iterdir()):
        if f.suffix.lower() in (".wav", ".mp3", ".flac"):
            audio, _ = librosa.load(str(f), sr=sr, mono=True)
            ambient_audios.append(audio)

    # Human voice clips
    voice_dir = data_dir / "human_voice"
    voice_files = sorted(
        list(voice_dir.glob("**/*.wav")) + list(voice_dir.glob("**/*.mp3"))
    )

    return drone_audio, ambient_audios, voice_files, sr


def _random_segment(audio, n_samples, rng):
    """Extract random temporal window (need.txt: True Randomness)."""
    if len(audio) <= n_samples:
        return np.pad(audio, (0, n_samples - len(audio))).astype(np.float32)
    start = rng.integers(0, len(audio) - n_samples)
    return audio[start:start + n_samples].astype(np.float32)


def _normalize_to_dbfs(audio, target_db):
    """Normalize audio to target dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    target_rms = 10.0 ** (target_db / 20.0)
    return audio * (target_rms / rms)


def _simulate_multichannel_array(
    voice_signal: np.ndarray,
    background_signal: np.ndarray,
    uav_pos: np.ndarray,
    person_pos: np.ndarray,
    sr: int = 48000,
    array_radius: float = 0.12,
    n_mics: int = 9,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """Generate multi-channel mic array signals with physics-correct delays.

    Only the far-field voice source produces inter-mic time delays.
    The drone ego-noise is near-field (rotors < 0.5m from all mics)
    and arrives nearly simultaneously at all mics — modeled as identical
    background on each channel with small independent noise.

    Returns: (n_mics, n_samples) array
    """
    n_samples = min(len(voice_signal), len(background_signal))
    voice_signal = voice_signal[:n_samples]
    background_signal = background_signal[:n_samples]

    # Mic positions (UCA + center)
    mic_positions = np.zeros((n_mics, 3))
    mic_positions[0] = uav_pos  # center mic
    for i in range(8):
        angle = 2 * np.pi * i / 8
        mic_positions[i + 1] = uav_pos + np.array([
            array_radius * np.cos(angle),
            array_radius * np.sin(angle),
            0.0,
        ])

    # Distance from person (far-field source) to each mic
    distances = np.linalg.norm(mic_positions - person_pos[np.newaxis, :], axis=1)
    ref_dist = distances[0]  # center mic as reference

    # Time delays for voice ONLY (far-field source)
    delay_samples = np.round((distances - ref_dist) / sound_speed * sr).astype(int)

    # Generate multi-channel: delayed voice + partially-decorrelated background
    #
    # Physics: drone rotor blades are an extended source (~0.3m diameter),
    # not a point source. Each mic receives slightly different rotor noise
    # due to its position relative to the blades. Additionally, turbulent
    # airflow adds independent noise per mic. We model this as:
    #   bg_mic_m = coherent_bg * (1 - decorr) + independent_noise_m * decorr
    #
    # Decorrelation factor: multirotor UAVs generate significant turbulent
    # airflow around each mic, causing 30-50% noise decorrelation between
    # channels (Ref: Ishiki & Kumon, "Design of an auditory system for
    # a mobile robot based on DNNs with a UCA"; Wang et al., "Noise
    # Source Identification of a Multi-rotor UAV").
    decorr_factor = 0.40
    rng_array = np.random.default_rng(int(abs(uav_pos[0]) * 1000) + 7)

    signals = np.zeros((n_mics, n_samples))
    for m in range(n_mics):
        # Partially decorrelated background per mic
        indep_noise = rng_array.standard_normal(n_samples).astype(np.float32)
        indep_noise *= np.sqrt(np.mean(background_signal ** 2))  # match RMS
        signals[m] = (
            background_signal * (1.0 - decorr_factor)
            + indep_noise * decorr_factor
        )

        # Voice is time-delayed per mic (far-field source)
        d = delay_samples[m]
        if d >= 0:
            if d < n_samples:
                signals[m, d:] += voice_signal[:n_samples - d]
        else:
            ad = abs(d)
            if ad < n_samples:
                signals[m, :n_samples - ad] += voice_signal[ad:]

    return signals


def run_real_flyover_simulation(
    terrain: str,
    geo: dict,
    n_mc: int = 100,
    n_passes: int = 100,
    window_size: int = 10,
    threshold_sigma: float = 0.0,
    score_mode: str = "top_k",
    top_k_ratio: float = 0.30,
    checkpoint_override: str | None = None,
    verbose: bool = True,
) -> dict:
    """End-to-end real simulation (need.txt compliant).

    For each position along the flight path:
      1. Generate physics-correct audio (real DJI + ambient + attenuated voice)
      2. Run ACTUAL trained SpecMAE model for anomaly detection
      3. On detection: GCC-PHAT TDOA → DOA → sliding-window triangulation

    100 MC trials with different random temporal windows.

    Returns same format as simulate_localization_sliding_window().
    """
    import torch
    import librosa

    from SpecMae.scripts.eval.eval_detection_timing import load_model
    from SpecMae.scripts.utils.feature_extraction import LogMelExtractor
    from SpecMae.simulation.core.doa_calculator import DOACalculator
    from SpecMae.scripts.utils.device import get_device

    params = geo["params"]
    path_x = geo["path_x"]
    mic_pos = geo["mic_pos"]
    person = geo["person_pos"]
    snr_physics = geo["snr_values"]
    n_pos = len(path_x)

    # ── Load model ──────────────────────────────────────────────────────
    device = get_device(verbose=verbose)
    if checkpoint_override:
        ckpt_path = Path(checkpoint_override)
    else:
        ckpt_path = _BEST_CHECKPOINTS[terrain]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, mask_ratio, cfg = load_model(ckpt_path, device)
    extractor = LogMelExtractor(cfg=cfg)
    sr = cfg.sample_rate  # 48000
    n_samples_1s = sr  # 1 second of audio

    # Load fixed Dth from checkpoint (computed on full normal dataset)
    import torch as _torch
    _ckpt_data = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
    fixed_dth = _ckpt_data.get("dth", None)

    if verbose:
        if fixed_dth is not None:
            print(f"    Model loaded: {ckpt_path.name} (mr={mask_ratio:.2f}, Dth={fixed_dth:.4f})")
        else:
            print(f"    Model loaded: {ckpt_path.name} (mr={mask_ratio:.2f}, Dth=dynamic)")

    # ── Load raw audio ──────────────────────────────────────────────────
    drone_audio, ambient_audios, voice_files, _ = _load_raw_audio(terrain)
    if verbose:
        print(f"    Audio: drone={len(drone_audio)/sr:.1f}s, "
              f"ambient={len(ambient_audios)} files, "
              f"voice={len(voice_files)} clips")

    # ── Propagation model ───────────────────────────────────────────────
    prop_model = PropagationModel(terrain=terrain)

    # ── DOA setup ─────────────────────────────────────────────────────
    from SpecMae.simulation.config.hardware_config import MicArrayConfig
    mic_cfg = MicArrayConfig(n_mics=9, array_radius=0.12)
    mic_positions_local = mic_cfg.get_mic_positions()  # (9, 3) relative to center
    doa_calculator = DOACalculator(mic_positions=mic_positions_local, sound_speed=343.0)

    # ── Detection parameters (eval_detection_timing.py standard) ──────────
    THRESHOLD_SIGMA = threshold_sigma
    N_BASELINE_WINDOWS = 5  # first N positions as noise-only baseline

    # ── Subsample path (every ~4m = 40 positions) ───────────────────────
    step = max(1, n_pos // 40)
    sample_indices = np.arange(0, n_pos, step)
    n_sampled = len(sample_indices)

    if verbose:
        print(f"    Path: {n_sampled} positions (step={step}), "
              f"{n_mc} MC trials, {n_passes} passes")

    # ── MC simulation: continuous-flight paradigm ───────────────────────
    # Each MC trial: UAV flies from -80m → +80m continuously.
    # First N positions (far from person) serve as noise-only baseline.
    # Detection = score exceeds baseline + 1.5σ (same as eval_detection_timing).
    all_errors = np.full((n_mc, n_sampled), np.nan)
    all_detected = np.zeros((n_mc, n_sampled), dtype=bool)
    all_scores = np.zeros((n_mc, n_sampled))
    all_snr_actual = np.zeros((n_mc, n_sampled))
    all_doa_errors = np.full((n_mc, n_sampled), np.nan)

    for mc in range(n_mc):
        rng = np.random.default_rng(mc * 137 + 42)
        window_points = []
        window_vectors = []

        # ── Generate continuous flight audio: score all positions ───
        scores_this_mc = np.zeros(n_sampled)

        for k, idx in enumerate(sample_indices):
            pos = mic_pos[idx]
            # Physics SNR from pre-computed geometry (snr_at_distance)
            snr_db_k = snr_physics[idx]

            # Physics audio mixing (need.txt: random temporal window per trial)
            drone_seg = _normalize_to_dbfs(
                _random_segment(drone_audio, n_samples_1s, rng), -20.0
            )
            amb_src = ambient_audios[rng.integers(0, len(ambient_audios))]
            amb_seg = _normalize_to_dbfs(
                _random_segment(amb_src, n_samples_1s, rng), -25.0
            )
            background = drone_seg + amb_seg  # CONSTANT

            # Voice: RMS-ratio calibrated mixing (same as generate_long_test_audio.py)
            # Ensures actual audio SNR == physics-derived SNR
            voice_file = voice_files[rng.integers(0, len(voice_files))]
            voice_audio, _ = librosa.load(str(voice_file), sr=sr, mono=True)
            voice_seg = _random_segment(voice_audio, n_samples_1s, rng)
            voice_seg = _normalize_to_dbfs(voice_seg, -15.0)  # reference level

            bg_rms = np.sqrt(np.mean(background ** 2))
            voice_rms = np.sqrt(np.mean(voice_seg ** 2))
            if voice_rms > 1e-10 and bg_rms > 1e-10:
                target_voice_rms = bg_rms * (10.0 ** (snr_db_k / 20.0))
                voice_attenuated = voice_seg * (target_voice_rms / voice_rms)
            else:
                voice_attenuated = voice_seg * 0.0

            mixed = background + voice_attenuated
            max_val = np.abs(mixed).max()
            if max_val > 0.99:
                mixed = mixed * 0.99 / max_val

            # True SNR (retroactively computed)
            v_power = np.sum(voice_attenuated ** 2)
            bg_power = np.sum(background ** 2)
            all_snr_actual[mc, k] = 10.0 * np.log10(
                v_power / (bg_power + 1e-20)) if v_power > 0 else -np.inf

            # Stage 1: SpecMAE scoring
            spec = extractor.extract(mixed.astype(np.float32))
            spec_tensor = spec.unsqueeze(0).to(device)
            with torch.no_grad():
                score = model.compute_anomaly_score(
                    spec_tensor, mask_ratio=mask_ratio,
                    n_passes=n_passes, score_mode=score_mode,
                    top_k_ratio=top_k_ratio,
                )
            scores_this_mc[k] = float(score)
            all_scores[mc, k] = float(score)

        # ── Detection threshold ──
        if fixed_dth is not None:
            # Use pre-computed Dth from checkpoint (max of normal dataset)
            threshold = fixed_dth
            bl_mean = np.mean(scores_this_mc[:N_BASELINE_WINDOWS])
            bl_std = max(np.std(scores_this_mc[:N_BASELINE_WINDOWS]), 1e-6)
        else:
            # Fallback: dynamic baseline from first N positions
            bl_scores = scores_this_mc[:N_BASELINE_WINDOWS]
            bl_mean = np.mean(bl_scores)
            bl_std = max(np.std(bl_scores), 1e-6)
            threshold = bl_mean + THRESHOLD_SIGMA * bl_std

        # ── Detection + TDOA/DOA pass ──────────────────────────────
        for k, idx in enumerate(sample_indices):
            detected = scores_this_mc[k] > threshold
            all_detected[mc, k] = detected

            if detected:
                pos = mic_pos[idx]
                # Physics SNR for this position (same as scoring loop)
                snr_db_tdoa = snr_physics[idx]

                # Regenerate audio for TDOA
                rng_tdoa = np.random.default_rng(mc * 137 + 42 + k * 1000)
                drone_seg = _normalize_to_dbfs(
                    _random_segment(drone_audio, n_samples_1s, rng_tdoa), -20.0
                )
                amb_src = ambient_audios[rng_tdoa.integers(0, len(ambient_audios))]
                amb_seg = _normalize_to_dbfs(
                    _random_segment(amb_src, n_samples_1s, rng_tdoa), -25.0
                )
                background_tdoa = drone_seg + amb_seg
                vf = voice_files[rng_tdoa.integers(0, len(voice_files))]
                va, _ = librosa.load(str(vf), sr=sr, mono=True)
                voice_seg = _normalize_to_dbfs(
                    _random_segment(va, n_samples_1s, rng_tdoa), -15.0
                )
                # RMS-ratio calibrated mixing (physics SNR → audio SNR)
                bg_rms_t = np.sqrt(np.mean(background_tdoa ** 2))
                v_rms_t = np.sqrt(np.mean(voice_seg ** 2))
                if v_rms_t > 1e-10 and bg_rms_t > 1e-10:
                    target_v_rms = bg_rms_t * (10.0 ** (snr_db_tdoa / 20.0))
                    voice_seg = voice_seg * (target_v_rms / v_rms_t)

                # Multi-channel array (voice delayed, bg partially decorrelated)
                mc_signals = _simulate_multichannel_array(
                    voice_seg, background_tdoa, pos, person,
                    sr=sr, array_radius=0.12, n_mics=9,
                )

                # Stage 2 ANC: subtract coherent drone noise before TDOA.
                # In a real system, the drone noise profile is estimated from
                # motor RPM + stored spectral template. In simulation, we use
                # the known background signal (equivalent to perfect ANC).
                # After subtraction, only delayed voice + residual noise remain.
                for ch in range(mc_signals.shape[0]):
                    mc_signals[ch] -= background_tdoa * (1.0 - 0.40)  # remove coherent part

                try:
                    tdoa_array, peak_values = estimate_tdoa_array(
                        mc_signals, fs=sr, reference_mic=0,
                        max_delay_samples=100)
                    doa_result = doa_calculator.calculate(tdoa_array)

                    if (doa_result.direction is not None
                            and not np.any(np.isnan(doa_result.direction))):
                        doa_vec = doa_result.direction / np.linalg.norm(
                            doa_result.direction)
                        doa_global = doa_calculator.to_global_frame(
                            doa_vec, pos, uav_yaw=0.0)

                        true_dir = person - pos
                        true_dir = true_dir / np.linalg.norm(true_dir)
                        cos_a = np.clip(np.dot(doa_global, true_dir), -1, 1)
                        all_doa_errors[mc, k] = float(np.degrees(np.arccos(cos_a)))

                        window_points.append(pos.copy())
                        window_vectors.append(doa_global.copy())
                        if len(window_points) > window_size:
                            window_points.pop(0)
                            window_vectors.pop(0)
                except Exception:
                    pass

            # Triangulation at ALL positions (frozen estimate after detection zone)
            if len(window_points) >= 3:
                try:
                    est_pos, _, _ = triangulate_source(
                        list(window_points), list(window_vectors))
                    err = np.linalg.norm(est_pos - person)
                    all_errors[mc, k] = min(err, 500.0)
                except Exception:
                    all_errors[mc, k] = np.nan

        if verbose and ((mc + 1) % max(1, n_mc // 10) == 0 or mc == 0):
            valid = ~np.isnan(all_errors[mc])
            det_rate = all_detected[mc].sum() / n_sampled * 100
            print(f"      MC {mc+1}/{n_mc}: det={det_rate:.0f}%, "
                  f"tri={valid.sum()}/{n_sampled}, "
                  f"bl={bl_mean:.4f}±{bl_std:.4f}, thr={threshold:.4f}")

    # ── Aggregate (no interpolation — need.txt) ─────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_error = np.nanmean(all_errors, axis=0)
        std_error = np.nanstd(all_errors, axis=0)
        mean_doa = np.nanmean(all_doa_errors, axis=0)
        det_probs = np.mean(all_detected, axis=0)
        mean_snr = np.mean(all_snr_actual, axis=0)
        mean_scores = np.mean(all_scores, axis=0)
        std_scores = np.std(all_scores, axis=0)

    sampled_x = path_x[sample_indices]

    if verbose:
        overall_det = np.mean(all_detected) * 100
        valid_tri = (~np.isnan(all_errors)).sum() / (n_mc * n_sampled) * 100
        print(f"    Result: det_rate={overall_det:.1f}%, "
              f"triangulation_valid={valid_tri:.1f}%")

    return {
        "path_x": sampled_x,
        "all_indices": np.arange(n_sampled),
        "mean_error": mean_error,
        "std_error": std_error,
        "doa_errors": mean_doa,
        "det_probs": det_probs,
        "snr_values": mean_snr,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "true_doa_vectors": [],
        "det_x": sampled_x[det_probs > 0.01],
        "det_indices": np.where(det_probs > 0.01)[0],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Load pre-computed simulation from .mat (skip MC simulation)
# ═══════════════════════════════════════════════════════════════════════════

def load_loc_data_from_mat(terrain: str) -> dict:
    """Load pre-computed simulation results from matlab/data/*.mat.

    Returns a loc_data dict compatible with run_real_flyover_simulation().
    The .mat files are produced by matlab/export_data.py.
    """
    from scipy.io import loadmat

    mat_path = _SPEC / "matlab" / "data" / f"flyover_{terrain}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"No cached .mat file at {mat_path}. "
            f"Run: python -m SpecMae.matlab.export_data"
        )
    d = loadmat(str(mat_path), squeeze_me=True)
    path_x = np.asarray(d["sim_path_x"], dtype=float)
    det_probs = np.asarray(d["sim_det_probs"], dtype=float)
    mean_scores = np.asarray(d["sim_mean_scores"], dtype=float)
    return {
        "path_x": path_x,
        "all_indices": np.arange(len(path_x)),
        "mean_error": np.asarray(d["sim_mean_error"], dtype=float),
        "std_error": np.asarray(d["sim_std_error"], dtype=float),
        "doa_errors": np.asarray(d["sim_doa_errors"], dtype=float),
        "det_probs": det_probs,
        "mean_scores": mean_scores,
        "std_scores": np.asarray(d["sim_std_scores"], dtype=float),
        "snr_values": np.zeros_like(path_x),
        "true_doa_vectors": [],
        "det_x": path_x[det_probs > 0.01],
        "det_indices": np.where(det_probs > 0.01)[0],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  3D plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d_flyover(
    terrain: str = "desert",
    fig=None,
    ax=None,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    loc_data: Optional[dict] = None,
) -> None:
    """Render 3D perspective flyover geometry for a single terrain.

    Shows UAV flight path at altitude, person on ground, 3D distance lines,
    and height/offset annotations.
    No cable, no SNR colorbar, no truncation sphere.
    """
    import matplotlib
    if not show and fig is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

    set_publication_style()

    geo = compute_flyover_geometry(terrain)
    params = geo["params"]
    path_x = geo["path_x"]
    uav_pos = geo["uav_pos"]
    mic_pos = geo["mic_pos"]
    person = geo["person_pos"]
    d3d = geo["distances_3d"]
    snr = geo["snr_values"]
    line_idx = geo["line_indices"]

    own_fig = fig is None
    if own_fig:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection="3d")
    # If ax was passed from GridSpec, it must already be projection='3d'

    h = params.flight_height
    offset_y = params.person_offset_y

    sentinel_y_start = params.sentinel_y_start  # -4

    # ── 1. Ground plane (semi-transparent) ────────────────────────────────
    gnd_color = "#FFF8E7" if terrain == "desert" else "#E8F5E9"
    x_lo, x_hi = params.path_start, params.path_end
    gnd_corners = np.array([
        [x_lo, sentinel_y_start - 2, 0], [x_hi, sentinel_y_start - 2, 0],
        [x_hi, offset_y + 5, 0], [x_lo, offset_y + 5, 0],
    ])
    ground = Poly3DCollection([gnd_corners], alpha=0.15, facecolor=gnd_color,
                               edgecolor="gray", linewidth=0.3)
    ax.add_collection3d(ground)

    # ── 2. L-shaped flight path — sentinel Y-segment + main X-segment ────
    # Determine responder region from fixed Dth (checkpoint)
    import torch as _torch
    _ckpt_path = _BEST_CHECKPOINTS.get(terrain)
    _ckpt_data = _torch.load(_ckpt_path, map_location="cpu", weights_only=False) if _ckpt_path and _ckpt_path.exists() else {}
    if (loc_data is not None and "mean_scores" in loc_data):
        det_px = loc_data["path_x"]
        dth_val = float(_ckpt_data.get("dth", np.mean(loc_data["mean_scores"][:5]) + 1.0 * np.std(loc_data["mean_scores"][:5])))
        above_dth = loc_data["mean_scores"] > dth_val
        if np.any(above_dth):
            resp_x_min = float(det_px[above_dth][0])
            resp_x_max = float(det_px[above_dth][-1])
        else:
            resp_x_min, resp_x_max = path_x[-1], path_x[-1]
    elif (loc_data is not None and "det_probs" in loc_data
            and np.any(loc_data["det_probs"] > 0.1)):
        det_px = loc_data["path_x"]
        det_mask = loc_data["det_probs"] > 0.1
        resp_x_min = float(det_px[det_mask][0])
        resp_x_max = float(det_px[det_mask][-1])
    elif np.any(snr >= 0.0):
        resp_x_min = path_x[np.argmax(snr >= 0.0)]
        resp_x_max = path_x[len(snr) - 1 - np.argmax((snr >= 0.0)[::-1])]
    else:
        resp_x_min = path_x[-1]
        resp_x_max = path_x[-1]

    resp_idx_start = int(np.argmin(np.abs(path_x - resp_x_min)))
    resp_idx_end = int(np.argmin(np.abs(path_x - resp_x_max)))

    # Segment A: Sentinel Y-leg (takeoff, pure orange)
    # (-300, -4, h) → (-300, offset_y, h)
    n_y = 10
    y_leg = np.linspace(sentinel_y_start, offset_y, n_y)
    ax.plot(np.full(n_y, path_x[0]), y_leg, np.full(n_y, h),
            color="#E67E22", linewidth=2.5, solid_capstyle="round",
            label="Sentinel mode")

    # Start marker (green circle at takeoff point)
    ax.scatter([path_x[0]], [sentinel_y_start], [h], c="#27AE60", s=60,
               marker="o", edgecolors="white", linewidths=0.8, zorder=15,
               label="Start")

    # Segment B: Sentinel X-leg before detection (orange)
    if resp_idx_start > 0:
        seg_x = path_x[:resp_idx_start + 1]
        ax.plot(seg_x, np.full_like(seg_x, offset_y), np.full_like(seg_x, h),
                color="#E67E22", linewidth=2.5, solid_capstyle="round")

    # Segment C: Responder (blue)
    seg_x = path_x[resp_idx_start:resp_idx_end + 1]
    ax.plot(seg_x, np.full_like(seg_x, offset_y), np.full_like(seg_x, h),
            color="#2171B5", linewidth=2.5, solid_capstyle="round",
            label="Responder mode")

    # Segment D: Post-detection sentinel (orange, result frozen)
    if resp_idx_end < len(path_x) - 1:
        seg_x = path_x[resp_idx_end:]
        ax.plot(seg_x, np.full_like(seg_x, offset_y), np.full_like(seg_x, h),
                color="#E67E22", linewidth=2.5, solid_capstyle="round")

    # End marker (red square)
    ax.scatter([path_x[-1]], [offset_y], [h], c="#E74C3C", s=55,
               marker="s", edgecolors="white", linewidths=0.8, zorder=15,
               label="End")

    # Flight direction arrow (on main segment)
    arr_x1 = path_x[0] + (path_x[-1] - path_x[0]) * 0.7
    arr_x2 = path_x[0] + (path_x[-1] - path_x[0]) * 0.8
    ax.plot([arr_x1, arr_x2], [offset_y, offset_y], [h + 2.5, h + 2.5],
            color="#2171B5", linewidth=2.0)
    ax.scatter([arr_x2], [offset_y], [h + 2.5], marker=">", s=55,
              color="#2171B5", zorder=15)

    # ── 3. Person on ground ───────────────────────────────────────────────
    ax.scatter([person[0]], [person[1]], [person[2]],
               c="red", s=50, marker="o", edgecolors="darkred",
               linewidths=0.8, zorder=10, label="Person (P)")
    ax.text(person[0] + 3, person[1] - 1.5, person[2], "P (0,0,0)",
            fontsize=9, weight="bold", color="#B8860B")

    # ── 4. DOA rays — ONLY in responder region (blue segment) ────────────
    #   Evenly sample within the responder segment for DOA lines
    resp_indices = list(range(resp_idx_start, resp_idx_end + 1))
    n_lines = min(8, len(resp_indices))
    if n_lines > 0:
        step = max(1, len(resp_indices) // n_lines)
        responder_line_idx = resp_indices[::step]
    else:
        responder_line_idx = []

    annotate_set = set()
    if responder_line_idx:
        annotate_set = {responder_line_idx[0], responder_line_idx[-1]}
        if len(responder_line_idx) >= 3:
            annotate_set.add(responder_line_idx[len(responder_line_idx) // 2])

    # Stagger annotation z-offsets to avoid overlap
    annotate_list = sorted(annotate_set)
    z_offsets = {annotate_list[j]: 1.0 + j * 2.5 for j in range(len(annotate_list))}

    for idx in responder_line_idx:
        ux, uy, uz = mic_pos[idx]
        ax.plot([ux, person[0]], [uy, person[1]], [uz, person[2]],
                color="#6BAED6", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.scatter([ux], [uy], [uz], c="white", s=18, marker="o",
                   edgecolors="#4292C6", linewidths=0.6, zorder=6)

        if idx in annotate_set:
            mx = (ux + person[0]) / 2
            my = (uy + person[1]) / 2
            mz = (uz + person[2]) / 2
            z_off = z_offsets.get(idx, 1.0)
            ax.text(mx, my, mz + z_off, f"d={d3d[idx]:.0f}m",
                    fontsize=8, color="#222222", ha="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.9, edgecolor="#555555", linewidth=0.5))

    # ── 5. Height line (vertical dashed, from UAV down to ground) ────────
    ax.plot([0, 0], [offset_y, offset_y], [0, h], color="black", linewidth=1.2,
            linestyle=":", alpha=0.8)
    ax.text(0, offset_y + 1.5, h / 2, f"h={h}m", fontsize=9, color="black",
            weight="bold")

    # ── 6. (offset line removed) ─────────────────────────────────────────

    # ── 7. Axes config ────────────────────────────────────────────────────
    ax.set_xlabel("X (m)", fontsize=12, labelpad=6, color="black", weight="bold")
    ax.set_ylabel("Y (m)", fontsize=12, labelpad=6, color="black", weight="bold")
    ax.set_zlabel("Z (m)", fontsize=12, labelpad=6, color="black", weight="bold")
    ax.set_xlim(x_lo - 5, x_hi + 5)
    ax.set_ylim(sentinel_y_start - 3, offset_y + 6)
    ax.set_zlim(-1, h + 8)

    # 3D viewing angle — flight path visually aligned upper-left diagonal
    ax.view_init(elev=20, azim=-50)
    ax.tick_params(labelsize=9, colors="black", width=0.8)

    # Grid: visible on all panes
    ax.xaxis._axinfo["grid"].update(color="gray", linewidth=0.5, linestyle="-")
    ax.yaxis._axinfo["grid"].update(color="gray", linewidth=0.5, linestyle="-")
    ax.zaxis._axinfo["grid"].update(color="gray", linewidth=0.5, linestyle="-")
    ax.grid(True)

    # Pane edges (box frame) darker
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.zaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)

    env_label = "Desert" if terrain == "desert" else "Forest"
    ax.set_title(
        f"Flyover Geometry — {env_label} (h={h}m)",
        fontsize=12, weight="bold", pad=12,
    )
    ax.legend(fontsize=8, loc="upper left", labelspacing=0.5,
              borderpad=0.6, handletextpad=0.4, framealpha=0.9,
              edgecolor="black", fancybox=False, markerscale=1.5)

    if own_fig:
        plt.tight_layout()
        _save_figure(fig, save_path, dpi, show)


def plot_3d_flyover_merged(
    ax,
    geos: dict,          # {"desert": geo_dict, "forest": geo_dict}
    loc_datas: dict,      # {"desert": loc_data, "forest": loc_data}
) -> None:
    """Draw both desert and forest flyover paths in a single 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import torch as _torch

    person = np.array([0.0, 0.0, 0.0])

    # Desert = red, Forest = blue (sync with fig_Mask_Ratio)
    STYLES = {
        "desert": {"color": "#CC2222", "doa": "#DC6460", "label": "Desert"},
        "forest": {"color": "#1A4EC0", "doa": "#6090D0", "label": "Forest"},
    }

    # Global axis limits
    all_params = [geos[t]["params"] for t in geos]
    x_lo = min(p.path_start for p in all_params)
    x_hi = max(p.path_end for p in all_params)
    h_max = max(p.flight_height for p in all_params)
    offset_y = all_params[0].person_offset_y
    sy = all_params[0].sentinel_y_start

    # Ground plane
    gnd_corners = np.array([
        [x_lo, sy - 2, 0], [x_hi, sy - 2, 0],
        [x_hi, offset_y + 5, 0], [x_lo, offset_y + 5, 0],
    ])
    ground = Poly3DCollection([gnd_corners], alpha=0.08, facecolor="#F5F5DC",
                               edgecolor="gray", linewidth=0.3)
    ax.add_collection3d(ground)

    # Person P
    ax.scatter([0], [0], [0], c="#222222", s=200, marker="*",
               edgecolors="black", linewidths=0.5, zorder=10,
               label="Person (P)")
    ax.text(3, -1.5, 0, "P (0,0,0)", fontsize=12, weight="bold", color="#B8860B",
            fontfamily="Times New Roman")

    for terrain in ["desert", "forest"]:
        geo = geos[terrain]
        loc_data = loc_datas[terrain]
        sty = STYLES[terrain]
        params = geo["params"]
        h = params.flight_height
        path_x = geo["path_x"]
        mic_pos = geo["mic_pos"]
        col = sty["color"]

        # ── Full flight path as one solid line ────────────────────────────
        # Y-leg (takeoff)
        n_y = 10
        y_leg = np.linspace(sy, offset_y, n_y)
        ax.plot(np.full(n_y, path_x[0]), y_leg, np.full(n_y, h),
                color=col, linewidth=2.5, solid_capstyle="round")

        # Main X-leg (entire path, single color)
        ax.plot(path_x, np.full_like(path_x, offset_y), np.full_like(path_x, h),
                color=col, linewidth=2.5, solid_capstyle="round",
                label=f'{sty["label"]} (h={h} m)')

        # Start / End markers
        ax.scatter([path_x[0]], [sy], [h], c=col, s=60, marker="o",
                   edgecolors="white", linewidths=0.8, zorder=15)
        ax.scatter([path_x[-1]], [offset_y], [h], c=col, s=55, marker="s",
                   edgecolors="white", linewidths=0.8, zorder=15)

        # ── Detection points + DOA rays ──────────────────────────────────
        _ckpt_path = _BEST_CHECKPOINTS.get(terrain)
        _ckpt_data = (_torch.load(_ckpt_path, map_location="cpu",
                       weights_only=False)
                      if _ckpt_path and _ckpt_path.exists() else {})
        if loc_data is not None and "mean_scores" in loc_data:
            det_px = loc_data["path_x"]
            dth_val = float(_ckpt_data.get("dth", 0.0))
            det_mask = loc_data["mean_scores"] > dth_val
            det_x_pts = det_px[det_mask]

            # Pick evenly-spaced subset for DOA rays
            n_rays = min(6, len(det_x_pts))
            if n_rays > 0:
                step = max(1, len(det_x_pts) // n_rays)
                ray_xs = det_x_pts[::step]

                # Markers ONLY at ray positions
                ax.scatter(ray_xs,
                           np.full_like(ray_xs, offset_y),
                           np.full_like(ray_xs, h),
                           c=col, s=35, marker="D", edgecolors="white",
                           linewidths=0.5, zorder=12, alpha=0.9,
                           label=f'{sty["label"]} detection pts')

                # DOA rays from those points to P
                for dx in ray_xs:
                    idx = int(np.argmin(np.abs(path_x - dx)))
                    ux, uy, uz = mic_pos[idx]
                    ax.plot([ux, 0], [uy, 0], [uz, 0],
                            color=sty["doa"], linewidth=1.0,
                            linestyle="--", alpha=0.7)

        # Height annotation
        ax.plot([0, 0], [offset_y, offset_y], [0, h],
                color=col, linewidth=1.0, linestyle=":", alpha=0.6)
        side = 2.5 if terrain == "desert" else -3.5
        ax.text(side, offset_y + 1.5, h / 2, f"h={h}m",
                fontsize=12, color=col, weight="bold",
                fontfamily="Times New Roman")

    # ── Axes config ───────────────────────────────────────────────────────
    ax.set_xlabel("X (m)", fontsize=16, labelpad=10, color="black",
                  weight="bold", fontfamily="Times New Roman")
    ax.set_ylabel("Y (m)", fontsize=16, labelpad=10, color="black",
                  weight="bold", fontfamily="Times New Roman")
    ax.set_zlabel("Z (m)", fontsize=16, labelpad=10, color="black",
                  weight="bold", fontfamily="Times New Roman")
    ax.set_xlim(x_lo - 5, x_hi + 5)
    ax.set_ylim(sy - 3, offset_y + 6)
    ax.set_zlim(-1, h_max + 8)
    ax.view_init(elev=20, azim=-50)
    ax.tick_params(labelsize=12, colors="black", width=0.8)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontfamily("Times New Roman")

    # Grid on all three panes
    ax.xaxis._axinfo["grid"].update(color="gray", linewidth=0.5, linestyle="-")
    ax.yaxis._axinfo["grid"].update(color="gray", linewidth=0.5, linestyle="-")
    ax.zaxis._axinfo["grid"].update(color="gray", linewidth=0.5, linestyle="-")
    ax.grid(True)

    # Pane faces: light fill + black edges (matplotlib draws back 3 faces only)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor("black")
        pane.set_linewidth(1.0)
        pane.set_alpha(0.04)
        pane.fill = True

    # Add 3 missing back-top edges to complete the rear frame
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    zl = ax.get_zlim()
    for xs, ys, zs in [
        # left-front vertical (z-direction, at xl[0], yl[0])
        ([xl[0], xl[0]], [yl[0], yl[0]], [zl[0], zl[1]]),
        # top-left edge (y-direction, at xl[0], zl[1])
        ([xl[0], xl[0]], [yl[0], yl[1]], [zl[1], zl[1]]),
        # top-back edge (x-direction, at yl[1], zl[1])
        ([xl[0], xl[1]], [yl[1], yl[1]], [zl[1], zl[1]]),
        # right-back vertical (z-direction, at xl[1], yl[1])
        ([xl[1], xl[1]], [yl[1], yl[1]], [zl[0], zl[1]]),
    ]:
        ax.plot3D(xs, ys, zs, color="black", linewidth=1.0)

    ax.set_title("Flyover Geometry — Desert & Forest",
                  fontsize=22, weight="bold", pad=30,
                  fontfamily="Times New Roman")
    ax.legend(fontsize=12, loc="upper left", labelspacing=0.5,
              borderpad=0.8, handletextpad=0.5, framealpha=0.95,
              edgecolor="black", fancybox=False, markerscale=1.6,
              ncol=1, prop={"family": "Times New Roman", "size": 12},
              bbox_to_anchor=(0.40, 0.92))


def plot_combined_flyover(
    show: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """1x2 panel: desert (left) + forest (right), 3D view."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    fig = plt.figure(figsize=(14, 5.5))
    for idx, terrain in enumerate(["desert", "forest"]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        plot_3d_flyover(terrain=terrain, fig=fig, ax=ax, show=False)

    plt.tight_layout()
    _save_figure(fig, save_path, dpi, show)


# ═══════════════════════════════════════════════════════════════════════════
#  Right-side dual-panel plotting (SNR + detection / loc error + DOA)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_snr_detection_panel(
    ax,
    geo: dict,
    loc_data: dict,
    terrain: str,
) -> None:
    """Top-right panel: SNR U-curve + detection probability."""
    path_x = geo["path_x"]
    snr = geo["snr_values"]
    params = geo["params"]

    # ── Determine responder boundary from fixed Dth (checkpoint) ────────
    import torch as _torch
    _ckpt_path = _BEST_CHECKPOINTS.get(terrain)
    _ckpt_data = _torch.load(_ckpt_path, map_location="cpu", weights_only=False) if _ckpt_path and _ckpt_path.exists() else {}
    resp_x_start = path_x[0]
    resp_x_end = path_x[-1]
    if "mean_scores" in loc_data:
        det_px = loc_data["path_x"]
        dth = float(_ckpt_data.get("dth", np.mean(loc_data["mean_scores"][:5]) + 1.0 * np.std(loc_data["mean_scores"][:5])))
        above_dth = loc_data["mean_scores"] > dth
        if np.any(above_dth):
            resp_x_start = float(det_px[above_dth][0])
            resp_x_end = float(det_px[above_dth][-1])
    elif "det_probs" in loc_data and np.any(loc_data["det_probs"] > det_threshold):
        det_px = loc_data["path_x"]
        det_mask = loc_data["det_probs"] > det_threshold
        resp_x_start = float(det_px[det_mask][0])
        resp_x_end = float(det_px[det_mask][-1])
        dth = None
    else:
        dth = None

    # ── SNR curve (primary Y axis) ───────────────────────────────────────
    color_snr = "#14532D"
    ax.plot(path_x, snr, color=color_snr, linewidth=2.2, label="SNR")
    ax.set_ylabel("SNR (dB)", fontsize=9, color=color_snr)
    ax.tick_params(axis="y", labelcolor=color_snr, labelsize=8)
    ax.tick_params(axis="x", labelsize=8, colors="black")

    # ── Reconstruction error (secondary Y axis) ────────────────────────
    ax2 = ax.twinx()
    color_det = "#C13628"
    if "mean_scores" in loc_data:
        scores = loc_data["mean_scores"]
        det_path_x = loc_data["path_x"]
        ax2.plot(det_path_x, scores, color=color_det,
                 linewidth=1.8, linestyle="-.", label="Recon. error (SpecMAE)")
    elif "det_probs" in loc_data:
        det_probs = loc_data["det_probs"]
        det_path_x = loc_data["path_x"]
        ax2.plot(det_path_x, det_probs * 100, color=color_det,
                 linewidth=1.8, linestyle="-.", label="P(detect) %")
    # Dth threshold line
    if dth is not None:
        ax2.axhline(dth, color=color_det, linewidth=1.0, linestyle="--", alpha=0.8)
        ax2.text(params.path_end - 5, dth + 0.01, f"$D_{{th}}$={dth:.2f}",
                 fontsize=7, color=color_det, ha="right", va="bottom")
    ax2.set_ylabel("Reconstruction error", fontsize=9, color=color_det)
    ax2.tick_params(axis="y", labelcolor=color_det, labelsize=8)

    # X axis — full flight path, box frame
    ax.set_xlim(params.path_start - 5, params.path_end + 5)
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    env_label = "Desert" if terrain == "desert" else "Forest"
    ax.set_title(
        f"SNR & Detection — {env_label}",
        fontsize=10, weight="bold", pad=8,
    )


def _plot_localization_error_panel(
    ax,
    geo: dict,
    loc_data: dict,
    terrain: str,
) -> None:
    """Bottom-right panel: sliding-window localization error + DOA error."""
    params = geo["params"]

    # Determine data source: sliding-window (full path) or legacy (det_x only)
    if "all_indices" in loc_data:
        # Sliding-window mode: data covers full path
        path_x = loc_data["path_x"]
        mean_err = loc_data["mean_error"]
        std_err = loc_data["std_error"]
        doa_err = loc_data["doa_errors"]
    else:
        # Legacy mode: data only at detection positions
        path_x = loc_data["det_x"]
        mean_err = loc_data["mean_error"]
        std_err = loc_data["std_error"]
        doa_err = loc_data["doa_errors"]

    # Determine responder boundary from fixed Dth (checkpoint)
    import torch as _torch
    _ckpt_path = _BEST_CHECKPOINTS.get(terrain)
    _ckpt_data = _torch.load(_ckpt_path, map_location="cpu", weights_only=False) if _ckpt_path and _ckpt_path.exists() else {}
    if "mean_scores" in loc_data:
        dth = float(_ckpt_data.get("dth", np.mean(loc_data["mean_scores"][:5]) + 1.0 * np.std(loc_data["mean_scores"][:5])))
        above_dth = loc_data["mean_scores"] > dth
        det_px = loc_data["path_x"]
        if np.any(above_dth):
            resp_x_start = float(det_px[above_dth][0])
            resp_x_end = float(det_px[above_dth][-1])
        else:
            resp_x_start, resp_x_end = float(path_x[0]), float(path_x[-1])
    else:
        resp_x_start, resp_x_end = float(path_x[0]), float(path_x[-1])

    # Mask: only valid AND within responder zone
    valid = ~np.isnan(mean_err) & (path_x >= resp_x_start) & (path_x <= resp_x_end)
    x_v = path_x[valid]
    me_v = mean_err[valid]
    se_v = std_err[valid]

    if len(x_v) == 0:
        ax.text(0.5, 0.5, "No valid triangulations", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        return

    # ── Primary axis: localization error [m] ─────────────────────────────
    color1 = "#B91C1C"
    ax.plot(x_v, me_v, color=color1, linewidth=2.0, label="Triangulation error (m)")
    ax.set_ylabel("Triangulation error (m)", fontsize=9, color=color1)
    ax.tick_params(axis="y", labelcolor=color1, labelsize=8)
    ax.tick_params(axis="x", labelsize=8, colors="black")
    ax.legend(fontsize=7, loc="upper right")

    # 5 m target reference line
    ax.axhline(5.0, color=color1, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(
        x_v[-1], 5.3, "5 m target", fontsize=7, color=color1,
        ha="right", va="bottom", alpha=0.7,
    )

    # X axis — full flight path, box frame
    ax.set_xlabel("Flight position X (m)", fontsize=9)
    ax.set_xlim(params.path_start - 5, params.path_end + 5)
    ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    env_label = "Desert" if terrain == "desert" else "Forest"
    ax.set_title(
        f"Localization Error — {env_label}",
        fontsize=10, weight="bold", pad=8,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  1x2 layout: 3D flyover + dual error panel
# ═══════════════════════════════════════════════════════════════════════════

def plot_flyover_with_error(
    terrain: str = "desert",
    show: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
    snr_threshold: float = -10.0,
    n_mc: int = 50,
    window_size: int = 10,
    height_sweep_data: dict | None = None,
    use_real: bool = False,
    n_passes: int = 100,
    threshold_sigma: float = 0.0,
    score_mode: str = "top_k",
    top_k_ratio: float = 0.30,
    checkpoint_override: str | None = None,
) -> None:
    """
    1x2 layout: top-down flyover (left) + dual panel (right).

    Right panel contains:
      - Top: SNR U-curve + detection probability
      - Bottom: localization error (sliding-window U-curve) + DOA error
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    set_publication_style()

    geo = compute_flyover_geometry(terrain)

    if use_real:
        print(f"    Running REAL end-to-end simulation for {terrain} "
              f"(n_mc={n_mc}, n_passes={n_passes}) ...")
        loc_data = run_real_flyover_simulation(
            terrain, geo,
            n_mc=n_mc, n_passes=n_passes,
            window_size=window_size,
            threshold_sigma=threshold_sigma,
            score_mode=score_mode,
            top_k_ratio=top_k_ratio,
            checkpoint_override=checkpoint_override,
        )
    else:
        print(f"    Simulating sliding-window localization (window={window_size}, n_mc={n_mc}) ...")
        loc_data = simulate_localization_sliding_window(
            geo,
            window_size=window_size,
            snr_threshold=snr_threshold,
            n_mc=n_mc,
            height_sweep_data=height_sweep_data,
        )

    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1],
                  hspace=0.35, wspace=0.35)

    # Left: 3D flyover (spans both rows)
    ax_top = fig.add_subplot(gs[:, 0], projection="3d")
    plot_3d_flyover(terrain=terrain, fig=fig, ax=ax_top, show=False, loc_data=loc_data)

    # Top-right: SNR + detection probability
    ax_snr = fig.add_subplot(gs[0, 1])
    _plot_snr_detection_panel(ax_snr, geo, loc_data, terrain)

    # Bottom-right: localization error + DOA error
    ax_err = fig.add_subplot(gs[1, 1])
    _plot_localization_error_panel(ax_err, geo, loc_data, terrain)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.08,
                        hspace=0.35, wspace=0.30)
    _save_figure(fig, save_path, dpi, show)


def plot_combined_flyover_with_error(
    show: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
    snr_threshold: float = -10.0,
    n_mc: int = 50,
    window_size: int = 10,
    height_sweep_data_desert: dict | None = None,
    height_sweep_data_forest: dict | None = None,
    use_real: bool = False,
    n_passes: int = 100,
    threshold_sigma: float = 0.0,
    score_mode: str = "top_k",
    top_k_ratio: float = 0.30,
    checkpoint_desert: str | None = None,
    checkpoint_forest: str | None = None,
    from_mat: bool = False,
) -> None:
    """
    2x2 layout: top row = desert, bottom row = forest.

    Each row: 3D flyover (left) + dual panel (right):
      - Top-right per row: SNR + detection probability
      - Bottom-right per row: localization error + DOA

    If from_mat=True, loads pre-computed data from matlab/data/*.mat
    (no simulation, instant).
    If use_real=True, runs end-to-end simulation with actual SpecMAE model
    instead of statistical approximation.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    set_publication_style()

    sweep_data_map = {
        "desert": height_sweep_data_desert,
        "forest": height_sweep_data_forest,
    }
    ckpt_map = {
        "desert": checkpoint_desert,
        "forest": checkpoint_forest,
    }

    # ── Load data for both terrains ──────────────────────────────────────
    geos = {}
    loc_datas = {}
    for terrain in ["desert", "forest"]:
        geos[terrain] = compute_flyover_geometry(terrain)

        if from_mat:
            print(f"    Loading {terrain} from .mat cache (no simulation)...")
            loc_datas[terrain] = load_loc_data_from_mat(terrain)
        elif use_real:
            print(f"    Running REAL end-to-end simulation for {terrain} "
                  f"(n_mc={n_mc}, n_passes={n_passes}) ...")
            loc_datas[terrain] = run_real_flyover_simulation(
                terrain, geos[terrain],
                n_mc=n_mc, n_passes=n_passes,
                window_size=window_size,
                threshold_sigma=threshold_sigma,
                score_mode=score_mode,
                top_k_ratio=top_k_ratio,
                checkpoint_override=ckpt_map.get(terrain),
            )
        else:
            hs_data = sweep_data_map.get(terrain)
            print(f"    Simulating sliding-window localization for {terrain} "
                  f"(window={window_size}, n_mc={n_mc}) ...")
            loc_datas[terrain] = simulate_localization_sliding_window(
                geos[terrain],
                window_size=window_size,
                snr_threshold=snr_threshold,
                n_mc=n_mc,
                height_sweep_data=hs_data,
            )

    # ── Layout: 1 merged 3D (left) + 4 right panels ─────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, width_ratios=[1.2, 1],
                  height_ratios=[1, 1, 1, 1],
                  hspace=0.45, wspace=0.30)

    # Left: single merged 3D plot (spans all 4 rows)
    ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
    plot_3d_flyover_merged(ax_3d, geos, loc_datas)

    # Right column: 4 stacked panels
    for row, terrain in enumerate(["desert", "forest"]):
        base = row * 2

        ax_snr = fig.add_subplot(gs[base, 1])
        _plot_snr_detection_panel(ax_snr, geos[terrain],
                                  loc_datas[terrain], terrain)

        ax_err = fig.add_subplot(gs[base + 1, 1])
        _plot_localization_error_panel(ax_err, geos[terrain],
                                       loc_datas[terrain], terrain)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.06,
                        hspace=0.38, wspace=0.28)
    _save_figure(fig, save_path, dpi, show)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_figure(fig, save_path: Optional[str], dpi: int, show: bool):
    """Save figure as PNG + PDF, then show or close."""
    import matplotlib.pyplot as plt

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_p), dpi=dpi, bbox_inches="tight")
        if save_p.suffix.lower() == ".png":
            fig.savefig(str(save_p.with_suffix(".pdf")), bbox_inches="tight")
        print(f"  Figure saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="3D SNR flyover geometry figure for SPAWC paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--terrain", default="desert", choices=["desert", "forest"],
                        help="Terrain to plot")
    parser.add_argument("--combined", action="store_true",
                        help="Generate combined desert + forest panel")
    parser.add_argument("--with_error", action="store_true",
                        help="Add localization error panel (1x2 or 2x2 layout)")
    parser.add_argument("--show", action="store_true",
                        help="Display figure interactively")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI")
    parser.add_argument("--snr_threshold", type=float, default=-10.0,
                        help="SNR threshold for detection [dB] (fallback when no sweep data)")
    parser.add_argument("--n_mc", type=int, default=100,
                        help="Monte Carlo runs for error estimation")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Sliding window size for triangulation")
    parser.add_argument("--best_mr_desert", type=float, default=0.10,
                        help="Best mask_ratio for desert (V4 100-MC sweep)")
    parser.add_argument("--best_mr_forest", type=float, default=0.10,
                        help="Best mask_ratio for forest (V4 100-MC sweep)")
    parser.add_argument("--real", action="store_true",
                        help="Use REAL end-to-end simulation (loads SpecMAE model + physical audio)")
    parser.add_argument("--n_passes", type=int, default=100,
                        help="MC passes for SpecMAE anomaly scoring (--real mode)")
    parser.add_argument("--threshold_sigma", type=float, default=1.0,
                        help="Detection threshold = mean + sigma*std (--real mode)")
    parser.add_argument("--score_mode", default="top_k",
                        choices=["mean", "top_k", "max"],
                        help="Anomaly score aggregation mode (--real mode)")
    parser.add_argument("--top_k_ratio", type=float, default=0.30,
                        help="Fraction of worst-reconstructed patches (--real mode)")
    parser.add_argument("--checkpoint_desert", type=str, default=None,
                        help="Override desert model checkpoint path")
    parser.add_argument("--checkpoint_forest", type=str, default=None,
                        help="Override forest model checkpoint path")
    parser.add_argument("--from-mat", action="store_true",
                        help="Load pre-computed data from matlab/data/*.mat (no simulation, instant)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load V4 height-sweep detection data (per-scenario best mr) ──────
    hs_desert = load_height_sweep_detection_data("desert", args.best_mr_desert)
    hs_forest = load_height_sweep_detection_data("forest", args.best_mr_forest)

    if hs_desert:
        print(f"  Loaded desert height-sweep data (mr={args.best_mr_desert}): "
              f"{len(hs_desert['snr_levels'])} SNR points")
    if hs_forest:
        print(f"  Loaded forest height-sweep data (mr={args.best_mr_forest}): "
              f"{len(hs_forest['snr_levels'])} SNR points")

    if args.combined and args.with_error:
        # Choose output filename: --from-mat uses separate file to protect originals
        from_mat = getattr(args, 'from_mat', False)
        if from_mat:
            out = str(FIGURES_DIR / "fig_3d_snr_flyover_combined_error_from_mat.png")
        elif args.n_mc < 100:
            out = str(FIGURES_DIR / f"fig_3d_snr_flyover_combined_error_draft_mc{args.n_mc}.png")
        else:
            out = str(FIGURES_DIR / "fig_3d_snr_flyover_combined_error.png")
        print(f"  Generating combined 2x2 flyover+error figure...")
        plot_combined_flyover_with_error(
            show=args.show, save_path=out, dpi=args.dpi,
            snr_threshold=args.snr_threshold, n_mc=args.n_mc,
            window_size=args.window_size,
            height_sweep_data_desert=hs_desert,
            height_sweep_data_forest=hs_forest,
            use_real=args.real, n_passes=args.n_passes,
            threshold_sigma=args.threshold_sigma,
            score_mode=args.score_mode,
            top_k_ratio=args.top_k_ratio,
            checkpoint_desert=args.checkpoint_desert,
            checkpoint_forest=args.checkpoint_forest,
            from_mat=from_mat,
        )
    elif args.combined:
        out = str(FIGURES_DIR / "fig_3d_snr_flyover_combined.png")
        print(f"  Generating combined desert+forest flyover figure...")
        plot_combined_flyover(show=args.show, save_path=out, dpi=args.dpi)
    elif args.with_error:
        t = args.terrain
        out = str(FIGURES_DIR / f"fig_3d_snr_flyover_{t}_error.png")
        print(f"  Generating {t} flyover+error figure...")
        hs_for_terrain = {"desert": hs_desert, "forest": hs_forest}
        ckpt = args.checkpoint_desert if t == "desert" else args.checkpoint_forest
        plot_flyover_with_error(
            terrain=t, show=args.show, save_path=out, dpi=args.dpi,
            snr_threshold=args.snr_threshold, n_mc=args.n_mc,
            window_size=args.window_size,
            height_sweep_data=hs_for_terrain.get(t),
            use_real=args.real, n_passes=args.n_passes,
            threshold_sigma=args.threshold_sigma,
            score_mode=args.score_mode,
            top_k_ratio=args.top_k_ratio,
            checkpoint_override=ckpt,
        )
    else:
        out = str(FIGURES_DIR / f"fig_3d_snr_flyover_{args.terrain}.png")
        print(f"  Generating {args.terrain} flyover figure...")
        plot_3d_flyover(terrain=args.terrain, show=args.show, save_path=out, dpi=args.dpi)

    print("  Done.")


if __name__ == "__main__":
    main()
