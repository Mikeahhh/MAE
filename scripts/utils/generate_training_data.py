"""
Generate training, validation, and test data for V4.

Uses DJI drone noise (48kHz) + scenario-specific ambient noise.
All clips are 1 second at 48kHz, mono, PCM_16.

Training/Validation: normal only (drone + ambient)
Test: normal + anomaly (drone + ambient + human voice at various SNRs)

Usage:
    python -m SpecMae.scripts.utils.generate_training_data
    python -m SpecMae.scripts.utils.generate_training_data --scenario desert
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_SPEC = Path(__file__).resolve().parents[2]  # SpecMae/
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from SpecMae.simulation.core.propagation_model import (
    PropagationModel,
    DEFAULT_SOURCE_SPL,
    DEFAULT_DRONE_NOISE_SPL,
    DEFAULT_HORIZONTAL_OFFSET,
)
from SpecMae.scripts.utils.snr_format import format_height_tag

# ── Config ────────────────────────────────────────────────────────────
SR = 48000
TARGET_SAMPLES = SR  # 1 second

N_TRAIN = 1000
N_VAL = 200
N_TEST_NORMAL = 200
N_TEST_ANOMALY_PER_SNR = 30
TEST_SNRS = [-15, -10, -5, 0, 5, 10, 15, 20]

# Height-based test configs (whiteboard)
HEIGHT_CONFIGS: dict[str, list[int]] = {
    "desert": [5, 10, 15, 20],
    "forest": [15, 20, 35, 50],
}
N_TEST_ANOMALY_PER_HEIGHT = 50

DRONE_PATH = _SPEC / "data" / "drone" / "dji_sound.wav"
VOICE_DIRS = [
    _SPEC / "data" / "human_voice" / "Child_Cry_400_600Hz",
    _SPEC / "data" / "human_voice" / "Male_Rescue_100_300Hz",
]

AMBIENT_DIRS = {
    "desert": _SPEC / "data" / "ambient" / "desert",
    "forest": _SPEC / "data" / "ambient" / "forest",
}

OUTPUT_DIRS = {
    "desert": _SPEC / "data" / "generated" / "desert",
    "forest": _SPEC / "data" / "generated" / "forest",
}

# ── Audio utilities ───────────────────────────────────────────────────

def load_clip(path: Path) -> np.ndarray:
    """Load a 1-second 48kHz WAV clip."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert len(audio) == TARGET_SAMPLES, f"Expected {TARGET_SAMPLES} samples, got {len(audio)} in {path}"
    return audio.astype(np.float32)


def load_long(path: Path) -> np.ndarray:
    """Load an arbitrarily long audio file, convert to mono float32 at SR."""
    audio, file_sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
    return audio.astype(np.float32)


def random_segment(audio: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Extract a random segment of n_samples from a long audio array."""
    if len(audio) <= n_samples:
        return np.pad(audio, (0, n_samples - len(audio))).astype(np.float32)
    start = rng.integers(0, len(audio) - n_samples)
    return audio[start : start + n_samples].copy()


def rms(audio: np.ndarray) -> float:
    r = np.sqrt(np.mean(audio ** 2))
    return r if r > 1e-10 else 1e-10


def active_rms(audio: np.ndarray, frame_len: int = 512, top_frac: float = 0.3) -> float:
    """Active-segment RMS: top fraction of most energetic frames."""
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return rms(audio)
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    n_active = max(1, int(n_frames * top_frac))
    active = np.sort(frame_rms)[-n_active:]
    r = float(np.sqrt(np.mean(active ** 2)))
    return r if r > 1e-10 else 1e-10


def normalize_to_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    target_rms = 10 ** (target_dbfs / 20.0)
    return audio * (target_rms / rms(audio))


def mix_snr(background: np.ndarray, signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal into background at target SNR (using active_rms for signal)."""
    bg_rms = rms(background)
    sig_active = active_rms(signal)
    target_sig_rms = bg_rms * (10 ** (snr_db / 20.0))
    signal_scaled = signal * (target_sig_rms / sig_active)
    return background + signal_scaled


def peak_normalize(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return audio
    limit = 10 ** (headroom_db / 20.0)
    if peak > limit:
        audio = audio * (limit / peak)
    return audio


def save(audio: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_16")


# ── Data generation ───────────────────────────────────────────────────

def make_background(
    drone_audio: np.ndarray,
    ambient_audios: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Random 1s segment from drone + random 1s from a random ambient file."""
    drone = normalize_to_dbfs(random_segment(drone_audio, TARGET_SAMPLES, rng), -20.0)
    amb_src = ambient_audios[rng.integers(0, len(ambient_audios))]
    ambient = normalize_to_dbfs(random_segment(amb_src, TARGET_SAMPLES, rng), -25.0)
    return drone + ambient


def generate_normal(
    drone_audio: np.ndarray, ambient_audios: list[np.ndarray],
    rng: np.random.Generator, out_dir: Path, n: int, prefix: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        bg = peak_normalize(make_background(drone_audio, ambient_audios, rng))
        save(bg, out_dir / f"{prefix}_{i:05d}.wav")


def generate_anomaly(
    drone_audio: np.ndarray, ambient_audios: list[np.ndarray],
    voice_files: list, rng: np.random.Generator,
    out_dir: Path, snr_db: float, n: int,
):
    """DEPRECATED (V3 legacy): uses RMS-ratio mixing with hardcoded SNR labels.
    V4 evaluation uses generate_anomaly_height() with physics-based attenuation.
    Kept for backward compatibility with V3 test data only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        bg = make_background(drone_audio, ambient_audios, rng)
        voice = load_clip(voice_files[rng.integers(0, len(voice_files))])
        mixed = mix_snr(bg, voice, snr_db)
        mixed = peak_normalize(mixed)
        save(mixed, out_dir / f"anomaly_snr{snr_db:+.0f}dB_{i:05d}.wav")


def generate_anomaly_height(
    drone_audio: np.ndarray, ambient_audios: list[np.ndarray],
    voice_files: list, rng: np.random.Generator,
    out_dir: Path, height_m: int, scenario: str, n: int,
):
    """Generate anomaly test data at a specific flight height using physics SNR.

    Uses RMS-ratio calibrated mixing (consistent with generate_long_test_audio.py)
    to ensure actual audio SNR matches physics-derived SNR from PropagationModel.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = PropagationModel(terrain=scenario)
    snr_db = model.snr_at_distance(
        distance=DEFAULT_HORIZONTAL_OFFSET,
        flight_height=float(height_m),
    )
    print(f"    h={height_m}m: SNR={snr_db:+.1f}dB (physics-derived)")
    for i in range(n):
        bg = make_background(drone_audio, ambient_audios, rng)
        voice = load_clip(voice_files[rng.integers(0, len(voice_files))])
        # RMS-ratio calibrated mixing: actual audio SNR = physics SNR
        voice = normalize_to_dbfs(voice, -15.0)
        bg_rms_val = rms(bg)
        v_rms_val = rms(voice)
        target_v_rms = bg_rms_val * (10.0 ** (snr_db / 20.0))
        voice_calibrated = voice * (target_v_rms / v_rms_val)
        mixed = bg + voice_calibrated
        mixed = peak_normalize(mixed)
        save(mixed, out_dir / f"anomaly_h{height_m}m_{i:05d}.wav")


def generate_scenario(scenario: str, seed: int = 42):
    """Generate all data for one scenario."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  Generating {scenario.upper()} data (DJI drone noise)")
    print(f"{'='*60}")

    # Load drone audio (single long file)
    print(f"  Loading drone audio: {DRONE_PATH}")
    drone_audio = load_long(DRONE_PATH)
    print(f"    Duration: {len(drone_audio)/SR:.1f}s ({len(drone_audio)} samples)")

    # Load ambient audio (long files)
    amb_dir = AMBIENT_DIRS[scenario]
    ambient_files = sorted(amb_dir.glob("*.wav")) + sorted(amb_dir.glob("*.mp3"))
    ambient_audios = [load_long(f) for f in ambient_files]
    print(f"    Ambient files: {len(ambient_audios)}")

    # Load voice clips (pre-cut 1s WAVs)
    voice_files = []
    for vd in VOICE_DIRS:
        voice_files += sorted(vd.glob("*.wav"))

    print(f"  Sources:")
    print(f"    DJI drone: 1 file, {len(drone_audio)/SR:.1f}s")
    print(f"    Ambient:   {len(ambient_audios)} long files")
    print(f"    Voice:     {len(voice_files)} clips")

    if not DRONE_PATH.exists():
        raise FileNotFoundError(f"Drone file not found: {DRONE_PATH}")
    if not ambient_audios:
        raise FileNotFoundError(f"No ambient files in {amb_dir}")
    if not voice_files:
        raise FileNotFoundError(f"No voice clips")

    out_root = OUTPUT_DIRS[scenario]

    # Training
    print(f"\n  [1/3] Training set ({N_TRAIN} normal)")
    generate_normal(drone_audio, ambient_audios, rng,
                    out_root / "train" / "normal", N_TRAIN, "normal")
    print(f"    → {out_root / 'train' / 'normal'}")

    # Validation
    print(f"  [2/3] Validation set ({N_VAL} normal)")
    generate_normal(drone_audio, ambient_audios, rng,
                    out_root / "val" / "normal", N_VAL, "normal")
    print(f"    → {out_root / 'val' / 'normal'}")

    # Test
    print(f"  [3/3] Test set")
    generate_normal(drone_audio, ambient_audios, rng,
                    out_root / "test" / "normal", N_TEST_NORMAL, "normal")
    print(f"    Normal: {N_TEST_NORMAL} → {out_root / 'test' / 'normal'}")

    for snr in TEST_SNRS:
        generate_anomaly(
            drone_audio, ambient_audios, voice_files, rng,
            out_root / "test" / "anomaly" / f"snr_{snr:+.0f}dB",
            snr_db=snr, n=N_TEST_ANOMALY_PER_SNR,
        )
        print(f"    Anomaly SNR={snr:+.0f}dB: {N_TEST_ANOMALY_PER_SNR}")

    # Height-based test anomaly (physics attenuation)
    heights = HEIGHT_CONFIGS.get(scenario, [])
    if heights:
        print(f"\n  [4/4] Height-based anomaly test (physics attenuation)")
        for h in heights:
            h_tag = format_height_tag(h)
            generate_anomaly_height(
                drone_audio, ambient_audios, voice_files, rng,
                out_root / "test" / "anomaly" / h_tag,
                height_m=h, scenario=scenario, n=N_TEST_ANOMALY_PER_HEIGHT,
            )

    n_height = len(heights) * N_TEST_ANOMALY_PER_HEIGHT
    total = N_TRAIN + N_VAL + N_TEST_NORMAL + len(TEST_SNRS) * N_TEST_ANOMALY_PER_SNR + n_height
    print(f"\n  Total: {total} clips for {scenario}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate V4 training data with DJI drone noise",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default=None, choices=["desert", "forest"],
                        help="Generate one scenario (default: both)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else ["desert", "forest"]
    for s in scenarios:
        generate_scenario(s, seed=args.seed)

    print("\nAll done.")


if __name__ == "__main__":
    main()
