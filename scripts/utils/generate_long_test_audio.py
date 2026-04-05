"""
Generate 10+ second test audio clips for sliding-window detection evaluation.

Each clip contains:
  - 12 seconds of continuous background (drone noise + environmental noise)
  - A human voice event injected at a random onset within [2.0, 9.0] seconds
  - The voice is mixed at a specified SNR relative to the background

Audio sources:
  - Drone noise:  data/dji_sound.wav (~267s, 48kHz stereo)
  - Desert env:   data/desert_environmental noise/ (3 files)
  - Forest env:   data/Forest environmental noise/ (7 MP3 files)
  - Human voice:  SpecMae/data/human_voice/
      Child_Cry_400_600Hz/   (8,639 x 1s WAV)
      Male_Rescue_100_300Hz/ (2,543 x 1s WAV)

Output:
  data/long_test/{scenario}/snr_{X}dB/
    test_{scenario}_{snr}dB_{idx:03d}.wav   (12s, 48kHz, mono)
    test_{scenario}_{snr}dB_{idx:03d}.json  (metadata)

Total: 2 scenarios x 7 SNR x 10 clips = 140 test audio files.

NOTE (V4 physics compliance):
    In the V4 evaluation pipeline, SNR values are NOT manually specified.
    They are computed by ``eval_height_sweep.py`` via
    ``PropagationModel.snr_at_distance()`` (log-distance + ISO 9613-1),
    ensuring all test data reflects physically-correct signal attenuation.
    The RMS-ratio mixing below produces audio at the physics-derived SNR.

Usage:
    python SpecMae/scripts/utils/generate_long_test_audio.py
    python SpecMae/scripts/utils/generate_long_test_audio.py --scenario desert
    python SpecMae/scripts/utils/generate_long_test_audio.py --scenario forest --n_clips 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent          # SpecMae
_PROJECT = _SPEC.parent                 # model_train_example
sys.path.insert(0, str(_PROJECT))

from SpecMae.scripts.utils.mix_audio import load_audio, normalize_audio, mix_audio_snr
from SpecMae.scripts.utils.snr_format import format_snr_tag, FINE_SNR


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

SR = 48_000
CLIP_DURATION = 12.0  # seconds
CLIP_SAMPLES  = int(SR * CLIP_DURATION)

VOICE_ONSET_MIN = 4.0   # earliest voice onset (seconds); baseline extends to 3.0s → 1s buffer
VOICE_ONSET_MAX = 9.0   # latest voice onset (seconds)
N_VOICE_CLIPS   = 2     # concatenate 2 voice clips for ~2s event

SNR_VALUES = [-10, -5, 0, 5, 10, 15, 20]

# Environment noise mixing weight relative to drone noise
ENV_WEIGHTS = {
    "desert": 0.4,   # desert: less dense ambient
    "forest": 0.6,   # forest: richer ambient soundscape
}

# Data paths (relative to _SPEC)
DRONE_PATH = _SPEC / "data" / "drone" / "dji_sound.wav"
ENV_DIRS = {
    "desert": _SPEC / "data" / "ambient" / "desert",
    "forest": _SPEC / "data" / "ambient" / "forest",
}
VOICE_DIR = _SPEC / "data" / "human_voice"
VOICE_SUBDIRS = ["Child_Cry_400_600Hz", "Male_Rescue_100_300Hz"]

OUTPUT_ROOT = _SPEC / "data" / "long_test"


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def collect_voice_files() -> list[Path]:
    """Gather all 1-second voice clips from both categories."""
    files = []
    for subdir in VOICE_SUBDIRS:
        d = VOICE_DIR / subdir
        if d.exists():
            files.extend(sorted(d.glob("*.wav")))
    if not files:
        raise FileNotFoundError(f"No voice files found under {VOICE_DIR}")
    return files


def collect_env_files(scenario: str) -> list[Path]:
    """Gather environmental noise files for a scenario."""
    d = ENV_DIRS[scenario]
    files = sorted(d.glob("*.wav")) + sorted(d.glob("*.mp3"))
    if not files:
        raise FileNotFoundError(f"No env noise files found in {d}")
    return files


def random_segment(audio: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Extract a random segment of n_samples from audio. Loop if too short."""
    if len(audio) < n_samples:
        # Tile to be long enough, then trim
        reps = (n_samples // len(audio)) + 2
        audio = np.tile(audio, reps)
    max_start = len(audio) - n_samples
    start = rng.integers(0, max(max_start, 1))
    return audio[start : start + n_samples].copy()


def load_and_cache(path: Path, cache: dict, sr: int = SR) -> np.ndarray:
    """Load audio with caching to avoid repeated disk reads."""
    key = str(path)
    if key not in cache:
        audio, _ = load_audio(str(path), sr=sr)
        cache[key] = audio
    return cache[key]


def generate_one_clip(
    drone_audio: np.ndarray,
    env_files: list[Path],
    voice_files: list[Path],
    scenario: str,
    snr_db: float,
    rng: np.random.Generator,
    audio_cache: dict,
) -> tuple[np.ndarray, dict]:
    """
    Generate a single 12-second test clip.

    Returns:
        audio: (CLIP_SAMPLES,) float32 array
        meta:  dict with onset info
    """
    # 1. Random 12s segment of drone noise
    drone_seg = random_segment(drone_audio, CLIP_SAMPLES, rng)
    drone_seg = normalize_audio(drone_seg, target_db=-20.0)

    # 2. Random 12s segment of environmental noise
    env_path = rng.choice(env_files)
    env_audio = load_and_cache(env_path, audio_cache)
    env_seg = random_segment(env_audio, CLIP_SAMPLES, rng)
    env_seg = normalize_audio(env_seg, target_db=-25.0)

    # 3. Mix background: drone + env (weighted)
    env_weight = ENV_WEIGHTS.get(scenario, 0.5)
    background = drone_seg + env_seg * env_weight

    # 4. Random voice onset
    voice_onset_sec = float(rng.uniform(VOICE_ONSET_MIN, VOICE_ONSET_MAX))
    voice_onset_sample = int(voice_onset_sec * SR)

    # 5. Select and concatenate voice clips for ~2s event
    voice_indices = rng.choice(len(voice_files), size=N_VOICE_CLIPS, replace=False)
    voice_segments = []
    voice_types = []
    for vi in voice_indices:
        vf = voice_files[vi]
        v_audio = load_and_cache(vf, audio_cache)
        voice_segments.append(v_audio)
        # Determine voice type from parent folder name
        voice_types.append(vf.parent.name)
    voice_signal = np.concatenate(voice_segments)

    # Ensure voice doesn't exceed clip boundary
    max_voice_len = CLIP_SAMPLES - voice_onset_sample
    if len(voice_signal) > max_voice_len:
        voice_signal = voice_signal[:max_voice_len]

    voice_duration_sec = len(voice_signal) / SR

    # 6. Normalize voice and mix at target SNR
    voice_signal = normalize_audio(voice_signal, target_db=-15.0)

    # Compute SNR relative to background at the voice region
    bg_region = background[voice_onset_sample : voice_onset_sample + len(voice_signal)]
    bg_rms = np.sqrt(np.mean(bg_region ** 2))
    voice_rms = np.sqrt(np.mean(voice_signal ** 2))

    if voice_rms > 1e-10 and bg_rms > 1e-10:
        # Target: voice_rms_new / bg_rms = 10^(snr_db/20)
        target_voice_rms = bg_rms * (10 ** (snr_db / 20))
        voice_gain = target_voice_rms / voice_rms
        voice_signal = voice_signal * voice_gain

    # 7. Overlay voice onto background
    mixed = background.copy()
    end_sample = voice_onset_sample + len(voice_signal)
    mixed[voice_onset_sample:end_sample] += voice_signal

    # Prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 0.99:
        mixed = mixed * 0.99 / max_val

    meta = {
        "voice_onset_sec": round(voice_onset_sec, 4),
        "voice_duration_sec": round(voice_duration_sec, 4),
        "snr_db": snr_db,
        "scenario": scenario,
        "voice_types": list(set(voice_types)),
        "n_voice_clips": N_VOICE_CLIPS,
        "clip_duration_sec": CLIP_DURATION,
        "sample_rate": SR,
        "env_file": env_path.name,
    }

    return mixed, meta


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def generate_scenario(
    scenario: str,
    n_clips: int,
    seed: int,
    snr_values: list[float] | None = None,
    output_root: Path | None = None,
) -> None:
    """Generate all test clips for one scenario."""
    snr_list = snr_values if snr_values is not None else SNR_VALUES
    out_root = output_root if output_root is not None else OUTPUT_ROOT

    print(f"\n{'='*60}")
    print(f"  Generating long test audio: {scenario}")
    print(f"  SNR values: {snr_list}")
    print(f"  Clips per SNR: {n_clips}")
    print(f"  Total clips: {len(snr_list) * n_clips}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    audio_cache: dict = {}

    # Load drone audio (single file, cache it)
    print(f"  Loading drone audio: {DRONE_PATH}")
    drone_audio, _ = load_audio(str(DRONE_PATH), sr=SR)
    print(f"  Drone audio: {len(drone_audio)/SR:.1f}s, {len(drone_audio)} samples")

    # Collect source files
    env_files = collect_env_files(scenario)
    voice_files = collect_voice_files()
    print(f"  Env noise files: {len(env_files)}")
    print(f"  Voice files: {len(voice_files)}")

    n_generated = 0
    for snr_db in snr_list:
        snr_tag = format_snr_tag(snr_db)
        out_dir = out_root / scenario / snr_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(n_clips):
            wav_name = f"test_{scenario}_{snr_tag}_{idx:03d}.wav"
            json_name = f"test_{scenario}_{snr_tag}_{idx:03d}.json"

            wav_path = out_dir / wav_name
            json_path = out_dir / json_name

            audio, meta = generate_one_clip(
                drone_audio, env_files, voice_files,
                scenario, snr_db, rng, audio_cache,
            )

            sf.write(str(wav_path), audio, SR)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)

            n_generated += 1

        print(f"  {snr_tag}: {n_clips} clips generated")

    print(f"\n  Total generated: {n_generated} clips")
    print(f"  Output: {out_root / scenario}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 12-second test audio for sliding-window detection evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default=None,
                        choices=["desert", "forest"],
                        help="Scenario to generate (default: both)")
    parser.add_argument("--snr", type=float, default=None,
                        help="Generate only this SNR level (e.g. --snr -15 or --snr -2.5)")
    parser.add_argument("--fine", action="store_true",
                        help="Use 0.5dB step SNR (71 levels, output to data/long_test_fine/)")
    parser.add_argument("--n_clips", type=int, default=10,
                        help="Number of clips per SNR level")
    parser.add_argument("--out_dir", default=None,
                        help="Override output directory (default: data/long_test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else ["desert", "forest"]
    output_root = Path(args.out_dir) if args.out_dir else None

    if args.fine:
        snr_values = FINE_SNR if args.snr is None else [args.snr]
        if output_root is None:
            output_root = _SPEC / "data" / "long_test_fine"
    elif args.snr is not None:
        snr_values = [args.snr]
    else:
        snr_values = None

    for scenario in scenarios:
        generate_scenario(
            scenario, args.n_clips, args.seed,
            snr_values=snr_values,
            output_root=output_root,
        )

    n_snrs = len(snr_values) if snr_values else len(SNR_VALUES)
    total = len(scenarios) * n_snrs * args.n_clips
    print(f"\nDone. Total: {total} test audio clips generated.")


if __name__ == "__main__":
    main()
