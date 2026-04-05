"""
Reconstruction error distribution diagnostic plot.

Validates the core assumption: anomalous audio (with human voice) produces
higher reconstruction error than normal audio (pure drone/env noise).

Loads one best model and computes MC reconstruction error distributions for:
  1. Normal training audio (200 x 1s clips)
  2. Test audio — pure noise windows (before voice onset)
  3. Test audio — voice windows (during voice event), grouped by SNR

Output:
    results/figures/fig_recon_distribution_{scenario}.png

Usage:
    python -m SpecMae.scripts.eval.plot_recon_distribution
    python -m SpecMae.scripts.eval.plot_recon_distribution --scenario desert
    python -m SpecMae.scripts.eval.plot_recon_distribution --mr 0.80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

_SPEC = Path(__file__).resolve().parents[2]  # SpecMae/
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from SpecMae.models.specmae import SpecMAE
from SpecMae.scripts.eval.eval_detection_timing import load_model, SR
from SpecMae.scripts.utils.feature_extraction import LogMelExtractor
from SpecMae.scripts.utils.mix_audio import load_audio
from SpecMae.scripts.utils.device import get_device
from SpecMae.simulation.visualization.scene_3d import set_publication_style

RESULTS_DIR = _SPEC / "results" / "figures"
TRAIN_DATA = {
    "desert": _SPEC / "data" / "generated" / "desert",
    "forest": _SPEC / "data" / "generated" / "forest",
}
TEST_DATA_ROOT = _SPEC / "data" / "test_height"
SWEEP_ROOT = _SPEC / "results"

N_TRAIN_CLIPS = 200
N_PASSES = 100
WINDOW_SEC = 1.0

# V4 100-MC sweep best mask ratios (per-scenario)
BEST_MR = {"desert": 0.10, "forest": 0.10}
WINDOW_SAMPLES = int(SR * WINDOW_SEC)


@torch.no_grad()
def score_clips(
    model: SpecMAE,
    extractor: LogMelExtractor,
    audio_segments: list[np.ndarray],
    device: torch.device,
    mask_ratio: float,
    n_passes: int = N_PASSES,
) -> list[float]:
    """Compute MC anomaly scores for a list of 1-second audio segments."""
    scores = []
    for seg in audio_segments:
        spec = extractor.extract(seg).unsqueeze(0).to(device)
        score = model.compute_anomaly_score(
            spec, mask_ratio=mask_ratio, n_passes=n_passes,
            score_mode="top_k", top_k_ratio=0.30,
        )
        scores.append(score.item())
    return scores


def collect_train_segments(scenario: str, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Load n random 1-second segments from normal training data."""
    train_dir = TRAIN_DATA[scenario]
    if not train_dir.exists():
        raise FileNotFoundError(f"Training data not found: {train_dir}")

    # Collect all normal WAV files (structure: desert/desert/train/normal/*.wav)
    wav_files = sorted(train_dir.glob("**/train/normal/*.wav"))
    if not wav_files:
        wav_files = sorted(train_dir.glob("**/normal/*.wav"))
    if not wav_files:
        wav_files = sorted(train_dir.glob("**/*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {train_dir}")

    segments = []
    indices = rng.choice(len(wav_files), size=min(n, len(wav_files)), replace=False)
    for idx in indices:
        audio, _ = load_audio(str(wav_files[idx]), sr=SR)
        if len(audio) >= WINDOW_SAMPLES:
            segments.append(audio[:WINDOW_SAMPLES])
        elif len(audio) > 0:
            padded = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            padded[:len(audio)] = audio
            segments.append(padded)
        if len(segments) >= n:
            break
    return segments


def _find_scenario_snr_dirs(scenario: str) -> list[Path]:
    """Find all snr_* directories containing test WAVs for a scenario.

    Handles two layouts:
      - long_test_height: root/h_XXm/{scenario}/snr_*/
      - long_test:        root/{scenario}/snr_*/
    """
    candidates = []

    roots = []
    for subdir in ("test_height", "long_test"):
        p = _SPEC / "data" / subdir
        if p.exists():
            roots.append(p)

    for root in roots:
        # Layout 1: root/h_XXm/scenario/snr_*
        for h_dir in sorted(root.glob("h_*")):
            sc_dir = h_dir / scenario
            if sc_dir.is_dir():
                candidates.extend(sorted(sc_dir.iterdir()))
        # Layout 2: root/scenario/snr_*
        sc_dir = root / scenario
        if sc_dir.is_dir():
            candidates.extend(sorted(sc_dir.iterdir()))

    return [d for d in candidates if d.is_dir()]


def collect_test_segments(scenario: str, max_clips: int = 50) -> tuple[list[np.ndarray], list[np.ndarray], dict[float, list[np.ndarray]]]:
    """Extract noise-only and voice windows from long test audio.

    Returns:
        noise_segments: 1-second windows from before voice onset
        voice_segments: 1-second windows during voice event
        voice_by_snr: {snr_db: [segments]}
    """
    noise_segments = []
    voice_segments = []
    voice_by_snr: dict[float, list[np.ndarray]] = {}

    snr_dirs = _find_scenario_snr_dirs(scenario)
    if not snr_dirs:
        return noise_segments, voice_segments, voice_by_snr

    for snr_dir in snr_dirs:
        for wav_path in sorted(snr_dir.glob("*.wav"))[:max_clips]:
            json_path = wav_path.with_suffix(".json")
            if not json_path.exists():
                continue
            with open(json_path) as f:
                meta = json.load(f)

            audio, _ = load_audio(str(wav_path), sr=SR)
            onset = meta["voice_onset_sec"]
            snr = meta["snr_db"]

            # Noise window: 1 second starting at t=1.0 (well before any voice)
            noise_start = int(1.0 * SR)
            if noise_start + WINDOW_SAMPLES <= int(onset * SR):
                noise_segments.append(audio[noise_start:noise_start + WINDOW_SAMPLES])

            # Voice window: 1 second starting at voice onset
            voice_start = int(onset * SR)
            if voice_start + WINDOW_SAMPLES <= len(audio):
                seg = audio[voice_start:voice_start + WINDOW_SAMPLES]
                voice_segments.append(seg)
                voice_by_snr.setdefault(snr, []).append(seg)

    return noise_segments, voice_segments, voice_by_snr


def find_best_model(scenario: str, target_mr: float | None = None) -> Path:
    """Find checkpoint for target mask ratio (default: V4 100-MC best)."""
    sweep_dir = SWEEP_ROOT / f"sweep_{scenario}"

    # Use explicit target, or fall back to per-scenario best
    mr = target_mr if target_mr is not None else BEST_MR.get(scenario)
    if mr is not None:
        mr_dir = sweep_dir / f"mr_{mr:.2f}"
        if mr_dir.exists():
            return mr_dir / "model.pth"

    # Last resort: pick first available
    for mr_dir in sorted(sweep_dir.glob("mr_*")):
        ckpt = mr_dir / "model.pth"
        if ckpt.exists():
            return ckpt
    raise FileNotFoundError(f"No checkpoints found in {sweep_dir}")


def plot_distribution(
    scenario: str,
    train_scores: list[float],
    noise_scores: list[float],
    voice_scores: list[float],
    voice_by_snr_scores: dict[float, list[float]],
    mask_ratio: float,
    out_dir: Path | None = None,
) -> Path:
    """Create box+violin diagnostic plot."""
    set_publication_style()

    if out_dir is None:
        out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Left panel: box plot of 3 categories ─────────────────────────────
    ax = axes[0]
    data = [train_scores, noise_scores, voice_scores]
    labels = [
        f"Normal\n(train, n={len(train_scores)})",
        f"Noise-only\n(test, n={len(noise_scores)})",
        f"Voice+Noise\n(test, n={len(voice_scores)})",
    ]
    colors_box = ["#2ecc71", "#3498db", "#e74c3c"]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5),
                    showfliers=False)
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Reconstruction Error")
    ax.set_title(f"Error Distribution — {scenario.capitalize()} (mr={mask_ratio:.2f})",
                 fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)

    # ── Right panel: voice scores by SNR ─────────────────────────────────
    ax2 = axes[1]
    snrs = sorted(voice_by_snr_scores.keys())
    if snrs:
        snr_data = [voice_by_snr_scores[s] for s in snrs]
        snr_labels = [f"{s:+.0f}" for s in snrs]
        bp2 = ax2.boxplot(snr_data, tick_labels=snr_labels, patch_artist=True, widths=0.5,
                          medianprops=dict(color="black", linewidth=1.5),
                          showfliers=False)
        # Color gradient: low SNR = red, high SNR = green
        cmap = plt.cm.RdYlGn
        for i, patch in enumerate(bp2["boxes"]):
            patch.set_facecolor(cmap(i / max(len(snrs) - 1, 1)))
            patch.set_alpha(0.6)

        # Add noise baseline
        if noise_scores:
            noise_median = float(np.median(noise_scores))
            ax2.axhline(noise_median, color="#3498db", linestyle="--", linewidth=1.5,
                        label=f"Noise baseline (median={noise_median:.4f})")
            ax2.legend(fontsize=8, loc="upper left")

    ax2.set_xlabel("Target SNR (dB)")
    ax2.set_ylabel("Reconstruction Error")
    ax2.set_title("Voice Error by SNR", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.2)

    fig.tight_layout()

    path = out_dir / f"fig_recon_distribution_{scenario}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def run_scenario(scenario: str, target_mr: float | None = None, out_dir: Path | None = None):
    """Full diagnostic pipeline for one scenario."""
    print(f"\n{'='*60}")
    print(f"  Reconstruction Error Diagnostic: {scenario.upper()}")
    print(f"{'='*60}")

    device = get_device(verbose=True)
    rng = np.random.default_rng(42)

    # Load model
    ckpt_path = find_best_model(scenario, target_mr)
    print(f"  Model: {ckpt_path}")
    model, mask_ratio, cfg = load_model(ckpt_path, device)
    extractor = LogMelExtractor(cfg=cfg)
    print(f"  mask_ratio={mask_ratio:.2f}")

    # Collect segments
    print(f"  Collecting training segments...")
    train_segs = collect_train_segments(scenario, N_TRAIN_CLIPS, rng)
    print(f"  Training segments: {len(train_segs)}")

    print(f"  Collecting test segments...")
    noise_segs, voice_segs, voice_by_snr = collect_test_segments(scenario)
    print(f"  Noise segments: {len(noise_segs)}")
    print(f"  Voice segments: {len(voice_segs)}")
    print(f"  SNR levels: {sorted(voice_by_snr.keys())}")

    if not noise_segs and not voice_segs:
        print("  WARNING: No test segments found. Run eval_height_sweep.py first to generate data.")
        print("  Falling back to training-only diagnostic...")

    # Score
    print(f"  Scoring training clips (n={len(train_segs)}, {N_PASSES} MC passes)...")
    train_scores = score_clips(model, extractor, train_segs, device, mask_ratio)

    noise_scores = []
    if noise_segs:
        print(f"  Scoring noise clips (n={len(noise_segs)})...")
        noise_scores = score_clips(model, extractor, noise_segs, device, mask_ratio)

    voice_scores = []
    if voice_segs:
        print(f"  Scoring voice clips (n={len(voice_segs)})...")
        voice_scores = score_clips(model, extractor, voice_segs, device, mask_ratio)

    voice_by_snr_scores: dict[float, list[float]] = {}
    for snr, segs in voice_by_snr.items():
        voice_by_snr_scores[snr] = score_clips(model, extractor, segs, device, mask_ratio)

    # Summary stats
    print(f"\n  {'Category':<20s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, scores in [("Train normal", train_scores), ("Test noise", noise_scores), ("Test voice", voice_scores)]:
        if scores:
            print(f"  {name:<20s}  {np.mean(scores):8.4f}  {np.median(scores):8.4f}  {np.std(scores):8.4f}")

    # Verify core assumption
    if noise_scores and voice_scores:
        noise_med = np.median(noise_scores)
        voice_med = np.median(voice_scores)
        if voice_med > noise_med:
            print(f"\n  PASS: Voice error ({voice_med:.4f}) > Noise error ({noise_med:.4f})")
        else:
            print(f"\n  WARN: Voice error ({voice_med:.4f}) <= Noise error ({noise_med:.4f})")

    # Plot
    path = plot_distribution(
        scenario, train_scores, noise_scores, voice_scores,
        voice_by_snr_scores, mask_ratio, out_dir,
    )
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruction error distribution diagnostic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default=None, choices=["desert", "forest"],
                        help="Scenario (default: both)")
    parser.add_argument("--mr", type=float, default=None,
                        help="Target mask ratio (e.g. 0.80)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override output directory")

    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None

    scenarios = [args.scenario] if args.scenario else ["desert", "forest"]
    for scenario in scenarios:
        mr = args.mr if args.mr is not None else BEST_MR.get(scenario)
        run_scenario(scenario, target_mr=mr, out_dir=out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
