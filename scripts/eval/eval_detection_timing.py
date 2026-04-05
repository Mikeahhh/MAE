"""
Sliding-window detection and timing evaluation on long test audio.

Pure reconstruction-based anomaly detection (PatchKNN removed).
For each 12-second test clip:
  1. Split into overlapping 1-second windows (hop = 0.5s)
  2. Score each window via MC-averaged reconstruction error
  3. Apply EMA-smoothed adaptive threshold with consecutive-trigger confirmation
  4. Compare detected onset vs ground truth

Usage:
    # Evaluate one model
    python SpecMae/scripts/eval/eval_detection_timing.py \
        --checkpoint results/sweep_desert/mr_0.75/model.pth \
        --scenario desert

    # Evaluate all sweep models for a scenario
    python SpecMae/scripts/eval/eval_detection_timing.py \
        --sweep_dir results/sweep_desert \
        --scenario desert

    # Evaluate all scenarios
    python SpecMae/scripts/eval/eval_detection_timing.py --all

    # Use different model size
    python SpecMae/scripts/eval/eval_detection_timing.py \
        --checkpoint results/train_desert/checkpoints/best_model.pth \
        --scenario desert --model_size tiny
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from SpecMae.models.specmae import SpecMAE, get_model_factory
from SpecMae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor
from SpecMae.scripts.utils.device import get_device
from SpecMae.scripts.utils.mix_audio import load_audio
from SpecMae.scripts.utils.snr_format import format_snr_tag


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

SR = 48_000
WINDOW_SEC = 1.0
HOP_SEC    = 0.5
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
HOP_SAMPLES    = int(SR * HOP_SEC)

# Detection parameters
BASELINE_WINDOWS = 5       # first 5 positions for noise-only baseline
THRESHOLD_SIGMA  = 1.0     # threshold = mean + sigma * std
DETECTION_TOLERANCE = 2.0  # seconds tolerance for correct detection
EMA_ALPHA        = 0.3     # EMA smoothing coefficient
CONSECUTIVE_TRIGGER = 1    # consecutive windows above threshold to confirm

# Reconstruction scoring defaults
RECON_N_PASSES   = 100     # MC passes per window (100 trials per need.txt)
RECON_SCORE_MODE = "top_k"
RECON_TOP_K_RATIO = 0.30   # top 30% worst-reconstructed patches (optimized via sweep)

TEST_DATA_ROOT = _SPEC / "data" / "long_test"  # generate_long_test_audio.py output
RESULTS_ROOT   = _SPEC / "results"


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(
    ckpt_path: Path,
    device: torch.device,
    model_size: str = "base",
) -> tuple[SpecMAE, float, AudioConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, mask_ratio, cfg


# ═══════════════════════════════════════════════════════════════════════════
#  Reconstruction-based sliding window scoring
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_window_recon_scores(
    model: SpecMAE,
    audio: np.ndarray,
    extractor: LogMelExtractor,
    device: torch.device,
    mask_ratio: float,
    n_passes: int = RECON_N_PASSES,
    score_mode: str = RECON_SCORE_MODE,
    top_k_ratio: float = RECON_TOP_K_RATIO,
) -> tuple[list[float], list[float]]:
    """
    Compute MC-averaged reconstruction anomaly scores for sliding windows.

    Returns:
        scores: list of anomaly scores per window
        window_times: list of window center times (seconds)
    """
    n_samples = len(audio)
    scores = []
    window_times = []

    start = 0
    while start + WINDOW_SAMPLES <= n_samples:
        window = audio[start : start + WINDOW_SAMPLES]
        spec = extractor.extract(window).unsqueeze(0).to(device)  # (1, 1, F, T)

        score = model.compute_anomaly_score(
            spec, mask_ratio=mask_ratio, n_passes=n_passes,
            score_mode=score_mode, top_k_ratio=top_k_ratio,
        )
        scores.append(score.item())

        center_time = (start + WINDOW_SAMPLES / 2) / SR
        window_times.append(center_time)

        start += HOP_SAMPLES

    return scores, window_times


@torch.no_grad()
def extract_window_multiscale_scores(
    model: SpecMAE,
    audio: np.ndarray,
    extractor: LogMelExtractor,
    device: torch.device,
    mask_ratios: tuple[float, ...] = (0.3, 0.5, 0.7),
    n_passes: int = 20,
    score_mode: str = "top_k",
    top_k_ratio: float = 0.15,
) -> tuple[list[float], list[float]]:
    """
    Multi-scale MC-averaged reconstruction scoring for sliding windows.

    Returns:
        scores: list of fused anomaly scores per window
        window_times: list of window center times (seconds)
    """
    n_samples = len(audio)
    scores = []
    window_times = []

    start = 0
    while start + WINDOW_SAMPLES <= n_samples:
        window = audio[start : start + WINDOW_SAMPLES]
        spec = extractor.extract(window).unsqueeze(0).to(device)

        score = model.compute_multiscale_anomaly_score(
            spec, mask_ratios=mask_ratios, n_passes=n_passes,
            score_mode=score_mode, top_k_ratio=top_k_ratio,
        )
        scores.append(score.item())

        center_time = (start + WINDOW_SAMPLES / 2) / SR
        window_times.append(center_time)

        start += HOP_SAMPLES

    return scores, window_times


# ═══════════════════════════════════════════════════════════════════════════
#  Detection logic (improved)
# ═══════════════════════════════════════════════════════════════════════════

def ema_smooth(scores: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    """Exponential moving average smoothing."""
    smoothed = []
    for i, s in enumerate(scores):
        if i == 0:
            smoothed.append(s)
        else:
            smoothed.append(alpha * s + (1 - alpha) * smoothed[-1])
    return smoothed


def detect_onset(
    scores: list[float],
    window_times: list[float],
    n_baseline: int = BASELINE_WINDOWS,
    sigma: float = THRESHOLD_SIGMA,
    use_ema: bool = True,
    consecutive: int = CONSECUTIVE_TRIGGER,
) -> tuple[float | None, float]:
    """
    Adaptive threshold detection with EMA smoothing and consecutive trigger.

    Uses one-sided threshold (anomaly = higher reconstruction error).
    Requires `consecutive` windows above threshold to confirm detection.

    Returns:
        detected_onset: time in seconds (or None if no detection)
        threshold: the computed threshold value
    """
    if len(scores) < n_baseline + 1:
        return None, 0.0

    # Apply EMA smoothing
    if use_ema:
        scores = ema_smooth(scores)

    baseline = scores[:n_baseline]
    bl_mean = np.mean(baseline)
    bl_std = max(np.std(baseline), 1e-6)
    threshold = bl_mean + sigma * bl_std

    # Consecutive trigger: need `consecutive` windows above threshold
    above_count = 0
    for i in range(n_baseline, len(scores)):
        if scores[i] > threshold:
            above_count += 1
            if above_count >= consecutive:
                # Report the first window that crossed
                onset_idx = i - consecutive + 1
                return window_times[onset_idx], threshold
        else:
            above_count = 0

    return None, threshold


def detect_onset_offset(
    scores: list[float],
    window_times: list[float],
    n_baseline: int = BASELINE_WINDOWS,
    sigma: float = THRESHOLD_SIGMA,
    use_ema: bool = True,
    consecutive: int = CONSECUTIVE_TRIGGER,
) -> tuple[float | None, float | None, float]:
    """
    Detect onset (score rises above threshold) and offset (score falls back).

    Returns:
        detected_onset: time in seconds (or None)
        detected_offset: time in seconds (or None if no offset / no onset)
        threshold: the computed threshold value
    """
    if len(scores) < n_baseline + 1:
        return None, None, 0.0

    if use_ema:
        scores = ema_smooth(scores)

    baseline = scores[:n_baseline]
    bl_mean = np.mean(baseline)
    bl_std = max(np.std(baseline), 1e-6)
    threshold = bl_mean + sigma * bl_std

    # Find onset
    onset_idx = None
    above_count = 0
    for i in range(n_baseline, len(scores)):
        if scores[i] > threshold:
            above_count += 1
            if above_count >= consecutive and onset_idx is None:
                onset_idx = i - consecutive + 1
        else:
            if onset_idx is not None:
                break
            above_count = 0

    if onset_idx is None:
        return None, None, threshold

    onset_time = window_times[onset_idx]

    # Find offset: first window after onset where score drops below threshold
    below_count = 0
    offset_idx = None
    for i in range(onset_idx + consecutive, len(scores)):
        if scores[i] <= threshold:
            below_count += 1
            if below_count >= consecutive:
                offset_idx = i - consecutive + 1
                break
        else:
            below_count = 0

    offset_time = window_times[offset_idx] if offset_idx is not None else window_times[-1]

    return onset_time, offset_time, threshold


def evaluate_one_clip(
    audio_path: Path,
    meta_path: Path,
    model: SpecMAE,
    extractor: LogMelExtractor,
    device: torch.device,
    mask_ratio: float = 0.75,
    n_passes: int = RECON_N_PASSES,
    use_multiscale: bool = False,
) -> dict:
    """Evaluate detection on a single 12-second clip using pure reconstruction."""
    audio, _ = load_audio(str(audio_path), sr=SR)

    with open(meta_path) as f:
        meta = json.load(f)
    true_onset = meta["voice_onset_sec"]
    true_duration = meta.get("voice_duration_sec")

    # Pure reconstruction scoring (MC averaged)
    if use_multiscale:
        recon_scores, window_times = extract_window_multiscale_scores(
            model, audio, extractor, device, n_passes=n_passes,
        )
    else:
        recon_scores, window_times = extract_window_recon_scores(
            model, audio, extractor, device,
            mask_ratio=mask_ratio, n_passes=n_passes,
        )

    # Detect onset + offset
    detected_onset, detected_offset, threshold = detect_onset_offset(
        recon_scores, window_times,
        n_baseline=BASELINE_WINDOWS,
        sigma=THRESHOLD_SIGMA,
        use_ema=True,
        consecutive=CONSECUTIVE_TRIGGER,
    )

    # Presence detection: any onset detected at all (regardless of timing accuracy)
    presence_detected = detected_onset is not None

    if detected_onset is not None:
        timing_error = abs(detected_onset - true_onset)
        detected = timing_error < DETECTION_TOLERANCE
    else:
        timing_error = None
        detected = False

    voice_duration = None
    if detected_onset is not None and detected_offset is not None:
        voice_duration = round(detected_offset - detected_onset, 4)

    return {
        "file": audio_path.name,
        "true_onset": true_onset,
        "true_duration": true_duration,
        "detected_onset": detected_onset,
        "detected_offset": detected_offset,
        "voice_duration": voice_duration,
        "timing_error": timing_error,
        "detected": detected,
        "presence_detected": presence_detected,
        "method_used": "multiscale_recon" if use_multiscale else "recon",
        "snr_db": meta["snr_db"],
        "threshold": threshold,
        "n_windows": len(recon_scores),
        "score_range": [float(min(recon_scores)), float(max(recon_scores))],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluate one model on all test clips for a scenario
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    ckpt_path: Path,
    scenario: str,
    device: torch.device,
    model_size: str = "base",
    n_passes: int = RECON_N_PASSES,
    use_multiscale: bool = False,
    test_data_root: Path | None = None,
) -> dict:
    """Full evaluation of one model on all long test clips."""

    model, mask_ratio, cfg = load_model(ckpt_path, device, model_size=model_size)
    extractor = LogMelExtractor(cfg=cfg)

    print(f"    Model: {model_size}, mask_ratio={mask_ratio:.2f}, "
          f"n_passes={n_passes}, multiscale={use_multiscale}")

    # Evaluate on test clips
    _root = test_data_root if test_data_root is not None else TEST_DATA_ROOT
    test_dir = _root / scenario
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_dir}. Run generate_long_test_audio.py first."
        )

    all_results = []
    per_snr = {}

    for snr_dir in sorted(test_dir.iterdir()):
        if not snr_dir.is_dir():
            continue

        wav_files = sorted(snr_dir.glob("*.wav"))
        for wav_path in wav_files:
            json_path = wav_path.with_suffix(".json")
            if not json_path.exists():
                continue

            result = evaluate_one_clip(
                wav_path, json_path, model, extractor, device,
                mask_ratio=mask_ratio, n_passes=n_passes,
                use_multiscale=use_multiscale,
            )
            all_results.append(result)

            snr = result["snr_db"]
            if snr not in per_snr:
                per_snr[snr] = []
            per_snr[snr].append(result)

    if not all_results:
        return {"error": "No test clips found"}

    total = len(all_results)
    detected = sum(1 for r in all_results if r["detected"])
    timing_errors = [r["timing_error"] for r in all_results
                     if r["detected_onset"] is not None]
    missed = sum(1 for r in all_results if r["detected_onset"] is None)

    metrics = {
        "mask_ratio": mask_ratio,
        "model_size": model_size,
        "scenario": scenario,
        "n_passes": n_passes,
        "use_multiscale": use_multiscale,
        "total_clips": total,
        "detection_accuracy": round(detected / total * 100, 2),
        "mean_timing_error": round(float(np.mean(timing_errors)), 4) if timing_errors else None,
        "std_timing_error": round(float(np.std(timing_errors)), 4) if timing_errors else None,
        "median_timing_error": round(float(np.median(timing_errors)), 4) if timing_errors else None,
        "missed_detections": missed,
        "missed_rate": round(missed / total * 100, 2),
    }

    # Per-SNR breakdown
    per_snr_metrics = {}
    for snr in sorted(per_snr.keys()):
        results = per_snr[snr]
        n = len(results)
        det = sum(1 for r in results if r["detected"])
        errs = [r["timing_error"] for r in results if r["detected_onset"] is not None]
        durations = [r["voice_duration"] for r in results if r["voice_duration"] is not None]
        per_snr_metrics[format_snr_tag(snr)] = {
            "n_clips": n,
            "detection_rate": round(det / n * 100, 2) if n > 0 else 0,
            "n_detected": det,
            "mean_timing_error": round(float(np.mean(errs)), 4) if errs else None,
            "mean_voice_duration": round(float(np.mean(durations)), 4) if durations else None,
        }

    metrics["per_snr"] = per_snr_metrics

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global BASELINE_WINDOWS, THRESHOLD_SIGMA, CONSECUTIVE_TRIGGER

    parser = argparse.ArgumentParser(
        description="Sliding-window detection timing evaluation (pure reconstruction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to single model checkpoint")
    parser.add_argument("--sweep_dir", default=None,
                        help="Path to sweep directory (evaluates all mask ratios)")
    parser.add_argument("--scenario", default=None,
                        choices=["desert", "forest"],
                        help="Scenario (required with --checkpoint or --sweep_dir)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all sweep models for all scenarios")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir for results (avoids writing into symlinked checkpoints)")
    parser.add_argument("--model_size", default="base",
                        choices=["tiny", "small", "base", "large"],
                        help="Model variant to use")
    parser.add_argument("--n_passes", type=int, default=RECON_N_PASSES,
                        help="MC passes for reconstruction scoring")
    parser.add_argument("--multiscale", action="store_true",
                        help="Use multi-scale reconstruction scoring")
    parser.add_argument("--baseline_windows", type=int, default=BASELINE_WINDOWS,
                        help="Number of baseline windows for threshold")
    parser.add_argument("--threshold_sigma", type=float, default=THRESHOLD_SIGMA,
                        help="Threshold = mean + sigma * std")
    parser.add_argument("--consecutive", type=int, default=CONSECUTIVE_TRIGGER,
                        help="Consecutive windows above threshold to confirm detection")
    parser.add_argument("--test_data_dir", default=None,
                        help="Test data root (default: data/long_test; use data/long_test_fine for 0.5dB)")
    args = parser.parse_args()

    BASELINE_WINDOWS = args.baseline_windows
    THRESHOLD_SIGMA = args.threshold_sigma
    CONSECUTIVE_TRIGGER = args.consecutive

    device = get_device(verbose=True)
    out_dir_base = Path(args.out_dir) if args.out_dir else None
    test_data_root = Path(args.test_data_dir) if args.test_data_dir else None

    if args.checkpoint:
        if not args.scenario:
            parser.error("--scenario required with --checkpoint")
        ckpt_path = Path(args.checkpoint)
        print(f"\n  Evaluating: {ckpt_path}")
        metrics = evaluate_model(
            ckpt_path, args.scenario, device,
            model_size=args.model_size,
            n_passes=args.n_passes,
            use_multiscale=args.multiscale,
            test_data_root=test_data_root,
        )

        if out_dir_base:
            out_path = out_dir_base / "detection_results.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = ckpt_path.parent / "detection_results.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  Results saved: {out_path}")
        print(f"  Detection accuracy: {metrics.get('detection_accuracy', 'N/A')}%")
        print(f"  Mean timing error:  {metrics.get('mean_timing_error', 'N/A')}s")

    elif args.sweep_dir or args.all:
        if args.all:
            sweep_configs = []
            for scenario in ["desert", "forest"]:
                sweep_dir = _SPEC / "results" / f"sweep_{scenario}"
                if sweep_dir.exists():
                    sweep_configs.append((sweep_dir, scenario))
        else:
            if not args.scenario:
                parser.error("--scenario required with --sweep_dir")
            sweep_configs = [(Path(args.sweep_dir), args.scenario)]

        for sweep_dir, scenario in sweep_configs:
            # Determine output directory for this scenario
            if out_dir_base:
                if args.all:
                    scenario_out_dir = out_dir_base / f"eval_{scenario}"
                else:
                    scenario_out_dir = out_dir_base
                scenario_out_dir.mkdir(parents=True, exist_ok=True)
            else:
                scenario_out_dir = None

            print(f"\n{'='*60}")
            print(f"  Evaluating sweep: {scenario} (pure reconstruction)")
            print(f"  Sweep dir: {sweep_dir}")
            if scenario_out_dir:
                print(f"  Output dir: {scenario_out_dir}")
            print(f"{'='*60}")

            all_metrics = []

            mr_dirs = sorted(sweep_dir.glob("mr_*"))
            for mr_dir in mr_dirs:
                model_path = mr_dir / "model.pth"
                if not model_path.exists():
                    continue

                mr_str = mr_dir.name.replace("mr_", "")
                print(f"\n  [{mr_str}] Evaluating...")

                try:
                    metrics = evaluate_model(
                        model_path, scenario, device,
                        model_size=args.model_size,
                        n_passes=args.n_passes,
                        use_multiscale=args.multiscale,
                        test_data_root=test_data_root,
                    )
                    all_metrics.append(metrics)

                    # Write results to out_dir (or alongside checkpoint)
                    if scenario_out_dir:
                        mr_out = scenario_out_dir / mr_dir.name
                        mr_out.mkdir(parents=True, exist_ok=True)
                        out_path = mr_out / "detection_results.json"
                    else:
                        out_path = mr_dir / "detection_results.json"
                    with open(out_path, "w") as f:
                        json.dump(metrics, f, indent=2)

                    print(f"    Accuracy: {metrics.get('detection_accuracy', 'N/A')}%  "
                          f"Timing: {metrics.get('mean_timing_error', 'N/A')}s")
                except Exception as e:
                    print(f"    ERROR: {e}")

            if all_metrics:
                if scenario_out_dir:
                    summary_path = scenario_out_dir / "detection_sweep_summary.json"
                else:
                    summary_path = sweep_dir / "detection_sweep_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(all_metrics, f, indent=2)
                print(f"\n  Sweep summary saved: {summary_path}")

                print(f"\n  {'mask_ratio':>10s}  {'accuracy':>10s}  {'timing_err':>10s}")
                print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
                for m in all_metrics:
                    mr = m.get("mask_ratio", "?")
                    acc = m.get("detection_accuracy", "N/A")
                    te = m.get("mean_timing_error", "N/A")
                    if te is None:
                        print(f"  {mr:>10.2f}  {acc:>9.1f}%  {'N/A':>10s}")
                    else:
                        print(f"  {mr:>10.2f}  {acc:>9.1f}%  {te:>9.3f}s")

                best = max(all_metrics, key=lambda m: m.get("detection_accuracy", 0))
                print(f"\n  Best: mask_ratio={best['mask_ratio']:.2f} "
                      f"accuracy={best['detection_accuracy']}%")
    else:
        parser.error("Provide --checkpoint, --sweep_dir, or --all")


if __name__ == "__main__":
    main()
