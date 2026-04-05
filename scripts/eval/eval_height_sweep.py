"""
Height-based physical SNR detection evaluation.

Instead of fixed SNR values, compute SNR from the physical propagation model
at different UAV flight heights, then evaluate detection across all mask ratios.

Usage:
    python -m SpecMae.scripts.eval.eval_height_sweep --scenario desert
    python -m SpecMae.scripts.eval.eval_height_sweep --all --n_clips 20 --n_passes 100
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Project root ──────────────────────────────────────────────────────────
_SPEC = Path(__file__).resolve().parents[2]  # SpecMae/

# ── Propagation model ────────────────────────────────────────────────────
from SpecMae.simulation.core.propagation_model import (
    PropagationModel,
    DEFAULT_SOURCE_SPL,
    DEFAULT_DRONE_NOISE_SPL,
    DEFAULT_HORIZONTAL_OFFSET,
)

# ── Audio generation ─────────────────────────────────────────────────────
from SpecMae.scripts.utils.generate_long_test_audio import generate_scenario

# ── Model loading & evaluation ───────────────────────────────────────────
from SpecMae.scripts.eval.eval_detection_timing import (
    load_model,
    evaluate_one_clip,
)
from SpecMae.scripts.utils.feature_extraction import LogMelExtractor
from SpecMae.scripts.utils.snr_format import format_snr_tag
from SpecMae.scripts.utils.device import get_device, empty_device_cache


# ═════════════════════════════════════════════════════════════════════════
#  Configuration
# ═════════════════════════════════════════════════════════════════════════

HEIGHT_CONFIGS: dict[str, list[int]] = {
    "desert": [5, 10, 15, 20],
    "forest": [15, 20, 35, 50],
}

# Physical parameters — whiteboard (SPAWC paper)
SOURCE_SPL = DEFAULT_SOURCE_SPL        # 120 dB SPL @ 1m (whiteboard)
NOISE_SPL = DEFAULT_DRONE_NOISE_SPL    # 75 dB drone noise
PEAK_DISTANCE = DEFAULT_HORIZONTAL_OFFSET  # 5 m horizontal offset

DATA_ROOT = _SPEC / "data" / "test_height"
RESULTS_ROOT = _SPEC / "results"
SWEEP_ROOT = _SPEC / "results"


# ═════════════════════════════════════════════════════════════════════════
#  Step 1: Compute peak SNR from physical model
# ═════════════════════════════════════════════════════════════════════════

def compute_peak_snr(
    scenario: str,
    height: int,
    source_spl: float = SOURCE_SPL,
    noise_spl: float = NOISE_SPL,
) -> float:
    """Compute peak SNR at the mic for a given flight height.

    Whiteboard: mic = UAV height, horizontal offset = 5m.
    """
    model = PropagationModel(terrain=scenario)
    snr = model.snr_at_distance(
        distance=PEAK_DISTANCE,
        source_spl=source_spl,
        noise_spl=noise_spl,
        flight_height=float(height),
    )
    return round(snr, 1)


def compute_all_peak_snrs(
    scenario: str,
    heights: list[int],
) -> dict[int, float]:
    """Compute peak SNR for each height. Returns {height: snr_db}."""
    return {h: compute_peak_snr(scenario, h) for h in heights}


# ═════════════════════════════════════════════════════════════════════════
#  Step 2: Generate test audio for each height
# ═════════════════════════════════════════════════════════════════════════

def generate_height_data(
    scenario: str,
    heights: list[int],
    height_snrs: dict[int, float],
    n_clips: int = 20,
    seed: int = 42,
) -> None:
    """Generate test audio clips for each height/SNR combination."""
    print(f"\n{'='*60}")
    print(f"  Generating test data for {scenario}")
    print(f"  Heights: {heights}")
    print(f"  SNRs: {[height_snrs[h] for h in heights]}")
    print(f"  Clips per height: {n_clips}")
    print(f"{'='*60}")

    for height in heights:
        snr_db = height_snrs[height]
        out_root = DATA_ROOT / f"h_{height:02d}m"

        # Check if data already exists
        snr_tag = format_snr_tag(snr_db)
        target_dir = out_root / scenario / snr_tag
        if target_dir.exists():
            n_existing = len(list(target_dir.glob("*.wav")))
            if n_existing >= n_clips:
                print(f"  h={height}m ({snr_tag}): {n_existing} clips exist, skipping")
                continue

        generate_scenario(
            scenario=scenario,
            n_clips=n_clips,
            seed=seed + height,  # Different seed per height for variety
            snr_values=[snr_db],
            output_root=out_root,
        )


# ═════════════════════════════════════════════════════════════════════════
#  Step 3: Evaluate all models across heights
# ═════════════════════════════════════════════════════════════════════════

def find_sweep_models(scenario: str) -> list[Path]:
    """Find all mr_* model directories for a scenario, sorted by mask_ratio."""
    sweep_dir = SWEEP_ROOT / f"sweep_{scenario}"
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    dirs = sorted(sweep_dir.glob("mr_*"))
    if not dirs:
        raise FileNotFoundError(f"No mr_* dirs found in {sweep_dir}")
    return dirs


def evaluate_scenario(
    scenario: str,
    heights: list[int],
    height_snrs: dict[int, float],
    device: torch.device,
    n_passes: int = 100,
) -> list[dict]:
    """Evaluate all mask_ratio models across all heights for one scenario.

    Outer loop: models (load once per mask_ratio)
    Inner loop: heights × clips

    Returns list of per-model result dicts.
    """
    mr_dirs = find_sweep_models(scenario)
    n_models = len(mr_dirs)
    results = []

    total_start = time.time()

    for model_idx, mr_dir in enumerate(mr_dirs):
        ckpt_path = mr_dir / "model.pth"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping")
            continue

        # Load model once
        model, mask_ratio, cfg = load_model(ckpt_path, device)
        extractor = LogMelExtractor(cfg=cfg)

        mr_label = f"mr={mask_ratio:.2f}"
        mr_result: dict = {
            "mask_ratio": mask_ratio,
            "per_height": {},
        }

        for h_idx, height in enumerate(heights):
            snr_db = height_snrs[height]
            snr_tag = format_snr_tag(snr_db)
            test_dir = DATA_ROOT / f"h_{height:02d}m" / scenario / snr_tag

            if not test_dir.exists():
                print(f"  WARNING: Test dir not found: {test_dir}")
                continue

            wav_files = sorted(test_dir.glob("*.wav"))
            if not wav_files:
                print(f"  WARNING: No WAV files in {test_dir}")
                continue

            clip_results = []
            for clip_idx, wav_path in enumerate(wav_files):
                json_path = wav_path.with_suffix(".json")
                if not json_path.exists():
                    continue

                result = evaluate_one_clip(
                    audio_path=wav_path,
                    meta_path=json_path,
                    model=model,
                    extractor=extractor,
                    device=device,
                    mask_ratio=mask_ratio,
                    n_passes=n_passes,
                )
                clip_results.append(result)

                # Progress
                elapsed = time.time() - total_start
                clips_done = (
                    model_idx * len(heights) * len(wav_files)
                    + h_idx * len(wav_files)
                    + clip_idx + 1
                )
                total_clips = n_models * len(heights) * len(wav_files)
                if total_clips > 0 and clips_done > 0:
                    rate = elapsed / clips_done
                    eta = rate * (total_clips - clips_done)
                    eta_str = _format_time(eta)
                else:
                    eta_str = "?"

                print(
                    f"\r  [{scenario}] Model {model_idx+1}/{n_models} ({mr_label}), "
                    f"Height {h_idx+1}/{len(heights)} (h={height}m, SNR={snr_db:+.1f}dB), "
                    f"Clip {clip_idx+1}/{len(wav_files)}  "
                    f"ETA: {eta_str}",
                    end="", flush=True,
                )

            # Aggregate per-height results
            if clip_results:
                detected = sum(1 for r in clip_results if r["detected"])
                presence_detected = sum(
                    1 for r in clip_results if r.get("presence_detected", False)
                )
                timing_errors = [
                    r["timing_error"] for r in clip_results
                    if r["timing_error"] is not None
                ]
                n_total = len(clip_results)

                # MC binomial std for error bars
                p_det = detected / n_total
                detection_std = round(
                    (p_det * (1 - p_det) / n_total) ** 0.5 * 100, 2
                )
                p_pres = presence_detected / n_total
                presence_std = round(
                    (p_pres * (1 - p_pres) / n_total) ** 0.5 * 100, 2
                )

                mr_result["per_height"][height] = {
                    "snr_db": snr_db,
                    "n_clips": n_total,
                    "n_detected": detected,
                    "detection_accuracy": round(detected / n_total * 100, 2),
                    "detection_std": detection_std,
                    "n_presence_detected": presence_detected,
                    "presence_accuracy": round(presence_detected / n_total * 100, 2),
                    "presence_std": presence_std,
                    "mean_timing_error": (
                        round(float(np.mean(timing_errors)), 4)
                        if timing_errors else None
                    ),
                    "median_timing_error": (
                        round(float(np.median(timing_errors)), 4)
                        if timing_errors else None
                    ),
                }

            print()  # newline after height

        # Free model memory
        del model, extractor
        empty_device_cache(device)
        gc.collect()

        results.append(mr_result)

    total_elapsed = time.time() - total_start
    print(f"\n  [{scenario}] Evaluation complete in {_format_time(total_elapsed)}")

    return results


def _format_time(seconds: float) -> str:
    """Format seconds as Xh Ym or Ym Zs."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


# ═════════════════════════════════════════════════════════════════════════
#  Step 4: Save results
# ═════════════════════════════════════════════════════════════════════════

def save_results(
    results: list[dict],
    scenario: str,
    heights: list[int],
    height_snrs: dict[int, float],
    n_passes: int,
    n_clips: int,
    out_dir: Path | None = None,
) -> Path:
    """Save evaluation results as JSON."""
    if out_dir is None:
        out_dir = RESULTS_ROOT / f"height_sweep_{scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "scenario": scenario,
        "n_passes": n_passes,
        "n_clips_per_height": n_clips,
        "source_spl": SOURCE_SPL,
        "noise_spl": NOISE_SPL,
        "peak_distance_m": PEAK_DISTANCE,
        "heights": {
            h: {
                "flight_height_m": h,
                "mic_height_m": float(h),  # mic = UAV height (no cable)
                "peak_snr_db": height_snrs[h],
            }
            for h in heights
        },
        "results": results,
    }

    out_path = out_dir / f"height_sweep_{scenario}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {out_path}")

    return out_path


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def run_scenario(
    scenario: str,
    heights: list[int] | None = None,
    n_clips: int = 20,
    n_passes: int = 100,
    seed: int = 42,
    skip_datagen: bool = False,
    out_dir: Path | None = None,
) -> Path:
    """Full pipeline for one scenario: compute SNR → generate data → evaluate → save."""
    if heights is None:
        heights = HEIGHT_CONFIGS[scenario]

    print(f"\n{'#'*60}")
    print(f"  Height-Based Detection Sweep: {scenario.upper()}")
    print(f"  Heights: {heights}")
    print(f"  n_clips={n_clips}, n_passes={n_passes}, seed={seed}")
    print(f"{'#'*60}")

    # 1. Compute physical SNRs (whiteboard formula)
    height_snrs = compute_all_peak_snrs(scenario, heights)
    print(f"\n  Peak SNR values (mic = UAV height, offset={PEAK_DISTANCE}m):")
    for h in heights:
        print(f"    h={h}m  →  SNR={height_snrs[h]:+.1f} dB")

    # 2. Generate test data
    if not skip_datagen:
        generate_height_data(scenario, heights, height_snrs, n_clips, seed)
    else:
        print("\n  Skipping data generation (--skip_datagen)")

    # 3. Evaluate
    device = get_device(verbose=True)
    results = evaluate_scenario(
        scenario=scenario,
        heights=heights,
        height_snrs=height_snrs,
        device=device,
        n_passes=n_passes,
    )

    # 4. Save
    return save_results(results, scenario, heights, height_snrs, n_passes, n_clips, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Height-based physical SNR detection evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario", default=None, choices=["desert", "forest"],
        help="Scenario to evaluate (default: use --all for both)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both desert and forest scenarios",
    )
    parser.add_argument(
        "--heights", type=int, nargs="+", default=None,
        help="Override default heights (e.g., --heights 5 10 15 20)",
    )
    parser.add_argument(
        "--n_clips", type=int, default=20,
        help="Number of test clips per height",
    )
    parser.add_argument(
        "--n_passes", type=int, default=100,
        help="MC passes per evaluation (advisor: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data generation",
    )
    parser.add_argument(
        "--skip_datagen", action="store_true",
        help="Skip audio generation (use existing data)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Override output directory for results JSON",
    )

    args = parser.parse_args()

    # Determine scenarios to run
    if args.all:
        scenarios = ["desert", "forest"]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.error("Specify --scenario or --all")

    out_dir = Path(args.out_dir) if args.out_dir else None

    for scenario in scenarios:
        run_scenario(
            scenario=scenario,
            heights=args.heights,
            n_clips=args.n_clips,
            n_passes=args.n_passes,
            seed=args.seed,
            skip_datagen=args.skip_datagen,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
