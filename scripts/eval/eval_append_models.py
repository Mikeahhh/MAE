"""
Evaluate specific mask_ratio models and APPEND results to existing height sweep JSON.

Unlike eval_height_sweep.py which overwrites the JSON, this script preserves
existing results and only adds new mask_ratio entries.

Usage:
    python -m SpecMae.scripts.eval.eval_append_models \
        --scenario desert --mask_ratios 0.00 0.05 0.95
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import gc

_SPEC = Path(__file__).resolve().parents[2]

from SpecMae.scripts.eval.eval_height_sweep import (
    HEIGHT_CONFIGS, DATA_ROOT, RESULTS_ROOT, SWEEP_ROOT,
    compute_all_peak_snrs, SOURCE_SPL, NOISE_SPL,
)
from SpecMae.scripts.eval.eval_detection_timing import (
    load_model, evaluate_one_clip,
)
from SpecMae.scripts.utils.feature_extraction import LogMelExtractor
from SpecMae.scripts.utils.snr_format import format_snr_tag
from SpecMae.scripts.utils.device import get_device, empty_device_cache


def evaluate_single_model(
    scenario: str,
    mask_ratio: float,
    heights: list[int],
    height_snrs: dict[int, float],
    device: torch.device,
    n_passes: int = 100,
) -> dict:
    """Evaluate one mask_ratio model across all heights."""
    mr_str = f"mr_{mask_ratio:.2f}"
    ckpt_path = SWEEP_ROOT / f"sweep_{scenario}" / mr_str / "model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model not found: {ckpt_path}")

    model, mr_loaded, cfg = load_model(ckpt_path, device)
    extractor = LogMelExtractor(cfg=cfg)

    mr_result = {"mask_ratio": mask_ratio, "per_height": {}}

    for height in heights:
        snr_db = height_snrs[height]
        snr_tag = format_snr_tag(snr_db)
        test_dir = DATA_ROOT / f"h_{height:02d}m" / scenario / snr_tag

        if not test_dir.exists():
            print(f"  WARNING: {test_dir} not found, skipping")
            continue

        wav_files = sorted(test_dir.glob("*.wav"))
        if not wav_files:
            print(f"  WARNING: No WAV files in {test_dir}")
            continue

        clip_results = []
        t0 = time.time()
        for i, wav_path in enumerate(wav_files):
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

            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(wav_files) - i - 1)
            print(
                f"\r  mr={mask_ratio:.2f} h={height}m "
                f"clip {i+1}/{len(wav_files)} ETA:{eta:.0f}s",
                end="", flush=True,
            )

        print()

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

    del model, extractor
    empty_device_cache(device)
    gc.collect()

    return mr_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate specific models and append to height sweep JSON",
    )
    parser.add_argument("--scenario", required=True, choices=["desert", "forest"])
    parser.add_argument("--mask_ratios", nargs="+", type=float, required=True)
    parser.add_argument("--n_passes", type=int, default=100)
    parser.add_argument("--n_passes_mr0", type=int, default=1,
                        help="n_passes for mr=0.00 (deterministic, default 1)")

    args = parser.parse_args()

    heights = HEIGHT_CONFIGS[args.scenario]
    height_snrs = compute_all_peak_snrs(args.scenario, heights)
    device = get_device(verbose=True)

    # Load existing JSON
    json_path = (RESULTS_ROOT / f"height_sweep_{args.scenario}"
                 / f"height_sweep_{args.scenario}.json")
    with open(json_path) as f:
        data = json.load(f)

    existing_mrs = {r["mask_ratio"] for r in data["results"]}
    print(f"Existing mask_ratios in JSON: {sorted(existing_mrs)}")

    for mr in args.mask_ratios:
        n_passes = args.n_passes_mr0 if mr == 0.0 else args.n_passes
        print(f"\n{'='*50}")
        print(f"  Evaluating mr={mr:.2f}, n_passes={n_passes}")
        print(f"{'='*50}")

        result = evaluate_single_model(
            args.scenario, mr, heights, height_snrs, device, n_passes,
        )

        # Remove existing entry for this mr if present
        data["results"] = [r for r in data["results"] if r["mask_ratio"] != mr]
        data["results"].append(result)

        print(f"  mr={mr:.2f} done. Heights: {list(result['per_height'].keys())}")

    # Sort results by mask_ratio
    data["results"].sort(key=lambda r: r["mask_ratio"])

    # Save
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {json_path}")
    print(f"Total mask_ratios: {len(data['results'])}")


if __name__ == "__main__":
    main()
