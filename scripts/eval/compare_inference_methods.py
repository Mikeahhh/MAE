"""
Compare inference methods: Paper (no-mask) vs Code (MC-100) vs Single-mask.

Tests three inference strategies on the same test clips:
  A) Paper method:  mask_ratio=0.0,     n_passes=1   (fastest)
  B) Single mask:   mask_ratio=trained,  n_passes=1   (cheapest with mask)
  C) MC-100:        mask_ratio=trained,  n_passes=100 (current code)

Usage:
    # Fast comparison (methods A + B only, ~20 min)
    python -m SpecMae.scripts.eval.compare_inference_methods

    # Full comparison including MC-100 (~hours)
    python -m SpecMae.scripts.eval.compare_inference_methods --include_mc100

    # Subsample for quick test
    python -m SpecMae.scripts.eval.compare_inference_methods --max_clips 10

    # Single scenario
    python -m SpecMae.scripts.eval.compare_inference_methods --scenario desert
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

# ── Detection parameters (identical across all methods) ──────────────
SR = 48_000
WINDOW_SEC = 1.0
HOP_SEC    = 0.5
WINDOW_SAMPLES = int(SR * WINDOW_SEC)
HOP_SAMPLES    = int(SR * HOP_SEC)

BASELINE_WINDOWS    = 5
THRESHOLD_SIGMA     = 1.0
DETECTION_TOLERANCE = 2.0
EMA_ALPHA           = 0.3
CONSECUTIVE_TRIGGER = 1
SCORE_MODE          = "top_k"
TOP_K_RATIO         = 0.30

TEST_DATA_ROOT = _SPEC / "data" / "test_height"

# ── Scenario configs ─────────────────────────────────────────────────
SCENARIOS = {
    "desert": {
        "ckpt": _SPEC / "results" / "sweep_desert" / "mr_0.10" / "model.pth",
        "heights": ["h_05m", "h_10m", "h_15m", "h_20m"],
    },
    "forest": {
        "ckpt": _SPEC / "results" / "sweep_forest" / "mr_0.10" / "model.pth",
        "heights": ["h_15m", "h_20m", "h_35m", "h_50m"],
    },
}


# ── Model loading ────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
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
    factory = get_model_factory("base")
    model = factory(mask_ratio=mask_ratio, norm_pix_loss=True, n_mels=cfg.n_mels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, mask_ratio, cfg


# ── Sliding-window scoring ───────────────────────────────────────────

@torch.no_grad()
def score_clip(model, audio, extractor, device, mask_ratio, n_passes):
    """Score all sliding windows in a 12s clip. Returns (scores, window_times)."""
    n_samples = len(audio)
    scores, times = [], []
    start = 0
    while start + WINDOW_SAMPLES <= n_samples:
        window = audio[start : start + WINDOW_SAMPLES]
        spec = extractor.extract(window).unsqueeze(0).to(device)
        s = model.compute_anomaly_score(
            spec, mask_ratio=mask_ratio, n_passes=n_passes,
            score_mode=SCORE_MODE, top_k_ratio=TOP_K_RATIO,
        )
        scores.append(s.item())
        times.append((start + WINDOW_SAMPLES / 2) / SR)
        start += HOP_SAMPLES
    return scores, times


# ── Detection logic ──────────────────────────────────────────────────

def detect_onset(scores, window_times):
    """Adaptive threshold detection with EMA smoothing."""
    if len(scores) < BASELINE_WINDOWS + 1:
        return None, 0.0

    # EMA smooth
    smoothed = []
    for i, s in enumerate(scores):
        smoothed.append(s if i == 0 else EMA_ALPHA * s + (1 - EMA_ALPHA) * smoothed[-1])

    baseline = smoothed[:BASELINE_WINDOWS]
    bl_mean = np.mean(baseline)
    bl_std = max(np.std(baseline), 1e-6)
    threshold = bl_mean + THRESHOLD_SIGMA * bl_std

    above_count = 0
    for i in range(BASELINE_WINDOWS, len(smoothed)):
        if smoothed[i] > threshold:
            above_count += 1
            if above_count >= CONSECUTIVE_TRIGGER:
                onset_idx = i - CONSECUTIVE_TRIGGER + 1
                return window_times[onset_idx], threshold
        else:
            above_count = 0
    return None, threshold


# ── Evaluate one clip with one method ────────────────────────────────

def evaluate_clip(model, audio, extractor, device, mask_ratio, n_passes, true_onset):
    scores, times = score_clip(model, audio, extractor, device, mask_ratio, n_passes)
    detected_onset, threshold = detect_onset(scores, times)

    presence = detected_onset is not None
    if detected_onset is not None:
        timing_error = abs(detected_onset - true_onset)
        detected = timing_error < DETECTION_TOLERANCE
    else:
        timing_error = None
        detected = False

    return {
        "presence": presence,
        "detected": detected,
        "timing_error": timing_error,
    }


# ── Run one scenario ────────────────────────────────────────────────

def run_scenario(scenario, device, methods, max_clips=None):
    cfg = SCENARIOS[scenario]
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario.upper()}")
    print(f"{'='*60}")

    model, trained_mr, audio_cfg = load_model(cfg["ckpt"], device)
    extractor = LogMelExtractor(cfg=audio_cfg)
    print(f"  Model loaded: mask_ratio={trained_mr:.2f}")

    results = {}  # method_name -> height -> {presence_list, detected_list, timing_errors}

    for method in methods:
        mr = method["mask_ratio"] if method["mask_ratio"] is not None else trained_mr
        n_passes = method["n_passes"]
        name = method["name"]
        results[name] = {}

        print(f"\n  Method: {name} (mr={mr}, n_passes={n_passes})")
        method_start = time.time()

        for height_tag in cfg["heights"]:
            height_dir = TEST_DATA_ROOT / height_tag / scenario
            if not height_dir.exists():
                print(f"    {height_tag}: directory not found, skipping")
                continue

            # Find SNR subdirectory
            snr_dirs = [d for d in height_dir.iterdir() if d.is_dir()]
            if not snr_dirs:
                continue
            snr_dir = snr_dirs[0]
            snr_tag = snr_dir.name

            wav_files = sorted(snr_dir.glob("*.wav"))
            if max_clips is not None:
                wav_files = wav_files[:max_clips]

            presence_list = []
            detected_list = []
            timing_errors = []

            for wav_path in wav_files:
                json_path = wav_path.with_suffix(".json")
                if not json_path.exists():
                    continue

                audio, _ = load_audio(str(wav_path), sr=SR)
                with open(json_path) as f:
                    meta = json.load(f)
                true_onset = meta["voice_onset_sec"]

                result = evaluate_clip(
                    model, audio, extractor, device, mr, n_passes, true_onset
                )
                presence_list.append(result["presence"])
                detected_list.append(result["detected"])
                if result["timing_error"] is not None:
                    timing_errors.append(result["timing_error"])

            n = len(presence_list)
            presence_acc = sum(presence_list) / n * 100 if n > 0 else 0
            detection_acc = sum(detected_list) / n * 100 if n > 0 else 0
            mean_te = np.mean(timing_errors) if timing_errors else float("nan")

            results[name][height_tag] = {
                "snr": snr_tag,
                "n_clips": n,
                "presence_acc": presence_acc,
                "detection_acc": detection_acc,
                "mean_timing_error": mean_te,
            }

            print(f"    {height_tag} ({snr_tag}): "
                  f"presence={presence_acc:.1f}% "
                  f"detection={detection_acc:.1f}% "
                  f"timing={mean_te:.3f}s "
                  f"[n={n}]")

        elapsed = time.time() - method_start
        print(f"  Method {name} done in {elapsed:.1f}s")
        results[name]["_elapsed_s"] = elapsed

    return results, trained_mr


# ── Print comparison table ───────────────────────────────────────────

def print_comparison(scenario, results, trained_mr, heights):
    method_names = [k for k in results if not k.startswith("_")]

    print(f"\n{'='*80}")
    print(f"  {scenario.upper()} (trained mr={trained_mr:.2f}) — COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    header = f"{'Height':>8} {'SNR':>10}"
    for name in method_names:
        header += f" | {name:>20}"
    print(header)
    print("-" * len(header))

    # Presence accuracy
    print(f"\n  ** Presence Accuracy (%) **")
    for h in heights:
        row = ""
        snr = ""
        for name in method_names:
            data = results[name].get(h, {})
            if not snr and "snr" in data:
                snr = data["snr"]
            val = data.get("presence_acc", float("nan"))
            row += f" | {val:>20.1f}"
        print(f"{h:>8} {snr:>10}{row}")

    # Detection accuracy
    print(f"\n  ** Detection Accuracy (%) — onset within ±2s **")
    for h in heights:
        row = ""
        snr = ""
        for name in method_names:
            data = results[name].get(h, {})
            if not snr and "snr" in data:
                snr = data["snr"]
            val = data.get("detection_acc", float("nan"))
            row += f" | {val:>20.1f}"
        print(f"{h:>8} {snr:>10}{row}")

    # Mean timing error
    print(f"\n  ** Mean Timing Error (s) **")
    for h in heights:
        row = ""
        snr = ""
        for name in method_names:
            data = results[name].get(h, {})
            if not snr and "snr" in data:
                snr = data["snr"]
            val = data.get("mean_timing_error", float("nan"))
            row += f" | {val:>20.3f}"
        print(f"{h:>8} {snr:>10}{row}")

    # Elapsed time
    print(f"\n  ** Wall-clock Time **")
    for name in method_names:
        elapsed = results[name].get("_elapsed_s", 0)
        print(f"    {name}: {elapsed:.1f}s")

    # Overall average
    print(f"\n  ** Overall Average **")
    for name in method_names:
        pres_vals = [results[name][h]["presence_acc"] for h in heights if h in results[name]]
        det_vals = [results[name][h]["detection_acc"] for h in heights if h in results[name]]
        if pres_vals:
            print(f"    {name}: presence={np.mean(pres_vals):.1f}%  detection={np.mean(det_vals):.1f}%")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare inference methods: Paper vs Single-mask vs MC-100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", choices=["desert", "forest"], default=None,
                        help="Run one scenario (default: both)")
    parser.add_argument("--include_mc100", action="store_true",
                        help="Include MC-100 method (slow, ~hours)")
    parser.add_argument("--mc_passes", type=int, default=100,
                        help="Number of MC passes for method C")
    parser.add_argument("--max_clips", type=int, default=None,
                        help="Max clips per height (for quick testing)")
    args = parser.parse_args()

    device = get_device(verbose=True)

    # Define methods
    methods = [
        {"name": "Paper(mr=0,n=1)", "mask_ratio": 0.0, "n_passes": 1},
        {"name": "Single(mr=T,n=1)", "mask_ratio": None, "n_passes": 1},
    ]
    if args.include_mc100:
        methods.append(
            {"name": f"MC-{args.mc_passes}(mr=T)", "mask_ratio": None, "n_passes": args.mc_passes}
        )

    scenarios = [args.scenario] if args.scenario else ["desert", "forest"]
    all_results = {}

    for scenario in scenarios:
        results, trained_mr = run_scenario(scenario, device, methods, args.max_clips)
        print_comparison(scenario, results, trained_mr, SCENARIOS[scenario]["heights"])
        all_results[scenario] = results

    # Save results
    out_path = _SPEC / "results" / "inference_method_comparison.json"
    serializable = {}
    for sc, res in all_results.items():
        serializable[sc] = {}
        for method, data in res.items():
            serializable[sc][method] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    serializable[sc][method][k] = {
                        kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                        for kk, vv in v.items()
                    }
                else:
                    serializable[sc][method][k] = float(v) if isinstance(v, (np.floating, float)) else v
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
