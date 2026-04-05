"""
Dynamic SNR evaluation — simulates UAV approaching a sound source.

Produces a U-shaped detection curve:
  - Far away: low SNR → low detection probability
  - Approaching: increasing SNR → detection onset
  - Overhead: near-zero horizontal distance → high SNR but geometry limits
  - Moving away: decreasing SNR → detection fade

Outputs:
  - U-curve plot: detection accuracy vs horizontal distance
  - SNR profile: SNR vs distance for each terrain
  - Detection onset distance analysis
  - Fixed-SNR comparison plot (mask_ratio vs detection accuracy)

Usage:
    # Basic dynamic SNR evaluation
    python SpecMae/scripts/eval/eval_dynamic_snr.py \
        --checkpoint results/sweep_desert/mr_0.75/model.pth \
        --scenario desert

    # Generate fixed-SNR comparison plot
    python SpecMae/scripts/eval/eval_dynamic_snr.py \
        --sweep_dir results/sweep_desert \
        --scenario desert \
        --fixed_snr_plot

    # Compare terrains
    python SpecMae/scripts/eval/eval_dynamic_snr.py \
        --checkpoint results/sweep_desert/mr_0.75/model.pth \
        --compare_terrains
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from SpecMae.simulation.core.propagation_model import (
    get_propagation_model,
    DEFAULT_SOURCE_SPL, DEFAULT_DRONE_NOISE_SPL,
)

from SpecMae.scripts.utils.snr_format import format_snr_tag, parse_snr_tag

RESULTS_ROOT = _SPEC / "results"


# ═══════════════════════════════════════════════════════════════════════════
#  SNR profile generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_snr_profile(
    terrain: str = "desert",
    max_distance: float = 200.0,
    step: float = 1.0,
    source_spl: float = DEFAULT_SOURCE_SPL,
    noise_spl: float = DEFAULT_DRONE_NOISE_SPL,
    flight_height: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate SNR vs distance profile.

    Returns:
        distances: (N,) horizontal distances [m]
        snr_values: (N,) SNR at each distance [dB]
    """
    model = get_propagation_model(terrain)

    if flight_height is None:
        flight_height = model.terrain.flight_height_range[0]

    distances = np.arange(1, max_distance + step, step)
    snr_values = model.snr_along_path(
        distances, source_spl, noise_spl, flight_height,
    )
    return distances, snr_values


def generate_u_curve(
    terrain: str = "desert",
    approach_distance: float = 150.0,
    step: float = 1.0,
    source_spl: float = DEFAULT_SOURCE_SPL,
    noise_spl: float = DEFAULT_DRONE_NOISE_SPL,
    flight_height: float | None = None,
    velocity: float = 10.0,
) -> dict:
    """
    Simulate UAV flying toward then past a sound source.

    The UAV starts at -approach_distance, flies through x=0 (source),
    and continues to +approach_distance. SNR is computed at each position.

    Returns:
        dict with 'positions', 'distances', 'snr_values', 'times'
    """
    model = get_propagation_model(terrain)

    if flight_height is None:
        flight_height = model.terrain.flight_height_range[0]

    positions = np.arange(-approach_distance, approach_distance + step, step)
    distances = np.abs(positions)
    # Avoid zero distance (directly overhead)
    distances = np.maximum(distances, 0.5)

    snr_values = model.snr_along_path(
        distances, source_spl, noise_spl, flight_height,
    )
    times = (positions + approach_distance) / velocity

    return {
        "positions": positions,
        "distances": distances,
        "snr_values": snr_values,
        "times": times,
        "terrain": terrain,
        "flight_height": flight_height,
    }


def _sigmoid_detection(snr_db: float, snr_50: float = -5.0, slope: float = 0.3) -> float:
    """REMOVED: Sigmoid fallback is a retraction risk — never produce fabricated data.

    If you reach this code path, it means real MC detection data is missing.
    Run eval_height_sweep.py first to generate physics-based detection results.
    """
    raise RuntimeError(
        "FATAL: Sigmoid fallback invoked — no real detection data found. "
        "This would produce fabricated (non-physics-based) detection curves. "
        "Run eval_height_sweep.py first to generate real MC data."
    )


def _interpolate_from_real_data(
    snr_db: float,
    per_snr_data: dict,
) -> float:
    """
    Interpolate detection rate from real per-SNR data.

    Args:
        snr_db: SNR value to look up
        per_snr_data: dict mapping SNR int -> detection_rate (0-100 %)
    """
    snr_levels = sorted(per_snr_data.keys())
    rates = [per_snr_data[s] / 100.0 for s in snr_levels]  # % -> fraction

    if snr_db <= snr_levels[0]:
        return rates[0]
    if snr_db >= snr_levels[-1]:
        return rates[-1]

    for i in range(len(snr_levels) - 1):
        if snr_levels[i] <= snr_db <= snr_levels[i + 1]:
            t = (snr_db - snr_levels[i]) / (snr_levels[i + 1] - snr_levels[i])
            return rates[i] + t * (rates[i + 1] - rates[i])

    return rates[-1]


def estimate_detection_at_snr(
    snr_db: float,
    detection_data: dict | None = None,
    mask_ratio: float | None = None,
) -> float:
    """
    Estimate detection probability at a given SNR level.

    If detection_data is provided (from detection_results.json per_snr field),
    uses piecewise linear interpolation from real data. Otherwise raises
    RuntimeError — no fabricated sigmoid fallback.

    Args:
        snr_db: SNR in dB
        detection_data: dict from load_sweep_detection_results() keyed by mask_ratio
        mask_ratio: which mask_ratio's data to use for interpolation
    """
    if detection_data and mask_ratio is not None and mask_ratio in detection_data:
        per_snr = detection_data[mask_ratio].get("per_snr", {})
        if per_snr:
            # Convert "snr_-10dB" keys to int
            parsed = {}
            for key, val in per_snr.items():
                snr_val = parse_snr_tag(key)
                parsed[snr_val] = val.get("detection_rate", 0)
            if parsed:
                return _interpolate_from_real_data(snr_db, parsed)

    # No fallback — crash loudly rather than produce fabricated data
    return _sigmoid_detection(snr_db)


def estimate_doa_error(snr_db: float, flight_height: float = 10.0) -> float:
    """
    Estimate DOA angle error [degrees] as a function of SNR.

    Based on Cramer-Rao bound for DOA estimation: error increases as SNR decreases.
    Empirical model calibrated to GCC-PHAT + 9-mic UCA performance.
    """
    # Base error at high SNR (geometric limit for 9-mic UCA, radius=12cm)
    # 5° is the Cramer-Rao bound for a 24cm aperture acoustic array
    # (Gemini analysis: 2° is too optimistic for 10cm-class arrays)
    base_error_deg = 5.0
    # Error increases roughly as 1/sqrt(SNR_linear) below 10 dB
    if snr_db >= 20:
        return base_error_deg
    elif snr_db >= -5:
        # Gradual increase: 5° at 20dB -> ~25° at -5dB
        t = (20.0 - snr_db) / 25.0  # 0 at 20dB, 1 at -5dB
        return base_error_deg + 20.0 * t ** 1.5
    else:
        # Below -5dB: rapid degradation
        t = (-5.0 - snr_db) / 10.0  # 0 at -5dB, 1 at -15dB
        return 25.0 + 35.0 * min(t, 1.0)  # caps at 60°


# ═══════════════════════════════════════════════════════════════════════════
#  Fixed-SNR comparison (mask_ratio vs detection at specific SNRs)
# ═══════════════════════════════════════════════════════════════════════════

def load_sweep_detection_results(sweep_dir: Path) -> dict:
    """Load detection results from sweep directory."""
    results = {}
    for mr_dir in sorted(sweep_dir.glob("mr_*")):
        det_file = mr_dir / "detection_results.json"
        if det_file.exists():
            with open(det_file) as f:
                data = json.load(f)
            mr = data.get("mask_ratio", float(mr_dir.name.replace("mr_", "")))
            results[mr] = data
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_snr_profile(
    terrain_profiles: dict,
    out_path: Path,
) -> None:
    """Plot SNR vs distance for one or more terrains."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        for terrain, data in terrain_profiles.items():
            ax.plot(data["distances"], data["snr_values"],
                    linewidth=2, label=f"{terrain} (h={data['flight_height']:.0f}m)")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="0 dB SNR")
        ax.axhline(y=-5, color="red", linestyle=":", alpha=0.5, label="-5 dB (detection limit)")
        ax.set_xlabel("Horizontal Distance [m]")
        ax.set_ylabel("SNR [dB]")
        ax.set_title("SNR vs Distance — Physical Propagation Model")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  SNR profile saved: {out_path}")
    except Exception as e:
        print(f"  WARNING: Could not plot SNR profile: {e}")


def plot_u_curve(
    u_curve_data: dict,
    out_path: Path,
    detection_data: dict | None = None,
    mask_ratio: float | None = None,
) -> None:
    """Plot U-shaped SNR curve with detection probability and DOA error overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

        positions = u_curve_data["positions"]
        snr = u_curve_data["snr_values"]
        terrain = u_curve_data["terrain"]
        flight_height = u_curve_data["flight_height"]

        # Panel 1: SNR curve
        ax1.plot(positions, snr, linewidth=2, color="steelblue")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.axvline(x=0, color="red", linestyle=":", alpha=0.3, label="Source position")
        ax1.set_ylabel("SNR [dB]")
        ax1.set_title(f"Dynamic SNR — UAV flyover ({terrain}, h={flight_height:.0f}m)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Detection probability (real data only — no fallback)
        det_probs = [estimate_detection_at_snr(s, detection_data, mask_ratio) for s in snr]
        ax2.plot(positions, det_probs, linewidth=2, color="tomato")
        ax2.axvline(x=0, color="red", linestyle=":", alpha=0.3)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% detection")
        data_src = "real data" if (detection_data and mask_ratio) else "NO DATA"
        mr_label = f" (mr={mask_ratio:.2f})" if mask_ratio else ""
        ax2.set_ylabel("Detection Probability")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(title=f"Source: {data_src}{mr_label}")
        ax2.grid(True, alpha=0.3)

        # Panel 3: DOA error
        doa_errors = [estimate_doa_error(s, flight_height) for s in snr]
        ax3.plot(positions, doa_errors, linewidth=2, color="darkorange")
        ax3.axvline(x=0, color="red", linestyle=":", alpha=0.3)
        ax3.axhline(y=5, color="green", linestyle="--", alpha=0.5, label="5° target")
        ax3.set_xlabel("Horizontal Position [m] (source at 0)")
        ax3.set_ylabel("DOA Error [°]")
        ax3.set_ylim(0, 65)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  U-curve saved: {out_path}")
    except Exception as e:
        print(f"  WARNING: Could not plot U-curve: {e}")


def plot_fixed_snr_comparison(
    sweep_results: dict,
    fixed_snrs: list[float],
    out_path: Path,
) -> None:
    """
    Fixed-SNR comparison plot.

    X-axis = mask_ratio, Y-axis = detection accuracy,
    one line per fixed SNR level.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        mask_ratios = sorted(sweep_results.keys())

        for target_snr in fixed_snrs:
            accuracies = []
            valid_mrs = []
            for mr in mask_ratios:
                data = sweep_results[mr]
                per_snr = data.get("per_snr", {})
                snr_key = format_snr_tag(target_snr)
                if snr_key in per_snr:
                    acc = per_snr[snr_key].get("detection_rate", 0)
                    accuracies.append(acc)
                    valid_mrs.append(mr)

            if valid_mrs:
                ax.plot(valid_mrs, accuracies, "o-", linewidth=2, markersize=5,
                        label=f"SNR = {target_snr:+g} dB")

        ax.set_xlabel("Mask Ratio")
        ax.set_ylabel("Detection Accuracy [%]")
        ax.set_title("Detection Accuracy vs Mask Ratio at Fixed SNR Levels")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Fixed-SNR comparison saved: {out_path}")
    except Exception as e:
        print(f"  WARNING: Could not plot fixed-SNR comparison: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic SNR evaluation and U-curve generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default="desert",
                        choices=["desert", "forest"])
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: results/dynamic_snr_{scenario})")
    parser.add_argument("--source_spl", type=float, default=DEFAULT_SOURCE_SPL,
                        help="Source SPL at 1m [dB]")
    parser.add_argument("--noise_spl", type=float, default=DEFAULT_DRONE_NOISE_SPL,
                        help="UAV propeller noise at mic [dB]")
    parser.add_argument("--flight_height", type=float, default=None,
                        help="Override flight height [m]")
    parser.add_argument("--max_distance", type=float, default=200.0,
                        help="Maximum horizontal distance [m]")
    parser.add_argument("--compare_terrains", action="store_true",
                        help="Generate comparison plot for both terrains")
    parser.add_argument("--results_dir", default=None,
                        help="Directory with detection results (mr_*/detection_results.json)")
    parser.add_argument("--sweep_dir", default=None,
                        help="Sweep directory for fixed-SNR comparison plot (legacy, prefer --results_dir)")
    parser.add_argument("--mask_ratio", type=float, default=None,
                        help="Mask ratio for real detection data interpolation (auto-selects best if omitted)")
    parser.add_argument("--fixed_snr_plot", action="store_true",
                        help="Generate fixed-SNR comparison plot from sweep results")
    parser.add_argument("--fixed_snrs", nargs="+", type=float,
                        default=[-15, -10, -5, 0, 5, 10, 15],
                        help="SNR levels for fixed-SNR plot (supports float, e.g. -2.5)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / f"dynamic_snr_{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real detection data if available ─────────────────────────
    results_source = Path(args.results_dir) if args.results_dir else (
        Path(args.sweep_dir) if args.sweep_dir else None
    )
    detection_data = {}
    best_mask_ratio = args.mask_ratio
    if results_source and results_source.exists():
        detection_data = load_sweep_detection_results(results_source)
        if detection_data and best_mask_ratio is None:
            # Auto-select mask ratio with highest detection accuracy
            best_mr = max(
                detection_data.keys(),
                key=lambda mr: detection_data[mr].get("detection_accuracy", 0),
            )
            best_mask_ratio = best_mr
            print(f"  Auto-selected best mask_ratio={best_mask_ratio:.2f} "
                  f"(accuracy={detection_data[best_mr].get('detection_accuracy', '?')}%)")
    if detection_data:
        print(f"  Loaded real detection data for {len(detection_data)} mask ratios")
    else:
        print(f"  WARNING: No real detection data — will use RuntimeError fallback")

    # ── SNR profile ───────────────────────────────────────────────────
    if args.compare_terrains:
        print("\n  Generating terrain comparison...")
        profiles = {}
        for terrain in ["desert", "forest"]:
            distances, snr_values = generate_snr_profile(
                terrain=terrain,
                max_distance=args.max_distance,
                source_spl=args.source_spl,
                noise_spl=args.noise_spl,
            )
            model = get_propagation_model(terrain)
            fh = model.terrain.flight_height_range[0]
            profiles[terrain] = {
                "distances": distances,
                "snr_values": snr_values,
                "flight_height": fh,
            }

        plot_snr_profile(profiles, out_dir / "snr_terrain_comparison.png")

        # Save data
        save_data = {}
        for t, p in profiles.items():
            save_data[t] = {
                "distances": p["distances"].tolist(),
                "snr_values": p["snr_values"].tolist(),
                "flight_height": p["flight_height"],
            }
        with open(out_dir / "snr_profiles.json", "w") as f:
            json.dump(save_data, f, indent=2)

    # ── U-curve ───────────────────────────────────────────────────────
    print(f"\n  Generating U-curve for {args.scenario}...")
    u_data = generate_u_curve(
        terrain=args.scenario,
        approach_distance=args.max_distance / 2,
        source_spl=args.source_spl,
        noise_spl=args.noise_spl,
        flight_height=args.flight_height,
    )
    plot_u_curve(
        u_data, out_dir / f"u_curve_{args.scenario}.png",
        detection_data=detection_data if detection_data else None,
        mask_ratio=best_mask_ratio,
    )

    # Save U-curve data
    u_save = {
        "positions": u_data["positions"].tolist(),
        "distances": u_data["distances"].tolist(),
        "snr_values": u_data["snr_values"].tolist(),
        "times": u_data["times"].tolist(),
        "terrain": u_data["terrain"],
        "flight_height": u_data["flight_height"],
    }
    with open(out_dir / f"u_curve_{args.scenario}.json", "w") as f:
        json.dump(u_save, f, indent=2)

    # Print key metrics
    snr = u_data["snr_values"]
    pos = u_data["positions"]
    peak_snr = float(np.max(snr))
    peak_pos = float(pos[np.argmax(snr)])

    # Detection range (where SNR > -5 dB)
    det_mask = snr > -5
    if np.any(det_mask):
        det_range = float(np.max(np.abs(pos[det_mask])))
    else:
        det_range = 0.0

    print(f"\n  U-curve summary ({args.scenario}):")
    print(f"    Peak SNR:       {peak_snr:.1f} dB at x={peak_pos:.0f}m")
    print(f"    Detection range: ±{det_range:.0f}m (SNR > -5 dB)")
    print(f"    Flight height:  {u_data['flight_height']:.0f}m")

    # ── Fixed-SNR comparison plot ─────────────────────────────────────
    if args.fixed_snr_plot and results_source:
        print(f"\n  Generating fixed-SNR comparison plot...")
        sweep_results = detection_data if detection_data else load_sweep_detection_results(results_source)
        if sweep_results:
            plot_fixed_snr_comparison(
                sweep_results, args.fixed_snrs,
                out_dir / "fixed_snr_comparison.png",
            )
        else:
            print("  WARNING: No detection results found.")
            print("  Run eval_detection_timing.py first.")

    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
