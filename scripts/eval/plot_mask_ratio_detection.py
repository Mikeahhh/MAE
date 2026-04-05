"""
Generate publication-quality mask-ratio selection figure from V4 height-sweep data.

Figures produced:
  1. fig_mask_ratio_selection.png/pdf
     Left:  mask_ratio vs avg presence accuracy (both scenarios)
     Right: mask_ratio vs avg detection accuracy (both scenarios)

Data sources:
  - height_sweep_{scenario}.json from eval_height_sweep.py (V4 physics-based)

Usage:
    python -m SpecMae.scripts.eval.plot_mask_ratio_detection
    python -m SpecMae.scripts.eval.plot_mask_ratio_detection --out_dir results/figures
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

from SpecMae.simulation.visualization.scene_3d import set_publication_style

RESULTS_ROOT = _SPEC / "results"
FIGURES_DIR  = RESULTS_ROOT / "figures"


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading — reads V4 height_sweep JSON
# ═══════════════════════════════════════════════════════════════════════════

def load_height_sweep(scenario: str) -> dict:
    """Load height_sweep_{scenario}.json produced by eval_height_sweep.py."""
    path = RESULTS_ROOT / f"height_sweep_{scenario}" / f"height_sweep_{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Height sweep results not found: {path}")
    with open(path) as f:
        return json.load(f)


def extract_mask_ratio_stats(
    data: dict,
    metric: str = "presence_accuracy",
) -> list[dict]:
    """Extract per-mask-ratio average metric across all heights.

    Returns list of {mask_ratio, avg_metric, std_metric, per_height: {h: val}}.
    """
    results = data["results"]
    heights = sorted(int(h) for h in data["heights"].keys())
    std_key = metric.replace("accuracy", "std")

    stats = []
    for r in results:
        mr = r["mask_ratio"]
        ph = r.get("per_height", {})

        values = []
        per_height = {}
        for h in heights:
            h_data = ph.get(str(h)) or ph.get(h)
            if h_data and metric in h_data:
                val = h_data[metric]
                values.append(val)
                per_height[h] = val

        if values:
            stats.append({
                "mask_ratio": mr,
                "avg": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_height": per_height,
            })

    stats.sort(key=lambda x: x["mask_ratio"])
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  Figure: Mask ratio vs detection performance (whiteboard requirement)
# ═══════════════════════════════════════════════════════════════════════════

def plot_mask_ratio_figure(out_dir: Path):
    """
    Single-panel figure: mask_ratio vs anomaly detection accuracy (%).
    Uses presence_accuracy — the SAR-relevant metric
    (did the system detect the voice signal?).
    Both scenarios overlaid. Error bars = std across heights.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    colors = {"desert": "#c0392b", "forest": "#27ae60"}
    markers = {"desert": "s", "forest": "^"}
    annot_offsets = {"desert": (12, 8), "forest": (12, -22)}

    source_spl = "?"
    n_clips = "?"
    n_passes = "?"

    for scenario in ["desert", "forest"]:
        try:
            data = load_height_sweep(scenario)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        source_spl = data.get("source_spl", "?")
        n_clips = data.get("n_clips_per_height", "?")
        n_passes = data.get("n_passes", "?")

        color = colors[scenario]
        marker = markers[scenario]
        label = scenario.capitalize()
        offset = annot_offsets[scenario]

        pres_stats = extract_mask_ratio_stats(data, "presence_accuracy")
        if pres_stats:
            mrs = [s["mask_ratio"] for s in pres_stats]
            avgs = [s["avg"] for s in pres_stats]
            stds = [s["std"] for s in pres_stats]

            ax.errorbar(mrs, avgs, yerr=stds, fmt=f"-{marker}",
                        color=color, label=label, linewidth=2,
                        markersize=7, capsize=3, capthick=1.5)

            # Mark best
            best_idx = int(np.argmax(avgs))
            ax.plot(mrs[best_idx], avgs[best_idx],
                    "*", color=color, markersize=16, zorder=5)
            ax.annotate(
                f"{avgs[best_idx]:.1f}% (mr={mrs[best_idx]:.2f})",
                xy=(mrs[best_idx], avgs[best_idx]),
                xytext=offset, textcoords="offset points",
                fontsize=9, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.5),
            )

    ax.set_xlabel("Mask Ratio", fontsize=12)
    ax.set_ylabel("Anomaly Detection Accuracy (%)", fontsize=12)
    ax.set_title(
        f"SpecMAE Mask Ratio Selection\n"
        f"(SOURCE={source_spl} dB, {n_clips}-MC, {n_passes} passes)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 105)

    fig.tight_layout()

    for ext in ("png",):
        fig_path = out_dir / f"fig_mask_ratio_selection.{ext}"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {fig_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate mask-ratio selection figure from V4 height-sweep data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", default=str(FIGURES_DIR),
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating mask-ratio selection figure...")
    print(f"  Output: {out_dir}")

    try:
        plot_mask_ratio_figure(out_dir)
    except Exception as e:
        print(f"  ERROR: {e}")
        raise

    print("\nDone.")


if __name__ == "__main__":
    main()
