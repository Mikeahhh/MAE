"""
Plot height-based detection accuracy vs mask_ratio.

Reads JSON results from eval_height_sweep.py and produces two publication-quality
figures (desert + forest), one per scenario. Each line = one flight height.

Usage:
    python -m SpecMae.scripts.eval.plot_height_detection
    python -m SpecMae.scripts.eval.plot_height_detection --scenario desert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────
_SPEC = Path(__file__).resolve().parents[2]  # SpecMae/

from SpecMae.simulation.visualization.scene_3d import set_publication_style

RESULTS_ROOT = _SPEC / "results"
FIGURES_DIR = _SPEC / "results" / "figures"

# ═════════════════════════════════════════════════════════════════════════
#  Color schemes — gradient: low height (cool) → high height (warm)
# ═════════════════════════════════════════════════════════════════════════

COLORS = {
    "desert": ["#1a5276", "#2e86c1", "#e67e22", "#c0392b"],
    "forest": ["#145a32", "#27ae60", "#e67e22", "#c0392b"],
}


# ═════════════════════════════════════════════════════════════════════════
#  Load results
# ═════════════════════════════════════════════════════════════════════════

def load_results(scenario: str, results_dir: Path | None = None) -> dict:
    """Load height sweep JSON for one scenario."""
    if results_dir is None:
        results_dir = RESULTS_ROOT / f"height_sweep_{scenario}"
    path = results_dir / f"height_sweep_{scenario}.json"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    with open(path) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════
#  Plot one scenario
# ═════════════════════════════════════════════════════════════════════════

METRIC_LABELS = {
    "detection_accuracy": "Detection Accuracy",
    "presence_accuracy": "Presence Accuracy",
}


def plot_scenario(
    data: dict,
    out_dir: Path | None = None,
    metric: str = "presence_accuracy",
) -> list[Path]:
    """Create detection/presence accuracy vs mask_ratio plot for one scenario.

    Args:
        metric: JSON key to plot — "presence_accuracy" or "detection_accuracy".

    Returns list of saved file paths.
    """
    set_publication_style()

    scenario = data["scenario"]
    heights_info = data["heights"]  # {str(h): {flight_height_m, mic_height_m, peak_snr_db}}
    results = data["results"]       # [{mask_ratio, per_height: {h: {...}}}]
    colors = COLORS.get(scenario, COLORS["desert"])

    if out_dir is None:
        out_dir = FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract sorted height list
    heights = sorted(int(h) for h in heights_info.keys())

    # Build arrays: mask_ratios × heights → metric + std
    mask_ratios = sorted(r["mask_ratio"] for r in results)
    mr_to_idx = {mr: i for i, mr in enumerate(mask_ratios)}

    # accuracy_matrix[h][mr_idx], std_matrix[h][mr_idx]
    accuracy_matrix = {h: [None] * len(mask_ratios) for h in heights}
    std_matrix = {h: [0.0] * len(mask_ratios) for h in heights}

    std_key = metric.replace("accuracy", "std")  # e.g. "presence_std"

    for r in results:
        mr = r["mask_ratio"]
        mr_idx = mr_to_idx[mr]
        for h_str, h_data in r["per_height"].items():
            h = int(h_str)
            if h in accuracy_matrix:
                # Fallback to detection_accuracy for backward compat with V3 results
                accuracy_matrix[h][mr_idx] = h_data.get(
                    metric, h_data.get("detection_accuracy")
                )
                std_matrix[h][mr_idx] = h_data.get(std_key, 0.0)

    # ── Create figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    mr_arr = np.array(mask_ratios)

    # Find global best (height, mask_ratio) point
    best_acc = -1.0
    best_height = heights[0]
    best_mr_idx = 0
    for h in heights:
        for i, v in enumerate(accuracy_matrix[h]):
            if v is not None and v > best_acc:
                best_acc = v
                best_height = h
                best_mr_idx = i

    for h_idx, height in enumerate(heights):
        accs = np.array(accuracy_matrix[height], dtype=float)
        stds = np.array(std_matrix[height], dtype=float)
        snr_db = heights_info[str(height)]["peak_snr_db"]

        is_best = (height == best_height)
        color = "#c0392b" if is_best else colors[h_idx % len(colors)]
        lw = 2.5 if is_best else 1.2
        alpha = 1.0 if is_best else 0.3
        ls = "-" if is_best else "--"
        zorder = 10 if is_best else 3
        ms = 5 if is_best else 3
        suffix = " (best)" if is_best else ""
        has_std = np.any(stds > 0)

        ax.plot(
            mr_arr, accs, color=color, linewidth=lw, zorder=zorder,
            alpha=alpha, linestyle=ls, marker="o", markersize=ms,
            label=f"h = {height} m (SNR {snr_db:+.0f} dB){suffix}",
        )

    # Star marker on the global best point
    ax.plot(
        mr_arr[best_mr_idx], best_acc,
        marker="*", markersize=18, color="#FFD700",
        zorder=20, markeredgecolor="#B8860B", markeredgewidth=0.8,
    )

    # ── Axes & labels ────────────────────────────────────────────────────
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel(f"{metric_label} (%)")
    ax.set_title(
        f"{metric_label} vs. Mask Ratio — {scenario.capitalize()} Scenario",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(min(mask_ratios) - 0.02, max(mask_ratios) + 0.02)
    ax.set_ylim(0, 105)

    ax.set_xticks([0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xticklabels(["0.0", "0.1", "0.3", "0.5", "0.7", "0.9"])

    ax.set_yticks(range(0, 110, 10))

    ax.grid(True, alpha=0.2, linewidth=0.5)

    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.7")

    # ── Annotation removed per user request ─────────────────────────────

    fig.tight_layout()

    # ── Save — filename includes metric prefix ───────────────────────────
    metric_prefix = metric.replace("_accuracy", "")
    saved = []
    for ext in ("png",):
        path = out_dir / f"fig_height_{metric_prefix}_{scenario}.{ext}"
        fig.savefig(path, dpi=300)
        saved.append(path)
        print(f"  Saved: {path}")

    plt.close(fig)
    return saved


# ═════════════════════════════════════════════════════════════════════════
#  Combined plot (desert + forest in one figure)
# ═════════════════════════════════════════════════════════════════════════

# Distinct visual styles per height (high contrast, paper-ready)
_HEIGHT_STYLES = [
    {"color": "#c0392b", "ls": "-",  "marker": "o", "ms": 6, "lw": 2.2},  # best: red solid circle
    {"color": "#2980b9", "ls": "--", "marker": "s", "ms": 5, "lw": 1.5},  # blue dashed square
    {"color": "#27ae60", "ls": "-.", "marker": "^", "ms": 5, "lw": 1.5},  # green dash-dot triangle
    {"color": "#8e44ad", "ls": ":",  "marker": "D", "ms": 4, "lw": 1.5},  # purple dotted diamond
]


def plot_combined(
    data_desert: dict,
    data_forest: dict,
    out_dir: Path | None = None,
    metric: str = "presence_accuracy",
) -> list[Path]:
    """Combined desert + forest presence accuracy in ONE plot.

    Desert = solid lines, Forest = dashed lines.
    Same color per height rank (best=red, 2nd=blue, 3rd=green, 4th=purple).
    No SNR in legend — advisor requirement.
    """
    set_publication_style()

    if out_dir is None:
        out_dir = FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    # Desert = black, Forest = blue; line style + marker distinguish heights
    _SCENARIO_STYLES = {
        "Desert": {
            "base_color": "#000000",
            "styles": [
                {"ls": "-",  "marker": "o", "ms": 7, "mfc": "k"},   # best: filled
                {"ls": "--", "marker": "s", "ms": 6, "mfc": "none"},
                {"ls": "-.", "marker": "^", "ms": 6, "mfc": "none"},
                {"ls": ":",  "marker": "D", "ms": 5, "mfc": "none"},
            ],
        },
        "Forest": {
            "base_color": "#1a5276",
            "styles": [
                {"ls": "-",  "marker": "o", "ms": 7, "mfc": "#1a5276"},
                {"ls": "--", "marker": "s", "ms": 6, "mfc": "none"},
                {"ls": "-.", "marker": "^", "ms": 6, "mfc": "none"},
                {"ls": ":",  "marker": "D", "ms": 5, "mfc": "none"},
            ],
        },
    }

    for data, scenario_label in [
        (data_desert, "Desert"),
        (data_forest, "Forest"),
    ]:
        heights_info = data["heights"]
        results = data["results"]
        sc = _SCENARIO_STYLES[scenario_label]
        color = sc["base_color"]

        heights = sorted(int(h) for h in heights_info.keys())
        mask_ratios = sorted(r["mask_ratio"] for r in results)
        mr_to_idx = {mr: i for i, mr in enumerate(mask_ratios)}

        accuracy_matrix = {h: [None] * len(mask_ratios) for h in heights}
        for r in results:
            mr_idx = mr_to_idx[r["mask_ratio"]]
            for h_str, h_data in r["per_height"].items():
                h = int(h_str)
                if h in accuracy_matrix:
                    accuracy_matrix[h][mr_idx] = h_data.get(
                        metric, h_data.get("detection_accuracy"))

        mr_arr = np.array(mask_ratios)

        # Find best point
        best_acc, best_height, best_mr_idx = -1.0, heights[0], 0
        for h in heights:
            for i, v in enumerate(accuracy_matrix[h]):
                if v is not None and v > best_acc:
                    best_acc, best_height, best_mr_idx = v, h, i

        # Sort heights so best is first
        sorted_heights = [best_height] + [h for h in heights if h != best_height]

        for style_idx, height in enumerate(sorted_heights):
            accs = np.array(accuracy_matrix[height], dtype=float)
            is_best = (height == best_height)
            s = sc["styles"][style_idx % len(sc["styles"])]
            lw = 2.0 if is_best else 1.3
            alpha = 1.0 if is_best else 0.6
            zorder = 10 if is_best else 3
            suffix = " *" if is_best else ""

            ax.plot(mr_arr, accs, color=color, linewidth=lw,
                    zorder=zorder, alpha=alpha, linestyle=s["ls"],
                    marker=s["marker"], markersize=s["ms"],
                    markerfacecolor=s["mfc"], markeredgecolor=color,
                    markeredgewidth=1.0,
                    label=f"{scenario_label} h={height}m{suffix}")

        # Gold star on best
        ax.plot(mr_arr[best_mr_idx], best_acc, marker="*", markersize=16,
                color="#FFD700", zorder=20, markeredgecolor="#B8860B",
                markeredgewidth=0.8)

    ax.set_xlabel("Mask Ratio", fontsize=10)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=10)
    ax.set_title(f"{metric_label} vs. Mask Ratio", fontsize=11, fontweight="bold")
    ax.set_xlim(min(mask_ratios) - 0.02, max(mask_ratios) + 0.02)
    ax.set_ylim(0, 105)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticks(range(0, 110, 10))
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="0.7",
              fontsize=7, ncol=2, columnspacing=1.0, handlelength=2.5)

    fig.tight_layout()

    metric_prefix = metric.replace("_accuracy", "")
    path = out_dir / f"fig_height_{metric_prefix}_combined.png"
    fig.savefig(path, dpi=300)
    print(f"  Saved: {path}")
    plt.close(fig)
    return [path]


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot height-based detection accuracy figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario", default=None, choices=["desert", "forest"],
        help="Plot one scenario (default: both if results exist)",
    )
    parser.add_argument(
        "--metric", default="presence_accuracy",
        choices=["presence_accuracy", "detection_accuracy"],
        help="Which accuracy metric to plot",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Override results directory",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Override figure output directory",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Generate combined desert+forest plot (no SNR in legend)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.combined:
        data_desert = load_results("desert")
        data_forest = load_results("forest")
        plot_combined(data_desert, data_forest, out_dir, metric=args.metric)
    else:
        if args.scenario:
            scenarios = [args.scenario]
        else:
            scenarios = []
            for s in ("desert", "forest"):
                json_path = RESULTS_ROOT / f"height_sweep_{s}" / f"height_sweep_{s}.json"
                if json_path.exists():
                    scenarios.append(s)
            if not scenarios:
                print("No height_sweep results found. Run eval_height_sweep.py first.")
                sys.exit(1)

        print(f"Plotting {len(scenarios)} scenario(s): {scenarios}  metric={args.metric}")
        for scenario in scenarios:
            results_dir = Path(args.results_dir) if args.results_dir else None
            data = load_results(scenario, results_dir)
            plot_scenario(data, out_dir, metric=args.metric)

    print("\nDone.")


if __name__ == "__main__":
    main()
