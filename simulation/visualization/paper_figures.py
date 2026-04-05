"""
Publication-quality multi-panel figures for SPAWC paper.

All labels in English. Outputs at 300 DPI (PNG) + vector PDF.

Figure types:
  1. Mission overview (2x2 panel)
  2. Desert vs Forest comparison (1x2)
  3. DOA error vs SNR curves
  4. Localization error convergence
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from .scene_3d import set_publication_style, MODE_COLORS, _draw_search_area, _draw_signal_range


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 1: Multi-panel mission overview (2x2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_mission_overview(
    trajectory: np.ndarray,
    detection_points: List[np.ndarray],
    doa_vectors: List[np.ndarray],
    estimated_position: Optional[np.ndarray],
    true_position: np.ndarray,
    anomaly_scores: Optional[List[float]] = None,
    anomaly_threshold: Optional[float] = None,
    localization_errors: Optional[List[float]] = None,
    area_bounds: Tuple[float, float, float, float] = (0, 35, 0, 35),
    detection_range: float = 15.0,
    mode_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    2x2 mission overview figure:
      (a) 3D trajectory with sentinel/responder coloring
      (b) 2D top-down ground track with DOA rays
      (c) Localization error convergence vs detection count
      (d) Anomaly score timeline with threshold
    """
    set_publication_style()

    fig = plt.figure(figsize=(7.5, 6.5))
    x_min, x_max, y_min, y_max = area_bounds

    # ── Panel (a): 3D trajectory ──────────────────────────────────────────
    ax_a = fig.add_subplot(221, projection='3d')

    if len(trajectory) > 0:
        if mode_labels is not None and len(mode_labels) == len(trajectory):
            for i in range(len(trajectory) - 1):
                color = MODE_COLORS.get(mode_labels[i], MODE_COLORS['sentinel'])
                ax_a.plot(trajectory[i:i + 2, 0], trajectory[i:i + 2, 1],
                          trajectory[i:i + 2, 2], color=color, linewidth=1.0, alpha=0.7)
        else:
            ax_a.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                      color=MODE_COLORS['sentinel'], linewidth=1.0, alpha=0.7)

    _draw_search_area(ax_a, area_bounds)
    _draw_signal_range(ax_a, true_position, detection_range)

    if detection_points:
        det = np.array(detection_points)
        ax_a.scatter(det[:, 0], det[:, 1], det[:, 2],
                     c=MODE_COLORS['responder'], s=20, alpha=0.7, edgecolors='#8C2D04', linewidths=0.3)

    ax_a.scatter(*true_position, c='#FFD700', s=120, marker='*',
                 edgecolors='#B8860B', linewidths=1, zorder=10)
    if estimated_position is not None:
        ax_a.scatter(*estimated_position, c='#7CFC00', s=120, marker='*',
                     edgecolors='#228B22', linewidths=1, zorder=10)

    ax_a.set_xlabel('X (m)', fontsize=7)
    ax_a.set_ylabel('Y (m)', fontsize=7)
    ax_a.set_zlabel('Z (m)', fontsize=7)
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_ylim(y_min, y_max)
    ax_a.set_zlim(0, 15)
    ax_a.view_init(elev=30, azim=45)
    ax_a.set_title('(a) 3D Trajectory', fontsize=9, weight='bold')

    # ── Panel (b): 2D top-down with DOA rays ─────────────────────────────
    ax_b = fig.add_subplot(222)

    # Search area
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                      facecolor='#F0F0F0', edgecolor='black', linewidth=0.8, linestyle='--')
    ax_b.add_patch(rect)

    # Signal range circle
    circle = plt.Circle((true_position[0], true_position[1]), detection_range,
                         color='#FFD700', alpha=0.15, linewidth=0.5, linestyle='-',
                         edgecolor='#FFA500')
    ax_b.add_patch(circle)

    # Trajectory
    if len(trajectory) > 0:
        if mode_labels is not None and len(mode_labels) == len(trajectory):
            for i in range(len(trajectory) - 1):
                color = MODE_COLORS.get(mode_labels[i], MODE_COLORS['sentinel'])
                ax_b.plot(trajectory[i:i + 2, 0], trajectory[i:i + 2, 1],
                          color=color, linewidth=0.8, alpha=0.6)
        else:
            ax_b.plot(trajectory[:, 0], trajectory[:, 1],
                      color=MODE_COLORS['sentinel'], linewidth=0.8, alpha=0.6)

    # DOA rays
    if detection_points and doa_vectors:
        ray_length = 20.0
        for pos, doa in zip(detection_points, doa_vectors):
            end = pos[:2] + doa[:2] / np.linalg.norm(doa[:2] + 1e-8) * ray_length
            ax_b.plot([pos[0], end[0]], [pos[1], end[1]],
                      color='#D62728', linewidth=0.8, alpha=0.5)

    # Markers
    ax_b.plot(true_position[0], true_position[1], '*', color='#FFD700',
              markersize=12, markeredgecolor='#B8860B', markeredgewidth=1)
    if estimated_position is not None:
        ax_b.plot(estimated_position[0], estimated_position[1], '*', color='#7CFC00',
                  markersize=12, markeredgecolor='#228B22', markeredgewidth=1)

    ax_b.set_xlim(x_min - 1, x_max + 1)
    ax_b.set_ylim(y_min - 1, y_max + 1)
    ax_b.set_xlabel('X (m)', fontsize=7)
    ax_b.set_ylabel('Y (m)', fontsize=7)
    ax_b.set_aspect('equal')
    ax_b.set_title('(b) Top-down View with DOA Rays', fontsize=9, weight='bold')
    ax_b.grid(True, alpha=0.2)

    # ── Panel (c): Localization error convergence ────────────────────────
    ax_c = fig.add_subplot(223)

    if localization_errors is not None and len(localization_errors) > 0:
        n_det = np.arange(1, len(localization_errors) + 1)
        ax_c.plot(n_det, localization_errors, 'o-', color='#2171B5',
                  linewidth=1.5, markersize=4)
        ax_c.axhline(y=5.0, color='#D62728', linestyle='--', linewidth=0.8,
                      label='Target: 5 m')
    else:
        ax_c.text(0.5, 0.5, 'No data', transform=ax_c.transAxes,
                  ha='center', va='center', fontsize=8, color='gray')

    ax_c.set_xlabel('Number of detections', fontsize=7)
    ax_c.set_ylabel('Localization error (m)', fontsize=7)
    ax_c.set_title('(c) Error Convergence', fontsize=9, weight='bold')
    ax_c.legend(fontsize=6)
    ax_c.grid(True, alpha=0.2)

    # ── Panel (d): Anomaly score timeline ────────────────────────────────
    ax_d = fig.add_subplot(224)

    if anomaly_scores is not None and len(anomaly_scores) > 0:
        indices = np.arange(len(anomaly_scores))
        ax_d.plot(indices, anomaly_scores, '-', color='#2171B5',
                  linewidth=0.8, alpha=0.8)
        if anomaly_threshold is not None:
            ax_d.axhline(y=anomaly_threshold, color='#D62728', linestyle='--',
                          linewidth=1.0, label=f'Threshold ({anomaly_threshold:.3f})')
    else:
        ax_d.text(0.5, 0.5, 'No data', transform=ax_d.transAxes,
                  ha='center', va='center', fontsize=8, color='gray')

    ax_d.set_xlabel('Sample index', fontsize=7)
    ax_d.set_ylabel('Anomaly score', fontsize=7)
    ax_d.set_title('(d) Anomaly Score Timeline', fontsize=9, weight='bold')
    ax_d.legend(fontsize=6)
    ax_d.grid(True, alpha=0.2)

    plt.tight_layout()
    _save_figure(fig, save_path, dpi, show)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 2: Desert vs Forest comparison (1x2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_desert_forest_comparison(
    desert_result,
    forest_result,
    area_bounds: Tuple[float, float, float, float] = (0, 35, 0, 35),
    detection_range: float = 15.0,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    Side-by-side 3D views of desert (RT60=0s) and forest (RT60=0.6s).

    Args:
        desert_result: SimulationResult from desert scenario
        forest_result: SimulationResult from forest scenario
    """
    set_publication_style()

    fig = plt.figure(figsize=(8, 4))

    for idx, (result, env_label) in enumerate([
        (desert_result, 'Desert (RT60=0 s)'),
        (forest_result, 'Forest (RT60=0.6 s)'),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        _draw_search_area(ax, area_bounds)
        _draw_signal_range(ax, result.true_position, detection_range)

        # Trajectory
        if result.trajectory is not None and len(result.trajectory) > 0:
            traj = result.trajectory
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=MODE_COLORS['sentinel'], linewidth=0.8, alpha=0.6)

        # Detection points
        if result.detection_positions:
            det = np.array(result.detection_positions)
            ax.scatter(det[:, 0], det[:, 1], det[:, 2],
                       c=MODE_COLORS['responder'], s=25, alpha=0.7,
                       edgecolors='#8C2D04', linewidths=0.3)

        # DOA vectors
        if result.detection_positions and result.doa_vectors:
            for pos, doa in zip(result.detection_positions, result.doa_vectors):
                ax.quiver(pos[0], pos[1], pos[2], doa[0], doa[1], doa[2],
                          length=4.0, color='#D62728', alpha=0.5,
                          arrow_length_ratio=0.25, linewidth=1.0)

        # True + estimated
        ax.scatter(*result.true_position, c='#FFD700', s=120, marker='*',
                   edgecolors='#B8860B', linewidths=1, zorder=10)
        if result.estimated_position is not None:
            ax.scatter(*result.estimated_position, c='#7CFC00', s=120, marker='*',
                       edgecolors='#228B22', linewidths=1, zorder=10)

        x_min, x_max, y_min, y_max = area_bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(0, 15)
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Y (m)', fontsize=7)
        ax.set_zlabel('Z (m)', fontsize=7)
        ax.view_init(elev=25, azim=45)

        err_str = f'Err: {result.localization_error:.2f} m' if result.estimated_position is not None else 'N/A'
        doa_str = ''
        if result.doa_errors_deg:
            doa_str = f' | DOA err: {np.mean(result.doa_errors_deg):.1f}°'
        ax.set_title(f'{env_label}\n{err_str}{doa_str}', fontsize=8, weight='bold')

    plt.tight_layout()
    _save_figure(fig, save_path, dpi, show)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 3: DOA error vs SNR / scenario curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_doa_error_curves(
    data: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    DOA error curves for multiple scenarios.

    Args:
        data: Dict mapping scenario name to {"snr": [...], "doa_error": [...]}
              or {"distance": [...], "doa_error": [...]}
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(4, 3))

    colors = ['#2171B5', '#E6550D', '#2CA02C', '#9467BD']
    markers = ['o', 's', '^', 'D']

    for i, (label, d) in enumerate(data.items()):
        x_key = 'snr' if 'snr' in d else 'distance'
        x_vals = d[x_key]
        y_vals = d['doa_error']

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(x_vals, y_vals, f'-{marker}', color=color, linewidth=1.2,
                markersize=4, label=label)

    # Reference lines
    ax.axhline(y=5.0, color='gray', linestyle=':', linewidth=0.8, label='5° general target')
    ax.axhline(y=2.1, color='gray', linestyle='--', linewidth=0.8, label='2.1° forest target')

    x_label = 'SNR (dB)' if 'snr' in list(data.values())[0] else 'Distance (m)'
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel('DOA Error (deg)', fontsize=8)
    ax.set_title('DOA Estimation Accuracy', fontsize=9, weight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save_figure(fig, save_path, dpi, show)


# ═══════════════════════════════════════════════════════════════════════════
#  Figure 4: Localization error vs number of Monte Carlo runs
# ═══════════════════════════════════════════════════════════════════════════

def plot_localization_error_distribution(
    errors: Dict[str, List[float]],
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    Box/violin plot of localization errors across Monte Carlo runs.

    Args:
        errors: Dict mapping scenario name to list of localization errors [m]
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(4, 3))

    labels = list(errors.keys())
    data = [errors[k] for k in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)

    colors = ['#2171B5', '#E6550D', '#2CA02C', '#9467BD']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.6)

    ax.axhline(y=5.0, color='#D62728', linestyle='--', linewidth=0.8,
               label='Target: 5 m')

    ax.set_ylabel('Localization Error (m)', fontsize=8)
    ax.set_title('Localization Accuracy (Monte Carlo)', fontsize=9, weight='bold')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    _save_figure(fig, save_path, dpi, show)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_figure(fig, save_path: Optional[str], dpi: int, show: bool):
    """Save figure as PNG + PDF, then show or close."""
    if save_path:
        save_p = Path(save_path)
        fig.savefig(str(save_p), dpi=dpi, bbox_inches='tight')
        if save_p.suffix.lower() == '.png':
            fig.savefig(str(save_p.with_suffix('.pdf')), bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
