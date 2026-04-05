"""
3D mission scene visualization — Publication quality.

All labels in English for SPAWC paper. Features:
  - Trajectory color-coded by mode (blue=sentinel, orange-red=responder)
  - DOA arrows with confidence-scaled linewidth/color
  - Signal propagation sphere around true source
  - Publication fonts (DejaVu Sans, 8pt)
  - 300 DPI PNG + PDF vector output
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from typing import List, Optional, Tuple
from pathlib import Path


# ── Publication rcParams ───────────────────────────────────────────────────

def set_publication_style():
    """Set matplotlib params for SPAWC publication figures."""
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,       # TrueType for PDF
        'ps.fonttype': 42,
    })


# ── Mode color mapping ────────────────────────────────────────────────────

MODE_COLORS = {
    'sentinel': '#2171B5',    # Blue
    'responder': '#E6550D',   # Orange-red
}


def visualize_mission_3d(
    trajectory: np.ndarray,
    detection_points: List[np.ndarray],
    doa_vectors: List[np.ndarray],
    estimated_position: Optional[np.ndarray],
    true_position: np.ndarray,
    area_bounds: Tuple[float, float, float, float] = (0, 35, 0, 35),
    detection_range: float = 15.0,
    mode_labels: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    title: Optional[str] = None,
):
    """
    Generate publication-quality 3D mission visualization.

    Args:
        trajectory: (N, 3) UAV trajectory
        detection_points: List of (3,) detection positions
        doa_vectors: List of (3,) DOA direction vectors
        estimated_position: (3,) estimated source position (or None)
        true_position: (3,) ground truth source position
        area_bounds: (x_min, x_max, y_min, y_max)
        detection_range: Signal detection range radius [m]
        mode_labels: List of mode strings per trajectory point ("sentinel"/"responder")
        confidences: List of TDOA confidence values per detection
        save_path: Output file path (supports .png and .pdf)
        dpi: Output DPI
        show: Whether to display interactively
        title: Custom figure title
    """
    set_publication_style()

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max, y_min, y_max = area_bounds

    # 1. Search area boundary
    _draw_search_area(ax, area_bounds)

    # 2. Signal propagation sphere
    _draw_signal_range(ax, true_position, detection_range)

    # 3. Trajectory — color-coded by mode
    if len(trajectory) > 0:
        if mode_labels is not None and len(mode_labels) == len(trajectory):
            _draw_mode_colored_trajectory(ax, trajectory, mode_labels)
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color=MODE_COLORS['sentinel'], linewidth=1.2, alpha=0.7,
                    label='UAV trajectory')

        # Start / end markers
        ax.scatter(*trajectory[0], c='#2CA02C', s=60, marker='o',
                   label='Start', edgecolors='black', linewidths=0.8, zorder=5)
        ax.scatter(*trajectory[-1], c='#D62728', s=60, marker='s',
                   label='End', edgecolors='black', linewidths=0.8, zorder=5)

    # 4. Detection points
    if detection_points:
        det_array = np.array(detection_points)
        ax.scatter(det_array[:, 0], det_array[:, 1], det_array[:, 2],
                   c=MODE_COLORS['responder'], s=40, marker='o',
                   alpha=0.8, edgecolors='#8C2D04', linewidths=0.5,
                   label='Detection')

    # 5. DOA vectors — linewidth scaled by confidence
    if detection_points and doa_vectors:
        arrow_length = 5.0
        for i, (pos, doa) in enumerate(zip(detection_points, doa_vectors)):
            lw = 1.5
            alpha = 0.7
            color = '#D62728'

            if confidences is not None and i < len(confidences):
                conf = confidences[i]
                lw = np.clip(conf / 2.0, 0.8, 3.0)
                alpha = np.clip(conf / 4.0, 0.4, 0.95)

            ax.quiver(pos[0], pos[1], pos[2],
                      doa[0], doa[1], doa[2],
                      length=arrow_length, color=color, alpha=alpha,
                      arrow_length_ratio=0.25, linewidth=lw)

        ax.plot([], [], color='#D62728', linewidth=1.5, label='DOA vector')

    # 6. True source position
    ax.scatter(*true_position, c='#FFD700', s=200, marker='*',
               label='True position', edgecolors='#B8860B', linewidths=1.5,
               zorder=10)

    # 7. Estimated position + error line
    if estimated_position is not None:
        ax.scatter(*estimated_position, c='#7CFC00', s=200, marker='*',
                   label='Estimated position', edgecolors='#228B22', linewidths=1.5,
                   zorder=10)

        ax.plot([true_position[0], estimated_position[0]],
                [true_position[1], estimated_position[1]],
                [true_position[2], estimated_position[2]],
                'k--', linewidth=1.5, alpha=0.7, label='Error')

        error = np.linalg.norm(estimated_position - true_position)
        mid = (true_position + estimated_position) / 2
        ax.text(mid[0], mid[1], mid[2] + 1.0,
                f'{error:.2f} m',
                fontsize=8, color='black', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85,
                          edgecolor='gray', linewidth=0.5))

    # Axes
    ax.set_xlabel('X (m)', fontsize=9, labelpad=5)
    ax.set_ylabel('Y (m)', fontsize=9, labelpad=5)
    ax.set_zlabel('Z (m)', fontsize=9, labelpad=5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(0, 15)

    # Title
    if title is None:
        title = 'UAV Search-and-Rescue Mission'
        if estimated_position is not None:
            error = np.linalg.norm(estimated_position - true_position)
            title += f'\nLoc. error: {error:.2f} m | Detections: {len(detection_points)}'
    ax.set_title(title, fontsize=10, weight='bold', pad=15)

    ax.legend(loc='upper left', fontsize=7, framealpha=0.9,
              edgecolor='gray', fancybox=False)
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        save_p = Path(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        # Also save PDF for vector graphics if PNG was requested
        if save_p.suffix.lower() == '.png':
            pdf_path = save_p.with_suffix('.pdf')
            plt.savefig(str(pdf_path), bbox_inches='tight')
        print(f"  3D visualization saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def _draw_mode_colored_trajectory(ax, trajectory: np.ndarray, mode_labels: List[str]):
    """Draw trajectory segments colored by sentinel (blue) / responder (orange-red)."""
    n = len(trajectory)
    for i in range(n - 1):
        color = MODE_COLORS.get(mode_labels[i], MODE_COLORS['sentinel'])
        ax.plot(trajectory[i:i + 2, 0],
                trajectory[i:i + 2, 1],
                trajectory[i:i + 2, 2],
                color=color, linewidth=1.2, alpha=0.7)

    # Legend entries
    ax.plot([], [], color=MODE_COLORS['sentinel'], linewidth=2, label='Sentinel mode')
    ax.plot([], [], color=MODE_COLORS['responder'], linewidth=2, label='Responder mode')


def _draw_search_area(ax, area_bounds: Tuple[float, float, float, float]):
    """Draw ground-plane search area rectangle."""
    x_min, x_max, y_min, y_max = area_bounds

    corners = np.array([
        [x_min, y_min, 0], [x_max, y_min, 0],
        [x_max, y_max, 0], [x_min, y_max, 0], [x_min, y_min, 0]
    ])
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            'k--', linewidth=0.8, alpha=0.4)

    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                      facecolor='lightgray', alpha=0.15)
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")


def _draw_signal_range(ax, center: np.ndarray, radius: float):
    """Draw semi-transparent signal propagation sphere.

    Simulation visualization only — NOT used in SPAWC paper figures.
    Paper figures use plot_3d_snr_flyover.py (no sphere).
    """
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    z = np.maximum(z, 0)

    ax.plot_surface(x, y, z, color='#FFD700', alpha=0.08,
                    linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(x, y, z, color='#FFA500', alpha=0.15,
                      linewidth=0.3, rstride=3, cstride=3)
