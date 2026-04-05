"""
Mission data logger.

Records UAV trajectory, detection events, and mission summaries
for analysis and visualization.
"""

import json
import csv
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import datetime


@dataclass
class DetectionEvent:
    """Detection event record."""
    timestamp: float                           # [s]
    position: np.ndarray                       # (3,) UAV position [m]
    anomaly_score: float                       # Anomaly score (reconstruction-based)
    detection_method: str = "reconstruction"   # always reconstruction-based anomaly scoring
    mode: str = "sentinel"                     # "sentinel" or "responder"
    tdoa: Optional[np.ndarray] = None          # (8,) TDOA vector [s]
    doa_local: Optional[np.ndarray] = None     # (3,) local DOA
    doa_global: Optional[np.ndarray] = None    # (3,) global DOA
    confidence: Optional[float] = None         # TDOA confidence
    residual_error: Optional[float] = None     # DOA fit residual
    doa_error_deg: Optional[float] = None      # DOA angular error vs ground truth [deg]

    def to_dict(self):
        d = {
            'timestamp': self.timestamp,
            'position': self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            'anomaly_score': self.anomaly_score,
            'detection_method': self.detection_method,
            'mode': self.mode,
        }

        if self.tdoa is not None:
            d['tdoa'] = self.tdoa.tolist() if isinstance(self.tdoa, np.ndarray) else self.tdoa
        if self.doa_local is not None:
            d['doa_local'] = self.doa_local.tolist() if isinstance(self.doa_local, np.ndarray) else self.doa_local
        if self.doa_global is not None:
            d['doa_global'] = self.doa_global.tolist() if isinstance(self.doa_global, np.ndarray) else self.doa_global
        if self.confidence is not None:
            d['confidence'] = self.confidence
        if self.residual_error is not None:
            d['residual_error'] = self.residual_error
        if self.doa_error_deg is not None:
            d['doa_error_deg'] = self.doa_error_deg

        return d


class DataLogger:
    """Mission data logger with CSV/JSON export."""

    def __init__(self, output_dir: str = "./simulation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trajectory = []       # UAV trajectory points
        self.detections = []       # Detection events
        self.mode_history = []     # (timestamp, mode) tracking
        self.mission_start_time = None
        self.mission_end_time = None

    def log_trajectory_point(self, timestamp: float, position: np.ndarray, mode: str = "sentinel"):
        """Record a trajectory point with current mode."""
        self.trajectory.append({
            'timestamp': timestamp,
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'mode': mode
        })

    def log_detection(self, event: DetectionEvent):
        self.detections.append(event)

    def log_mode_change(self, timestamp: float, mode: str):
        self.mode_history.append({'timestamp': timestamp, 'mode': mode})

    def start_mission(self):
        self.mission_start_time = datetime.datetime.now()

    def end_mission(self):
        self.mission_end_time = datetime.datetime.now()

    def save_trajectory(self, filename: str = "trajectory.csv"):
        filepath = self.output_dir / filename
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'x', 'y', 'z', 'mode'])
            writer.writeheader()
            writer.writerows(self.trajectory)
        print(f"  Trajectory saved: {filepath}")

    def save_detections(self, filename: str = "detections.json"):
        filepath = self.output_dir / filename
        detections_dict = [event.to_dict() for event in self.detections]
        with open(filepath, 'w') as f:
            json.dump(detections_dict, f, indent=2)
        print(f"  Detections saved: {filepath}")

    def save_summary(
        self,
        estimated_position: Optional[np.ndarray] = None,
        true_position: Optional[np.ndarray] = None,
        filename: str = "mission_summary.json"
    ):
        filepath = self.output_dir / filename
        summary = {
            'mission_start': self.mission_start_time.isoformat() if self.mission_start_time else None,
            'mission_end': self.mission_end_time.isoformat() if self.mission_end_time else None,
            'num_trajectory_points': len(self.trajectory),
            'num_detections': len(self.detections),
            'mode_changes': self.mode_history,
        }

        if self.mission_start_time and self.mission_end_time:
            summary['mission_duration_s'] = (self.mission_end_time - self.mission_start_time).total_seconds()

        if estimated_position is not None:
            summary['estimated_position'] = estimated_position.tolist()
        if true_position is not None:
            summary['true_position'] = true_position.tolist()
        if estimated_position is not None and true_position is not None:
            summary['localization_error_m'] = float(np.linalg.norm(estimated_position - true_position))

        # DOA error statistics
        doa_errors = [d.doa_error_deg for d in self.detections if d.doa_error_deg is not None]
        if doa_errors:
            summary['doa_error_mean_deg'] = float(np.mean(doa_errors))
            summary['doa_error_std_deg'] = float(np.std(doa_errors))
            summary['doa_error_max_deg'] = float(np.max(doa_errors))

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {filepath}")

    def save_all(
        self,
        estimated_position: Optional[np.ndarray] = None,
        true_position: Optional[np.ndarray] = None
    ):
        self.save_trajectory()
        self.save_detections()
        self.save_summary(estimated_position, true_position)

    def get_detection_positions(self) -> List[np.ndarray]:
        return [event.position for event in self.detections]

    def get_doa_vectors(self) -> List[np.ndarray]:
        return [event.doa_global for event in self.detections if event.doa_global is not None]

    def get_doa_errors(self) -> List[float]:
        return [event.doa_error_deg for event in self.detections if event.doa_error_deg is not None]

    def get_mode_at_time(self, timestamp: float) -> str:
        """Get the mode active at a given timestamp."""
        mode = "sentinel"
        for entry in self.mode_history:
            if entry['timestamp'] <= timestamp:
                mode = entry['mode']
        return mode
