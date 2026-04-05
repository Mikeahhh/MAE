"""
CCPP (Complete Coverage Path Planning) module.

Implements Boustrophedon (lawnmower) decomposition for systematic
UAV area search coverage.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PathPoint:
    """Waypoint with 3D position and timestamp."""
    x: float
    y: float
    z: float
    timestamp: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class PathPlanner:
    """CCPP Boustrophedon path planner."""

    def __init__(
        self,
        area_bounds: Tuple[float, float, float, float],
        flight_height: float = 10.0,
        coverage_radius: float = 15.0,
        overlap_ratio: float = 0.2,
        velocity: float = 5.0
    ):
        """
        Args:
            area_bounds: (x_min, x_max, y_min, y_max)
            flight_height: Flight altitude [m]
            coverage_radius: Detection coverage radius [m]
            overlap_ratio: Strip overlap ratio (0-1)
            velocity: Flight speed [m/s]
        """
        self.x_min, self.x_max, self.y_min, self.y_max = area_bounds
        self.flight_height = flight_height
        self.coverage_radius = coverage_radius
        self.overlap_ratio = overlap_ratio
        self.velocity = velocity
        self.strip_width = coverage_radius * 2 * (1 - overlap_ratio)

    def generate_path(self) -> List[PathPoint]:
        """Generate CCPP waypoints with timestamps."""
        waypoints = []
        y_current = self.y_min
        direction = 1  # 1: left-to-right, -1: right-to-left

        while y_current <= self.y_max:
            if direction == 1:
                waypoints.append((self.x_min, y_current, self.flight_height))
                waypoints.append((self.x_max, y_current, self.flight_height))
            else:
                waypoints.append((self.x_max, y_current, self.flight_height))
                waypoints.append((self.x_min, y_current, self.flight_height))
            y_current += self.strip_width
            direction *= -1

        path_points = []
        t = 0.0
        for i in range(len(waypoints)):
            x, y, z = waypoints[i]
            path_points.append(PathPoint(x, y, z, t))
            if i < len(waypoints) - 1:
                dist = self._euclidean_distance(waypoints[i], waypoints[i + 1])
                t += dist / self.velocity

        return path_points

    def generate_sampling_points(self, sampling_interval: float = 0.1) -> List[PathPoint]:
        """Generate time-interpolated sampling points along the path."""
        path = self.generate_path()
        sampling_points = []

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            segment_time = p2.timestamp - p1.timestamp
            num_samples = int(segment_time / sampling_interval)

            for j in range(num_samples):
                t_ratio = j / num_samples
                x = p1.x + (p2.x - p1.x) * t_ratio
                y = p1.y + (p2.y - p1.y) * t_ratio
                z = p1.z + (p2.z - p1.z) * t_ratio
                t = p1.timestamp + segment_time * t_ratio
                sampling_points.append(PathPoint(x, y, z, t))

        sampling_points.append(path[-1])
        return sampling_points

    def _euclidean_distance(self, p1: Tuple, p2: Tuple) -> float:
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def get_coverage_area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def get_path_length(self) -> float:
        path = self.generate_path()
        total = 0.0
        for i in range(len(path) - 1):
            total += np.linalg.norm(path[i + 1].to_array() - path[i].to_array())
        return total

    def get_mission_time(self) -> float:
        path = self.generate_path()
        return path[-1].timestamp if path else 0.0
