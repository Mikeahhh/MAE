"""
Multi-point ray-intersection triangulation module.

Estimates source position from multiple DOA observations using
weighted least-squares ray crossing: A = sum_i w_i * (I - d_i d_i^T).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TriangulationResult:
    """Triangulation result."""
    source_position: np.ndarray  # (3,) estimated source position [m]
    residual_error: float        # Mean ray-to-point distance [m]
    n_rays: int                  # Number of rays used
    condition_number: float      # Numerical stability indicator


class Triangulator:
    """Multi-point ray-crossing triangulator."""

    def __init__(self, min_rays: int = 3, max_residual: float = 10.0):
        self.min_rays = min_rays
        self.max_residual = max_residual

    def triangulate(
        self,
        detection_points: List[np.ndarray],
        doa_vectors: List[np.ndarray],
        weights: Optional[List[float]] = None
    ) -> TriangulationResult:
        if len(detection_points) < self.min_rays:
            raise ValueError(
                f"Need at least {self.min_rays} detection points, got {len(detection_points)}"
            )

        source_pos, residual, cond = triangulate_source(
            detection_points, doa_vectors, weights
        )

        return TriangulationResult(source_pos, residual, len(detection_points), cond)


def triangulate_source(
    detection_points: List[np.ndarray],
    doa_vectors: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Ray-intersection least-squares localization.

    Minimizes: sum_i w_i * ||s - (p_i + t_i * d_i)||^2

    Closed-form:
        A = sum_i w_i * (I - d_i d_i^T)
        b = sum_i w_i * (I - d_i d_i^T) * p_i
        s = A^{-1} b

    Returns:
        (source_position, residual_error, condition_number)
    """
    n_rays = len(detection_points)

    if weights is None:
        weights = [1.0] * n_rays

    P = np.array(detection_points)
    D = np.array(doa_vectors)
    W = np.array(weights)

    # Normalize DOA vectors
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    I = np.eye(3)
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(n_rays):
        d_i = D[i]
        p_i = P[i]
        w_i = W[i]
        proj_matrix = I - np.outer(d_i, d_i)
        A += w_i * proj_matrix
        b += w_i * (proj_matrix @ p_i)

    try:
        source_position = np.linalg.solve(A, b)
        condition_number = np.linalg.cond(A)
    except np.linalg.LinAlgError:
        source_position = np.linalg.lstsq(A, b, rcond=None)[0]
        condition_number = np.inf

    # Compute residuals (point-to-ray distances)
    residuals = []
    for i in range(n_rays):
        v = source_position - P[i]
        t_i = np.dot(v, D[i])
        closest_point = P[i] + t_i * D[i]
        dist = np.linalg.norm(source_position - closest_point)
        residuals.append(dist)

    residual_error = np.mean(residuals)
    return source_position, residual_error, condition_number


def triangulate_2d(
    detection_points: List[np.ndarray],
    doa_vectors: List[np.ndarray]
) -> Tuple[np.ndarray, float]:
    """2D triangulation in XY plane (assumes ground source Z=0)."""
    P_2d = np.array([p[:2] for p in detection_points])
    D_2d = np.array([d[:2] for d in doa_vectors])
    D_2d = D_2d / np.linalg.norm(D_2d, axis=1, keepdims=True)

    n_rays = len(detection_points)
    I = np.eye(2)
    A = np.zeros((2, 2))
    b = np.zeros(2)

    for i in range(n_rays):
        proj_matrix = I - np.outer(D_2d[i], D_2d[i])
        A += proj_matrix
        b += proj_matrix @ P_2d[i]

    source_position_2d = np.linalg.solve(A, b)

    residuals = []
    for i in range(n_rays):
        v = source_position_2d - P_2d[i]
        t_i = np.dot(v, D_2d[i])
        closest_point = P_2d[i] + t_i * D_2d[i]
        dist = np.linalg.norm(source_position_2d - closest_point)
        residuals.append(dist)

    return source_position_2d, np.mean(residuals)


def calculate_geometric_dilution_of_precision(
    detection_points: List[np.ndarray],
    doa_vectors: List[np.ndarray]
) -> float:
    """
    Compute GDOP (Geometric Dilution of Precision).

    GDOP < 2: excellent, 2-5: good, 5-10: moderate, >=10: poor
    """
    D = np.array(doa_vectors)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)

    I = np.eye(3)
    A = np.zeros((3, 3))

    for i in range(len(detection_points)):
        proj_matrix = I - np.outer(D[i], D[i])
        A += proj_matrix

    try:
        A_inv = np.linalg.inv(A)
        return np.sqrt(np.trace(A_inv))
    except np.linalg.LinAlgError:
        return np.inf
