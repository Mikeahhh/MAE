"""
DOA (Direction of Arrival) calculation module.

Computes source direction from TDOA measurements using spherical
least-squares estimation with a 9-mic UCA array.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class DOAResult:
    """DOA calculation result."""
    direction: np.ndarray      # (3,) unit direction vector
    residual_error: float      # Fit residual [s]
    azimuth: float             # Azimuth [deg]
    elevation: float           # Elevation [deg]


class DOACalculator:
    """DOA calculator using spherical least-squares."""

    def __init__(self, mic_positions: np.ndarray, sound_speed: float = 343.0):
        """
        Args:
            mic_positions: (n_mics, 3) microphone position matrix [m]
            sound_speed: Speed of sound [m/s]
        """
        self.mic_positions = mic_positions
        self.sound_speed = sound_speed
        self.n_mics = mic_positions.shape[0]

    def calculate(self, tdoa_measurements: np.ndarray, reference_mic: int = 0) -> DOAResult:
        """Compute DOA direction from TDOA measurements."""
        direction, residual = calculate_doa(
            self.mic_positions, tdoa_measurements, self.sound_speed, reference_mic
        )
        azimuth, elevation = cartesian_to_spherical(direction)
        return DOAResult(direction, residual, azimuth, elevation)

    def to_global_frame(
        self,
        doa_local: np.ndarray,
        uav_position: np.ndarray,
        uav_yaw: float = 0.0
    ) -> np.ndarray:
        """Convert local DOA to global coordinate frame via yaw rotation."""
        return local_to_global_doa(doa_local, uav_position, uav_yaw)


def calculate_doa(
    mic_positions: np.ndarray,
    tdoa_measurements: np.ndarray,
    sound_speed: float = 343.0,
    reference_mic: int = 0
) -> Tuple[np.ndarray, float]:
    """
    Spherical least-squares DOA estimation.

    Model: tau_i = (r_i - r_0) . u_hat / c
    Solve: A . u_hat = b, then normalize to unit vector.

    Returns:
        (doa_vector, residual_error) — unit direction vector and fit residual [s]
    """
    r0 = mic_positions[reference_mic]
    other_mics = [i for i in range(len(mic_positions)) if i != reference_mic]
    A = (mic_positions[other_mics] - r0) / sound_speed
    b = tdoa_measurements

    u_unconstrained, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    u_norm = u_unconstrained / np.linalg.norm(u_unconstrained)

    predicted_tdoa = A @ u_norm
    residual_error = np.sqrt(np.mean((b - predicted_tdoa) ** 2))

    return u_norm, residual_error


def compute_doa_weight(peak_values: np.ndarray) -> float:
    """DOA observation weight from GCC-PHAT peak values.

    w_k = (1/M) * sum_{m=1}^{M} gamma_m^{(k)}

    where gamma_m is the GCC-PHAT normalized cross-correlation peak
    for the m-th microphone pair. Higher peak → stronger signal →
    more reliable DOA → higher weight in triangulation.
    """
    return float(np.mean(peak_values))


def local_to_global_doa(
    doa_local: np.ndarray,
    uav_position: np.ndarray,
    uav_yaw: float = 0.0
) -> np.ndarray:
    """Rotate local DOA to global frame using Z-axis yaw rotation."""
    cos_yaw = np.cos(uav_yaw)
    sin_yaw = np.sin(uav_yaw)

    R_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])

    return R_z @ doa_local


def cartesian_to_spherical(direction: np.ndarray) -> Tuple[float, float]:
    """
    Convert direction vector to spherical coordinates.

    Returns:
        (azimuth, elevation) in degrees.
        azimuth: [-180, 180], 0 = North (+Y), 90 = East (+X)
        elevation: [-90, 90], 0 = horizontal, -90 = straight down
    """
    x, y, z = direction
    azimuth = np.degrees(np.arctan2(x, y))
    r_xy = np.sqrt(x ** 2 + y ** 2)
    elevation = np.degrees(np.arctan2(-z, r_xy))
    return azimuth, elevation


def spherical_to_cartesian(azimuth: float, elevation: float) -> np.ndarray:
    """Convert spherical (azimuth, elevation) in degrees to unit direction vector."""
    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)
    x = np.cos(el_rad) * np.sin(az_rad)
    y = np.cos(el_rad) * np.cos(az_rad)
    z = -np.sin(el_rad)
    return np.array([x, y, z])


def get_9mic_array_positions(
    center: np.ndarray = np.array([0, 0, 0]),
    radius: float = 0.1
) -> np.ndarray:
    """
    Generate 9-mic UCA positions (M topology).

    Returns:
        (9, 3) array — index 0 = center m0, indices 1-8 = ring
    """
    positions = np.zeros((9, 3))
    positions[0] = center

    for i in range(8):
        angle = 2 * np.pi * i / 8
        positions[i + 1] = center + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0
        ])

    return positions
