"""
Physical acoustic propagation model for dynamic SNR simulation.

Computes path loss between a sound source and a UAV-mounted microphone
using a log-distance model with atmospheric absorption:

    L(d) = L_0 + 10 * n * log10(d / d_0) + alpha * d

Whiteboard SNR formula (SPAWC):
    SNR = P / (N_uav + N_env)   [linear domain]
    where P = source power after distance attenuation
          N_uav = drone propeller noise (75 dB)
          N_env = ambient noise (desert 25 dB / forest 35 dB)
    Source: 120 dB SPL @ 1m (human voice)
    Mic height = UAV flight height (no suspension cable)
    Horizontal offset = 5 m

Components:
    - Spherical spreading loss (n = 2.0 free-space)
    - Atmospheric absorption (frequency-dependent, ISO 9613-1)
    - Ground reflection (terrain-dependent excess attenuation)
    - Forest: n = 2.5 + canopy attenuation (physically necessary)

References:
    - ISO 9613-1:1993 — Atmospheric sound absorption
    - ISO 9613-2:1996 — Ground attenuation
    - Kinsler et al., "Fundamentals of Acoustics", 4th ed.
    - Embleton (1996) — Outdoor sound propagation
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  Whiteboard physical constants (SPAWC paper)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SOURCE_SPL = 120.0        # Human voice at 1 m [dB SPL]
DEFAULT_DRONE_NOISE_SPL = 75.0    # UAV propeller noise at mic [dB SPL]
DEFAULT_HORIZONTAL_OFFSET = 5.0   # Horizontal offset between UAV and person [m]


@dataclass
class TerrainProfile:
    """Terrain-specific acoustic parameters."""
    name: str
    path_loss_exponent: float    # n — 2.0 = free field, >2 = excess attenuation
    ground_factor: float         # dB additional ground attenuation per 100m
    humidity_pct: float          # relative humidity [%]
    temperature_c: float         # ambient temperature [°C]
    canopy_attenuation: float    # dB/m through canopy (0 for open terrain)
    flight_height_range: Tuple[float, float]  # (min, max) safe flight height [m]
    ambient_spl: float           # environmental ambient noise [dB SPL]


# Predefined terrain profiles
TERRAIN_DESERT = TerrainProfile(
    name="desert",
    path_loss_exponent=2.0,
    ground_factor=0.0,
    humidity_pct=15.0,
    temperature_c=35.0,
    canopy_attenuation=0.0,
    flight_height_range=(5.0, 20.0),
    ambient_spl=25.0,
)

TERRAIN_FOREST = TerrainProfile(
    name="forest",
    path_loss_exponent=2.5,
    ground_factor=3.0,
    humidity_pct=70.0,
    temperature_c=22.0,
    canopy_attenuation=0.02,    # ~0.02 dB/m through moderate canopy
    flight_height_range=(15.0, 50.0),
    ambient_spl=35.0,
)

TERRAIN_REGISTRY = {
    "desert": TERRAIN_DESERT,
    "forest": TERRAIN_FOREST,
}


class PropagationModel:
    """
    Log-distance propagation model with atmospheric absorption.

    L(d) = L_ref + 10 * n * log10(d / d_ref) + alpha * d + L_ground + L_canopy

    where:
        L_ref   = reference loss at d_ref [dB]
        n       = path-loss exponent (terrain-dependent)
        alpha   = atmospheric absorption coefficient [dB/m]
        L_ground = terrain-specific ground attenuation [dB]
        L_canopy = canopy penetration loss [dB]

    Whiteboard SNR:
        SNR = (source_spl - TL) - 10*log10(10^(drone/10) + 10^(ambient/10))
    Mic height = UAV flight height (no suspension cable).
    """

    def __init__(
        self,
        terrain: TerrainProfile | str = "desert",
        center_freq_hz: float = 2000.0,
        d_ref: float = 1.0,
        l_ref: float = 0.0,
    ):
        if isinstance(terrain, str):
            terrain = TERRAIN_REGISTRY[terrain]

        self.terrain = terrain
        self.center_freq = center_freq_hz
        self.d_ref = d_ref
        self.l_ref = l_ref

        # Compute atmospheric absorption coefficient
        self.alpha = self._atmospheric_absorption(
            freq_hz=center_freq_hz,
            temperature_c=terrain.temperature_c,
            humidity_pct=terrain.humidity_pct,
        )

    @staticmethod
    def _atmospheric_absorption(
        freq_hz: float,
        temperature_c: float = 20.0,
        humidity_pct: float = 50.0,
        pressure_kpa: float = 101.325,
    ) -> float:
        """
        Atmospheric absorption coefficient [dB/m].

        Simplified ISO 9613-1 model for outdoor sound propagation.
        """
        T = temperature_c + 273.15   # Kelvin
        T_ref = 293.15
        T_01 = 273.16               # triple point
        p = pressure_kpa
        p_ref = 101.325

        # Molar concentration of water vapor
        C = -6.8346 * (T_01 / T) ** 1.261 + 4.6151
        h = humidity_pct * (10.0 ** C) * (p_ref / p)

        # Relaxation frequencies (O2 and N2)
        f_rO = (p / p_ref) * (
            24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
        )
        f_rN = (p / p_ref) * (T / T_ref) ** (-0.5) * (
            9.0 + 280.0 * h * math.exp(-4.170 * ((T / T_ref) ** (-1.0/3.0) - 1.0))
        )

        f = freq_hz
        # Absorption in dB/m (ISO 9613-1 formula, simplified)
        alpha = 8.686 * f * f * (
            1.84e-11 * (p_ref / p) * (T / T_ref) ** 0.5
            + (T / T_ref) ** (-2.5) * (
                0.01275 * math.exp(-2239.1 / T) / (f_rO + f * f / f_rO)
                + 0.1068  * math.exp(-3352.0 / T) / (f_rN + f * f / f_rN)
            )
        )
        return alpha

    def path_loss(
        self,
        distance: float,
        include_ground: bool = True,
        canopy_thickness: float = 0.0,
    ) -> float:
        """
        Total path loss [dB] at given source-receiver distance.

        Args:
            distance: source-receiver distance [m]
            include_ground: include terrain-specific ground attenuation
            canopy_thickness: thickness of canopy between source and mic [m]

        Returns:
            Total path loss in dB (positive = attenuation)
        """
        d = max(distance, self.d_ref)

        # Spherical spreading + excess path loss
        spreading = 10.0 * self.terrain.path_loss_exponent * math.log10(d / self.d_ref)

        # Atmospheric absorption
        absorption = self.alpha * d

        # Ground attenuation
        ground = 0.0
        if include_ground:
            ground = self.terrain.ground_factor * (d / 100.0)

        # Canopy penetration loss
        canopy = self.terrain.canopy_attenuation * canopy_thickness

        return self.l_ref + spreading + absorption + ground + canopy

    def snr_at_distance(
        self,
        distance: float,
        source_spl: float = DEFAULT_SOURCE_SPL,
        noise_spl: float = DEFAULT_DRONE_NOISE_SPL,
        flight_height: float = 10.0,
        source_height: float = 0.0,
        canopy_thickness: float = 0.0,
    ) -> float:
        """
        Compute SNR at the UAV microphone — whiteboard formula.

        SNR = P / (N_uav + N_env) in linear domain, returned in dB.

        Args:
            distance: horizontal distance between UAV and source [m]
            source_spl: source sound pressure level at 1m [dB SPL]
            noise_spl: UAV propeller noise at microphone [dB SPL]
            flight_height: UAV flight height [m]
            source_height: height of sound source [m] (0 = ground level)
            canopy_thickness: canopy penetration thickness [m]

        Returns:
            SNR in dB at the microphone
        """
        # Mic = UAV height (no suspension cable)
        mic_height = flight_height
        dz = mic_height - source_height
        d_3d = math.sqrt(distance ** 2 + dz ** 2)

        # Signal: source attenuated by path loss
        tl = self.path_loss(d_3d, canopy_thickness=canopy_thickness)
        signal_db = source_spl - tl

        # Noise: N_uav + N_env in linear domain
        p_drone = 10.0 ** (noise_spl / 10.0)
        p_ambient = 10.0 ** (self.terrain.ambient_spl / 10.0)
        nl_db = 10.0 * math.log10(p_drone + p_ambient)

        return signal_db - nl_db

    def attenuation_linear(
        self,
        distance: float,
        canopy_thickness: float = 0.0,
    ) -> float:
        """
        Waveform-domain attenuation factor: 10^(-TL/20) ≈ 1/d.

        Use this to scale voice waveforms for physics-based audio mixing.
        """
        tl = self.path_loss(distance, canopy_thickness=canopy_thickness)
        return 10.0 ** (-tl / 20.0)

    def snr_along_path(
        self,
        distances: np.ndarray,
        source_spl: float = DEFAULT_SOURCE_SPL,
        noise_spl: float = DEFAULT_DRONE_NOISE_SPL,
        flight_height: float = 10.0,
        source_height: float = 0.0,
        canopy_thickness: float = 0.0,
    ) -> np.ndarray:
        """
        Compute SNR array along a set of horizontal distances.

        This produces the U-shaped SNR curve as the UAV approaches,
        passes over, and moves away from the source.

        Returns:
            snr_array: (N,) SNR values in dB
        """
        return np.array([
            self.snr_at_distance(
                float(d), source_spl, noise_spl, flight_height,
                source_height, canopy_thickness,
            )
            for d in distances
        ])


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def get_propagation_model(
    terrain: str = "desert",
    center_freq_hz: float = 2000.0,
) -> PropagationModel:
    """Create a propagation model with standard parameters."""
    return PropagationModel(
        terrain=terrain,
        center_freq_hz=center_freq_hz,
    )


def compute_detection_range(
    terrain: str = "desert",
    source_spl: float = DEFAULT_SOURCE_SPL,
    noise_spl: float = DEFAULT_DRONE_NOISE_SPL,
    min_snr_db: float = -5.0,
    flight_height: float = 10.0,
) -> float:
    """
    Estimate maximum horizontal detection range [m].

    Finds the distance at which SNR drops below min_snr_db.
    """
    model = get_propagation_model(terrain)
    for d in np.arange(1, 500, 1):
        snr = model.snr_at_distance(
            float(d), source_spl, noise_spl, flight_height,
        )
        if snr < min_snr_db:
            return float(d)
    return 500.0
