"""
Hardware configuration module.

Microphone array geometry, UAV specifications, and power budget
parameters matching Paper Table I.
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class MicArrayConfig:
    """Microphone array configuration (9-mic UCA)."""

    n_mics: int = 9
    array_radius: float = 0.12  # [m] (Paper Table I: 0.12 m)
    center_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def get_mic_positions(self) -> np.ndarray:
        """
        Generate (9, 3) microphone position matrix.

        Index 0: center m0, indices 1-8: ring (uniform circular array).
        """
        positions = np.zeros((self.n_mics, 3))
        positions[0] = np.array(self.center_position)

        for i in range(8):
            angle = 2 * np.pi * i / 8
            positions[i + 1] = np.array(self.center_position) + np.array([
                self.array_radius * np.cos(angle),
                self.array_radius * np.sin(angle),
                0
            ])

        return positions

    def get_max_tdoa(self, sound_speed: float = 343.0) -> float:
        """Maximum possible TDOA for this array geometry [s]."""
        return 2 * self.array_radius / sound_speed


@dataclass
class HardwareConfig:
    """UAV hardware configuration."""

    # Microphone array
    mic_array: MicArrayConfig = None

    # Sampling
    sample_rate: int = 48000
    bit_depth: int = 16
    buffer_size: int = 2048

    # UAV
    uav_mass: float = 1.5        # [kg]
    max_velocity: float = 10.0   # [m/s]
    max_acceleration: float = 5.0  # [m/s^2]

    # Power budget
    power_dsp: float = 0.015     # [W]
    power_npu: float = 15.0      # [W]
    power_rotor: float = 35.0    # [W]
    battery_capacity: float = 30.0  # [Wh]

    # Acoustics
    sound_speed: float = 343.0   # [m/s]
    air_temperature: float = 20.0  # [C]
    air_humidity: float = 50.0   # [%]

    def __post_init__(self):
        if self.mic_array is None:
            self.mic_array = MicArrayConfig()

    def get_flight_endurance(self) -> float:
        """Flight endurance in minutes."""
        total_power = self.power_rotor + self.power_dsp
        battery_joules = self.battery_capacity * 3600
        return battery_joules / total_power / 60

    def get_sound_speed_corrected(self) -> float:
        """Temperature-corrected speed of sound: c = 331.3 + 0.606*T [m/s]."""
        return 331.3 + 0.606 * self.air_temperature
