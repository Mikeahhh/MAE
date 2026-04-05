"""
Acoustic simulation module.

Implements high-fidelity 3D acoustic propagation using pyroomacoustics
for desert (RT60=0s) and forest (RT60=0.6s) environments.
"""

import numpy as np
import pyroomacoustics as pra
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AcousticEnvironment:
    """Acoustic environment configuration."""
    room_dimensions: Tuple[float, float, float] = (35, 35, 12)  # [m]
    env_type: str = "forest"  # "desert" or "forest"
    sample_rate: int = 48000  # [Hz]

    # Environment parameters
    rt60_desert: float = 0.0   # Desert reverberation time [s]
    rt60_forest: float = 0.6   # Forest reverberation time [s]

    # Source parameters
    source_frequency: float = 1000.0  # Fundamental frequency [Hz]
    source_duration: float = 0.5      # Signal duration [s]
    source_amplitude: float = 10.0    # Source amplitude (120 dB SPL at 1m, human distress voice)

    # Mic array parameters (Paper Table I: 0.12 m)
    array_radius: float = 0.12  # [m]


class AcousticSimulator:
    """Full 3D acoustic simulator using pyroomacoustics."""

    def __init__(self, env_config: AcousticEnvironment):
        self.config = env_config
        self.room = None
        self.source_position = None

    def setup_room(self, env_type: str = "forest"):
        """Set up the acoustic room for the given environment type."""
        room_dim = self.config.room_dimensions
        fs = self.config.sample_rate

        if env_type == "desert":
            e_absorption = 0.99
            max_order = 0
        elif env_type == "forest":
            rt60 = self.config.rt60_forest
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")

        self.room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            ray_tracing=True,
            air_absorption=True
        )

    def add_microphone_array(self, uav_position: np.ndarray):
        """Add 9-mic UCA array at the UAV position.

        Microphone height == UAV flight height (no suspension cable).
        This is enforced by using uav_position[2] directly as mic Z.
        """
        if self.room is None:
            raise RuntimeError("Room not setup. Call setup_room() first.")
        assert uav_position.shape == (3,), f"UAV position must be (3,), got {uav_position.shape}"

        array_radius = self.config.array_radius

        # 8 ring microphones (UCA)
        R_ring = pra.circular_2D_array(
            center=uav_position[:2],
            M=8,
            phi0=0,
            radius=array_radius
        )

        # Center microphone m0
        R_center = np.array([[uav_position[0]], [uav_position[1]]])

        # Combine 2D coordinates (center=index 0, ring=1-8)
        R_2d = np.hstack((R_center, R_ring))

        # Add Z axis
        R_3d = np.vstack((R_2d, np.full(9, uav_position[2])))

        self.room.add_microphone_array(pra.MicrophoneArray(R_3d, self.room.fs))

    def add_source(self, source_position: np.ndarray, signal: Optional[np.ndarray] = None):
        """Add a sound source at the given position."""
        if self.room is None:
            raise RuntimeError("Room not setup. Call setup_room() first.")

        self.source_position = source_position

        if signal is None:
            signal = self._generate_human_voice_signal()

        self.room.add_source(source_position.tolist(), signal=signal)

    def _generate_human_voice_signal(self) -> np.ndarray:
        """Generate a synthetic human voice signal with harmonics and envelope."""
        fs = self.config.sample_rate
        duration = self.config.source_duration
        n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs

        f0 = 120.0  # Male voice fundamental [Hz]

        signal = np.zeros(n_samples)
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
        amplitudes = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]

        for h, amp in zip(harmonics, amplitudes):
            signal += amp * np.sin(2 * np.pi * f0 * h * t)

        # Amplitude envelope (exponential decay + 5 Hz modulation)
        envelope = np.exp(-3 * t / duration)
        envelope = envelope * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))

        signal = signal * envelope
        signal = signal / np.max(np.abs(signal)) * self.config.source_amplitude

        return signal

    def simulate(self) -> np.ndarray:
        """Run acoustic simulation. Returns (9, n_samples) mic signal matrix."""
        if self.room is None:
            raise RuntimeError("Room not setup. Call setup_room() first.")

        self.room.simulate()
        mic_signals = self.room.mic_array.signals

        max_val = np.max(np.abs(mic_signals))
        if max_val > 1.0:
            mic_signals = mic_signals / max_val

        return mic_signals

    def simulate_at_position(
        self,
        uav_position: np.ndarray,
        source_position: np.ndarray,
        env_type: str = "forest"
    ) -> np.ndarray:
        """
        Run full acoustic simulation at specified positions.

        Args:
            uav_position: (3,) UAV position [m]
            source_position: (3,) source position [m]
            env_type: "desert" or "forest"

        Returns:
            (9, n_samples) microphone signal matrix
        """
        room_dim = self.config.room_dimensions

        # Clip positions to room bounds (0.5m margin)
        uav_pos_clipped = np.clip(
            uav_position,
            [0.5, 0.5, 0.5],
            [room_dim[0] - 0.5, room_dim[1] - 0.5, room_dim[2] - 0.5]
        )
        source_pos_clipped = np.clip(
            source_position,
            [0.5, 0.5, 0.1],
            [room_dim[0] - 0.5, room_dim[1] - 0.5, room_dim[2] - 0.5]
        )

        self.setup_room(env_type)
        self.add_microphone_array(uav_pos_clipped)
        self.add_source(source_pos_clipped)
        mic_signals = self.simulate()

        return mic_signals
