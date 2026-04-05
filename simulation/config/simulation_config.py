"""
Simulation configuration module.

Unified dataclass for all simulation parameters including environment,
path planning, anomaly detection, TDOA/DOA, localization, and visualization.
"""

from dataclasses import dataclass
from typing import Tuple
from enum import Enum

from SpecMae.simulation.core.propagation_model import DEFAULT_SOURCE_SPL


class EnvironmentType(Enum):
    """Environment type."""
    DESERT = "desert"  # No reverberation (RT60=0s)
    FOREST = "forest"  # Strong reverberation (RT60=0.6s)


@dataclass
class SimulationConfig:
    """Unified simulation configuration parameters."""

    # === Environment ===
    area_bounds: Tuple[float, float, float, float] = (0, 35, 0, 35)  # [x_min, x_max, y_min, y_max]
    flight_height: float = 5.0        # [m] (optimal for desert)
    env_type: EnvironmentType = EnvironmentType.FOREST

    # === Path planning ===
    coverage_radius: float = 15.0     # [m] path-planning search radius (NOT acoustic cutoff)
    overlap_ratio: float = 0.2
    velocity: float = 10.0            # [m/s] (Paper Table I: 5-15 m/s; Section VI.B: 10 m/s)
    sampling_interval: float = 0.1    # [s]

    # === Anomaly detection (SpecMAE pure reconstruction) ===
    anomaly_threshold: float = 0.5    # Reconstruction threshold (set via Youden's J)
    detection_window: float = 1.0     # [s]
    model_checkpoint: str = ""        # Path to SpecMAE checkpoint
    model_size: str = "base"          # Model variant: tiny, small, base, large

    # Primary: reconstruction MSE detection (pure reconstruction, PatchKNN removed)
    recon_mask_ratio: float = 0.75
    recon_n_passes: int = 20
    recon_score_mode: str = "top_k"
    recon_top_k_ratio: float = 0.15
    recon_use_multiscale: bool = False

    # === TDOA/DOA parameters ===
    tdoa_window_samples: int = 2048
    tdoa_confidence_threshold: float = 1.5
    sound_speed: float = 343.0        # [m/s]
    max_delay_samples: int = 100

    # === Localization ===
    min_detection_points: int = 3
    max_detection_points: int = 10
    max_residual: float = 10.0        # [m]

    # === Microphone ===
    # Mic = UAV height (no suspension cable — whiteboard requirement)

    # === Terrain-specific flight heights ===
    desert_flight_height: float = 5.0    # [m] (optimal from height sweep)
    forest_flight_height: float = 15.0   # [m] (optimal from height sweep)

    # === Acoustic simulation ===
    sample_rate: int = 48000          # [Hz]
    room_dimensions: Tuple[float, float, float] = (35, 35, 12)
    rt60_desert: float = 0.0
    rt60_forest: float = 0.6
    use_full_acoustic_sim: bool = True

    # === Dynamic SNR propagation model (whiteboard) ===
    propagation_center_freq: float = 2000.0  # [Hz] center freq for absorption calc
    source_spl: float = DEFAULT_SOURCE_SPL  # [dB SPL] human voice at 1m (120, whiteboard)
    ambient_noise_spl_desert: float = 25.0  # [dB SPL] desert ambient
    ambient_noise_spl_forest: float = 35.0  # [dB SPL] forest ambient
    propeller_noise_spl: float = 75.0  # [dB SPL] UAV drone noise at mic

    # === Visualization ===
    enable_realtime_viz: bool = False
    save_animation: bool = False
    output_dir: str = "./simulation_results"
    figure_dpi: int = 300

    # === Retroactive DMA buffer (Paper Section II.C) ===
    retroactive_buffer_dt: float = 0.5     # [s]
    hardware_wake_latency: float = 0.3     # [s]
    enable_retroactive_buffer: bool = True

    # === Localization algorithm selection (Paper Section IV.C) ===
    use_ukf: bool = True
    use_ekf_baseline: bool = True
    use_ls_baseline: bool = True
    ukf_alpha: float = 1e-3
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    energy_eta: float = 2.0
    energy_gamma: float = 3.0

    # === Target signal type (Paper Section VI.B) ===
    target_signal_type: str = "male"  # "male", "child", "whistle"

    # === Misc ===
    random_seed: int = 42
    verbose: bool = True

    def get_terrain_flight_height(self) -> float:
        """Get the flight height appropriate for the current environment."""
        if self.env_type == EnvironmentType.DESERT:
            return self.desert_flight_height
        return self.forest_flight_height

    def get_rt60(self) -> float:
        if self.env_type == EnvironmentType.DESERT:
            return self.rt60_desert
        return self.rt60_forest

    def get_max_order(self) -> int:
        if self.env_type == EnvironmentType.DESERT:
            return 0
        return 15

    def validate(self) -> bool:
        errors = []
        x_min, x_max, y_min, y_max = self.area_bounds
        if x_min >= x_max or y_min >= y_max:
            errors.append("Invalid area bounds")
        if self.flight_height <= 0:
            errors.append("Flight height must be positive")
        if self.coverage_radius <= 0:
            errors.append("Coverage radius must be positive")
        if self.velocity <= 0:
            errors.append("Velocity must be positive")
        if self.sampling_interval <= 0:
            errors.append("Sampling interval must be positive")
        if self.min_detection_points < 2:
            errors.append("min_detection_points must be >= 2")

        if errors:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        return True


# ── Predefined configurations ──────────────────────────────────────────────

def get_default_config() -> SimulationConfig:
    return SimulationConfig()


def get_quick_test_config() -> SimulationConfig:
    """Small area, fast flight, simplified acoustics."""
    return SimulationConfig(
        area_bounds=(0, 20, 0, 20),
        velocity=10.0,
        sampling_interval=0.2,
        min_detection_points=2,
        use_full_acoustic_sim=False,
        verbose=True
    )


def get_high_precision_config() -> SimulationConfig:
    """Slow flight, dense sampling, full acoustic simulation."""
    return SimulationConfig(
        velocity=5.0,
        sampling_interval=0.05,
        overlap_ratio=0.3,
        min_detection_points=5,
        max_detection_points=20,
        tdoa_confidence_threshold=2.0,
        use_full_acoustic_sim=True
    )
