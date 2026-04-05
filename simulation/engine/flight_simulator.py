"""
Flight simulator — Core integrated simulation engine.

Orchestrates the sentinel-responder UAV search-and-rescue pipeline:
  1. CCPP path planning (Boustrophedon sweep)
  2. Acoustic simulation (pyroomacoustics + physical propagation model)
  3. Anomaly detection (SpecMAE pure reconstruction via DetectorBridge)
  4. TDOA estimation (GCC-PHAT)
  5. DOA calculation (spherical least-squares)
  6. Multi-point triangulation (ray-crossing LS)
  7. Mode tracking (sentinel / responder)

Refactored for SpecMae: pure reconstruction-based SpecMAE detection
via DetectorBridge, physics-based SNR from PropagationModel.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..core.path_planner import PathPlanner, PathPoint
from ..core.tdoa_estimator import TDOAEstimator, estimate_tdoa_with_confidence
from ..core.doa_calculator import DOACalculator, get_9mic_array_positions
from ..core.triangulation import Triangulator, triangulate_2d, calculate_geometric_dilution_of_precision
from ..core.ring_buffer import RetroactiveRingBuffer
from ..config.simulation_config import SimulationConfig
from ..config.hardware_config import HardwareConfig
from .data_logger import DataLogger, DetectionEvent


@dataclass
class SimulationResult:
    """Complete simulation result."""
    estimated_position: Optional[np.ndarray]
    true_position: np.ndarray
    localization_error: float         # [m]
    num_detections: int
    mission_time: float               # [s]
    path_length: float                # [m]
    detection_rate: float
    false_alarm_rate: float
    doa_errors_deg: List[float] = field(default_factory=list)
    mode_switches: int = 0            # sentinel <-> responder transitions
    detection_positions: List[np.ndarray] = field(default_factory=list)
    doa_vectors: List[np.ndarray] = field(default_factory=list)
    trajectory: Optional[np.ndarray] = None
    mode_history: List[dict] = field(default_factory=list)


class FlightSimulator:
    """
    Integrated flight simulation engine.

    Uses dependency injection: accepts a DetectorBridge for anomaly detection,
    keeping the simulation decoupled from the specific model architecture.
    """

    def __init__(
        self,
        sim_config: Optional[SimulationConfig] = None,
        hw_config: Optional[HardwareConfig] = None,
        detector_bridge=None,
    ):
        """
        Args:
            sim_config: Simulation configuration
            hw_config: Hardware configuration
            detector_bridge: DetectorBridge instance (SpecMAE reconstruction)
                             If None, uses heuristic fallback detection.
        """
        self.sim_config = sim_config if sim_config is not None else SimulationConfig()
        self.hw_config = hw_config if hw_config is not None else HardwareConfig()
        self.detector = detector_bridge

        if not self.sim_config.validate():
            raise ValueError("Invalid simulation configuration")

        # Core components
        self.path_planner = None
        self.tdoa_estimator = TDOAEstimator(
            fs=self.sim_config.sample_rate,
            window_samples=self.sim_config.tdoa_window_samples,
            confidence_threshold=self.sim_config.tdoa_confidence_threshold
        )
        self.doa_calculator = DOACalculator(
            mic_positions=self.hw_config.mic_array.get_mic_positions(),
            sound_speed=self.hw_config.get_sound_speed_corrected()
        )
        self.triangulator = Triangulator(
            min_rays=self.sim_config.min_detection_points,
            max_residual=self.sim_config.max_residual
        )
        self.data_logger = DataLogger(self.sim_config.output_dir)

        # Ring buffer for retroactive capture
        self.ring_buffer = None
        if self.sim_config.enable_retroactive_buffer:
            self.ring_buffer = RetroactiveRingBuffer(
                n_mics=self.hw_config.mic_array.n_mics,
                fs=self.sim_config.sample_rate,
                delta_t=self.sim_config.retroactive_buffer_dt,
                delta_wake=self.sim_config.hardware_wake_latency,
            )

        # Simulation state
        self.target_position = None
        self.current_time = 0.0
        self.current_mode = "sentinel"   # "sentinel" or "responder"
        self.mode_switches = 0

        # Audio streams
        self.background_noise_stream = None
        self.target_signal_stream = None

    def setup_environment(self, target_position: np.ndarray):
        """
        Set up the simulation environment.

        Args:
            target_position: (3,) ground truth source position [m]
        """
        self.target_position = target_position

        self.path_planner = PathPlanner(
            area_bounds=self.sim_config.area_bounds,
            flight_height=self.sim_config.flight_height,
            coverage_radius=self.sim_config.coverage_radius,
            overlap_ratio=self.sim_config.overlap_ratio,
            velocity=self.sim_config.velocity
        )

        self._initialize_audio_streams()

        if self.sim_config.verbose:
            print(f"  Environment setup complete")
            print(f"    Target position: {target_position}")
            print(f"    Search area: {self.sim_config.area_bounds}")
            print(f"    Flight height: {self.sim_config.flight_height} m")
            print(f"    Detector: {'SpecMAE recon' if self.detector else 'heuristic fallback'}")

    def _initialize_audio_streams(self):
        """Initialize continuous background noise and target signal audio streams."""
        fs = self.sim_config.sample_rate

        # Background noise: UAV rotor (BPF=120 Hz, Paper Section VI.A)
        base_duration = 10.0
        base_samples = int(base_duration * fs)
        t = np.arange(base_samples) / fs

        rotor_freq = 120.0
        background = np.random.randn(base_samples) * 0.05
        background += 0.03 * np.sin(2 * np.pi * rotor_freq * t)
        background += 0.02 * np.sin(2 * np.pi * rotor_freq * 2 * t)
        self.background_noise_stream = background

        # Target signal: human voice (3s period, looping)
        help_duration = 3.0
        help_samples = int(help_duration * fs)
        t_help = np.arange(help_samples) / fs

        f0 = 120.0
        help_signal = np.zeros(help_samples)
        harmonics = [1, 2, 3, 4, 5, 6, 7, 8]
        amplitudes = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]
        for h, amp in zip(harmonics, amplitudes):
            help_signal += amp * np.sin(2 * np.pi * f0 * h * t_help)

        envelope = np.exp(-2 * t_help / help_duration)
        envelope = envelope * (1 + 0.5 * np.sin(2 * np.pi * 3 * t_help))
        help_signal = help_signal * envelope
        help_signal = help_signal / np.max(np.abs(help_signal))
        self.target_signal_stream = help_signal

    def run_mission(self) -> SimulationResult:
        """
        Execute the full search-and-rescue mission.

        Returns:
            SimulationResult with all metrics and trajectory data.
        """
        if self.target_position is None:
            raise ValueError("Call setup_environment() first")

        if self.sim_config.verbose:
            print(f"\n{'=' * 60}")
            print(f"  Starting SAR mission simulation")
            print(f"{'=' * 60}")

        self.data_logger.start_mission()
        self.current_mode = "sentinel"
        self.mode_switches = 0
        self.data_logger.log_mode_change(0.0, "sentinel")

        # 1. Generate path
        path = self.path_planner.generate_sampling_points(
            self.sim_config.sampling_interval
        )

        if self.sim_config.verbose:
            print(f"\n  Path planning:")
            print(f"    Waypoints: {len(path)}")
            print(f"    Path length: {self.path_planner.get_path_length():.1f} m")
            print(f"    Est. time: {self.path_planner.get_mission_time():.1f} s")

        # 2. Fly path, detect anomalies
        detection_count = 0
        first_detection_time = None

        for i, waypoint in enumerate(path):
            self.current_time = waypoint.timestamp
            position = waypoint.to_array()

            # Record trajectory with current mode
            self.data_logger.log_trajectory_point(waypoint.timestamp, position, self.current_mode)

            # Anomaly detection
            is_anomaly, score, method = self._detect_anomaly(position)

            if is_anomaly:
                detection_count += 1

                # Switch to responder mode on first detection
                if self.current_mode == "sentinel":
                    self.current_mode = "responder"
                    self.mode_switches += 1
                    self.data_logger.log_mode_change(waypoint.timestamp, "responder")
                    first_detection_time = waypoint.timestamp
                    if self.sim_config.verbose:
                        print(f"\n  [MODE SWITCH] sentinel -> responder at t={waypoint.timestamp:.1f}s")

                # Perform TDOA/DOA localization
                doa_result = self._perform_localization(position)

                # Compute DOA error against ground truth
                doa_error_deg = None
                if doa_result and doa_result['doa_global'] is not None:
                    doa_error_deg = self._compute_doa_error(
                        position, doa_result['doa_global']
                    )

                event = DetectionEvent(
                    timestamp=waypoint.timestamp,
                    position=position.copy(),
                    anomaly_score=score,
                    detection_method=method,
                    mode=self.current_mode,
                    doa_global=doa_result['doa_global'] if doa_result else None,
                    confidence=doa_result['confidence'] if doa_result else None,
                    residual_error=doa_result['residual'] if doa_result else None,
                    doa_error_deg=doa_error_deg,
                )
                self.data_logger.log_detection(event)

                if self.sim_config.verbose and detection_count <= 5:
                    print(f"\n  [DETECT #{detection_count}] t={waypoint.timestamp:.1f}s "
                          f"pos=({position[0]:.1f},{position[1]:.1f},{position[2]:.1f}) "
                          f"score={score:.4f} method={method}")
                    if doa_error_deg is not None:
                        print(f"    DOA error: {doa_error_deg:.2f} deg")

                if detection_count >= self.sim_config.max_detection_points:
                    if self.sim_config.verbose:
                        print(f"\n  Max detections reached ({self.sim_config.max_detection_points})")
                    break

            # Progress
            if self.sim_config.verbose and (i + 1) % 200 == 0:
                print(f"  Progress: {(i + 1) / len(path) * 100:.0f}% "
                      f"({i + 1}/{len(path)}) mode={self.current_mode}")

        self.data_logger.end_mission()

        # 3. Triangulate source position
        estimated_position = None
        if detection_count >= self.sim_config.min_detection_points:
            estimated_position = self._triangulate_position()

        # 4. Compute result
        result = self._compute_result(estimated_position, detection_count, path)

        # 5. Save data
        self.data_logger.save_all(estimated_position, self.target_position)

        if self.sim_config.verbose:
            print(f"\n{'=' * 60}")
            print(f"  Mission complete")
            print(f"{'=' * 60}")
            self._print_result(result)

        return result

    def _detect_anomaly(self, position: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect anomaly using DetectorBridge or heuristic fallback.

        Returns:
            (is_anomaly, score, method)
        """
        if self.detector is not None:
            return self._detect_anomaly_with_model(position)
        else:
            is_anomaly, score = self._detect_anomaly_heuristic(position)
            return is_anomaly, score, "heuristic"

    def _detect_anomaly_with_model(self, position: np.ndarray) -> Tuple[bool, float, str]:
        """Detect anomaly using SpecMAE pure reconstruction via DetectorBridge.

        Uses a two-stage pipeline matching the real system:
          Stage 1: Signal energy gate (hardware DMA wake-up trigger, Paper II.C)
                   — skip model inference if no target energy is present
          Stage 2: SpecMAE reconstruction-based anomaly detection
        """
        try:
            mic_signals = self._simulate_acoustic_signals(position)

            # Write to ring buffer (continuous DMA)
            if self.ring_buffer is not None:
                self.ring_buffer.write(mic_signals.astype(np.float32))

            # Stage 1: Signal energy gate (simulates DMA wake-up threshold)
            # Uses cached baseline noise energy to avoid random variation
            # between independent noise generations.
            if not hasattr(self, '_baseline_noise_energy'):
                # Compute baseline once from deterministic seed
                rng_state = np.random.get_state()
                np.random.seed(12345)
                self._baseline_noise_energy = np.mean(
                    self._generate_background_noise()[0] ** 2
                )
                np.random.set_state(rng_state)

            sig_energy = np.mean(mic_signals[0] ** 2)
            energy_ratio = sig_energy / (self._baseline_noise_energy + 1e-12)

            # Energy gate: signal must exceed baseline by 5% to justify
            # model inference. This simulates the DMA (Direct Memory Access)
            # hardware wake-up threshold that avoids powering on the SpecMAE
            # processor for pure-noise segments. The 5% margin accounts for
            # stochastic noise variation while avoiding false wake-ups.
            if energy_ratio < 1.05:
                return False, 0.0, "energy_gate"

            # Stage 2: Model inference on the target-containing signal
            audio_signal = mic_signals[0]
            is_anomaly, score, method = self.detector.detect(audio_signal)
            return is_anomaly, score, method

        except Exception as e:
            if self.sim_config.verbose:
                print(f"  [WARN] Model detection failed: {e}, using heuristic")
            is_anomaly, score = self._detect_anomaly_heuristic(position)
            return is_anomaly, score, "heuristic"

    def _detect_anomaly_heuristic(self, position: np.ndarray) -> Tuple[bool, float]:
        """Physics-based heuristic fallback detection using propagation model SNR."""
        from ..core.propagation_model import PropagationModel

        distance = np.linalg.norm(position[:2] - self.target_position[:2])
        env_str = (self.sim_config.env_type.value
                   if hasattr(self.sim_config.env_type, 'value')
                   else str(self.sim_config.env_type))

        model = PropagationModel(terrain=env_str)
        snr_db = model.snr_at_distance(
            distance=max(distance, 0.5),
            source_spl=self.sim_config.source_spl,
            noise_spl=self.sim_config.propeller_noise_spl,
            flight_height=self.sim_config.flight_height,
        )

        # Convert SNR to a pseudo anomaly score: higher SNR → higher score
        # Score above threshold means anomaly detected
        score = self.sim_config.anomaly_threshold * (1.0 + snr_db / 20.0)
        is_anomaly = score > self.sim_config.anomaly_threshold
        return is_anomaly, max(score, 0.0)

    def _perform_localization(self, uav_position: np.ndarray) -> Optional[dict]:
        """Perform TDOA/DOA localization from acoustic signals.

        Uses the pure pyroomacoustics source signal (without background noise)
        for TDOA estimation, simulating the real system's ANC (Active Noise
        Cancellation) that removes rotor harmonics before DOA processing.
        """
        try:
            # Get clean source signal for TDOA (post-ANC simulation)
            source_signals = self._get_clean_source_signals(uav_position)
            if source_signals is None:
                return None

            tdoa_measurements, peak_values = self._estimate_tdoa_from_signals(source_signals)
            doa_result = self.doa_calculator.calculate(tdoa_measurements)

            if doa_result.direction is None or np.any(np.isnan(doa_result.direction)):
                return None

            doa_global = self.doa_calculator.to_global_frame(
                doa_result.direction, uav_position, uav_yaw=0.0
            )

            if np.any(np.isnan(doa_global)):
                return None

            return {
                'doa_local': doa_result.direction,
                'doa_global': doa_global,
                'confidence': doa_result.residual_error if doa_result.residual_error > 0 else 1.5,
                'residual': doa_result.residual_error,
            }
        except Exception:
            return None

    def _get_clean_source_signals(self, uav_position: np.ndarray) -> Optional[np.ndarray]:
        """Get pure source signals from pyroomacoustics (post-ANC simulation).

        No distance cutoff — signal strength is determined by physics
        (propagation model), not by an artificial boundary.
        """
        from ..core.acoustic_simulator import AcousticSimulator, AcousticEnvironment

        env_type_str = (self.sim_config.env_type.value
                        if hasattr(self.sim_config.env_type, 'value')
                        else str(self.sim_config.env_type))

        env_config = AcousticEnvironment(
            room_dimensions=(
                self.sim_config.area_bounds[1] - self.sim_config.area_bounds[0],
                self.sim_config.area_bounds[3] - self.sim_config.area_bounds[2],
                12.0
            ),
            env_type=env_type_str,
            sample_rate=self.sim_config.sample_rate,
            source_duration=1.0,
            array_radius=self.hw_config.mic_array.array_radius,
            rt60_forest=self.sim_config.rt60_forest,
        )

        simulator = AcousticSimulator(env_config)
        signals = simulator.simulate_at_position(
            uav_position=uav_position,
            source_position=self.target_position,
            env_type=env_type_str
        )

        # Trim to fixed length
        n_target = self.sim_config.sample_rate
        if signals.shape[1] > n_target:
            signals = signals[:, :n_target]
        elif signals.shape[1] < n_target:
            pad = np.zeros((signals.shape[0], n_target - signals.shape[1]))
            signals = np.concatenate([signals, pad], axis=1)

        return signals

    def _compute_doa_error(self, uav_position: np.ndarray, doa_global: np.ndarray) -> float:
        """Compute angular error between estimated DOA and true direction to source."""
        true_direction = self.target_position - uav_position
        true_direction = true_direction / np.linalg.norm(true_direction)

        doa_norm = doa_global / np.linalg.norm(doa_global)
        cos_angle = np.clip(np.dot(doa_norm, true_direction), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _simulate_acoustic_signals(self, uav_position: np.ndarray) -> np.ndarray:
        """Generate microphone array signals (full pyroomacoustics or simplified)."""
        use_full_sim = self.sim_config.use_full_acoustic_sim or (self.detector is not None)

        if use_full_sim:
            return self._simulate_acoustic_signals_full(uav_position)
        else:
            return self._simulate_acoustic_signals_simplified(uav_position)

    def _simulate_acoustic_signals_full(self, uav_position: np.ndarray) -> np.ndarray:
        """Full pyroomacoustics simulation.

        No distance cutoff — pyroomacoustics naturally attenuates signals
        by distance. At large distances the source signal will be negligible
        compared to background noise, which is the correct physical behavior.
        """
        from ..core.acoustic_simulator import AcousticSimulator, AcousticEnvironment

        env_type_str = (self.sim_config.env_type.value
                        if hasattr(self.sim_config.env_type, 'value')
                        else str(self.sim_config.env_type))

        env_config = AcousticEnvironment(
            room_dimensions=(
                self.sim_config.area_bounds[1] - self.sim_config.area_bounds[0],
                self.sim_config.area_bounds[3] - self.sim_config.area_bounds[2],
                12.0
            ),
            env_type=env_type_str,
            sample_rate=self.sim_config.sample_rate,
            source_duration=1.0,
            array_radius=self.hw_config.mic_array.array_radius,
            rt60_forest=self.sim_config.rt60_forest,
        )

        simulator = AcousticSimulator(env_config)
        source_signals = simulator.simulate_at_position(
            uav_position=uav_position,
            source_position=self.target_position,
            env_type=env_type_str
        )

        # Ensure fixed output length (1.0s = sample_rate samples)
        # pyroomacoustics may return longer signals due to reverb tails
        n_target = self.sim_config.sample_rate
        n_actual = source_signals.shape[1]
        if n_actual > n_target:
            source_signals = source_signals[:, :n_target]
        elif n_actual < n_target:
            pad = np.zeros((source_signals.shape[0], n_target - n_actual))
            source_signals = np.concatenate([source_signals, pad], axis=1)

        # Mix with background noise (UAV rotor + environment)
        # In reality, mics always capture rotor noise + any source signal
        background = self._generate_background_noise()
        mic_signals = background + source_signals

        return mic_signals

    def _generate_background_noise(self) -> np.ndarray:
        """Extract background noise segment from continuous stream."""
        fs = self.sim_config.sample_rate
        n_samples = int(1.0 * fs)
        stream_length = len(self.background_noise_stream)
        start = int(self.current_time * fs) % stream_length

        if start + n_samples <= stream_length:
            segment = self.background_noise_stream[start:start + n_samples]
        else:
            part1 = self.background_noise_stream[start:]
            part2 = self.background_noise_stream[:n_samples - len(part1)]
            segment = np.concatenate([part1, part2])

        # Each mic gets: common rotor harmonics (low-amplitude) + independent noise.
        # The independent component prevents GCC-PHAT from locking to lag=0.
        # In reality, each mic captures different reflections and sensor noise.
        mic_signals = np.zeros((9, n_samples))
        for i in range(9):
            # Rotor harmonics (partially correlated, ~30% of total energy)
            mic_signals[i] = segment * 0.3
            # Independent sensor + environmental noise (~70% of total energy)
            mic_signals[i] += np.random.randn(n_samples) * np.std(segment) * 0.7

        return mic_signals

    def _simulate_acoustic_signals_simplified(self, uav_position: np.ndarray) -> np.ndarray:
        """Simplified delay-and-attenuate model for fast simulation."""
        fs = self.sim_config.sample_rate
        n_samples = int(1.0 * fs)

        mic_positions_local = self.hw_config.mic_array.get_mic_positions()
        mic_positions_global = mic_positions_local + uav_position
        distances = np.linalg.norm(mic_positions_global - self.target_position, axis=1)
        sound_speed = self.hw_config.get_sound_speed_corrected()
        arrival_times = distances / sound_speed

        stream_length = len(self.target_signal_stream)
        start = int(self.current_time * fs) % stream_length
        if start + n_samples <= stream_length:
            source_signal = self.target_signal_stream[start:start + n_samples]
        else:
            part1 = self.target_signal_stream[start:]
            part2 = self.target_signal_stream[:n_samples - len(part1)]
            source_signal = np.concatenate([part1, part2])

        mic_signals = np.zeros((9, n_samples))
        for i in range(9):
            delay_samples = int(arrival_times[i] * fs)
            if delay_samples < n_samples:
                mic_signals[i, delay_samples:] = source_signal[:n_samples - delay_samples]
            attenuation = 1.0 / (distances[i] + 1.0)
            mic_signals[i] *= attenuation
            mic_signals[i] += np.random.randn(n_samples) * 0.05

        return mic_signals

    def _estimate_tdoa_from_signals(self, mic_signals: np.ndarray):
        """Extract TDOA and GCC-PHAT peak values from microphone signals.

        Returns (tdoa_array, peak_values) — both (n_mics-1,).
        """
        from ..core.tdoa_estimator import estimate_tdoa_array

        return estimate_tdoa_array(
            mic_signals,
            fs=self.sim_config.sample_rate,
            reference_mic=0,
            max_delay_samples=100
        )

    def _triangulate_position(self) -> Optional[np.ndarray]:
        """Triangulate source from collected detections (2D ground projection)."""
        # Only use detections that have valid DOA vectors
        valid_positions = []
        valid_doas = []
        for event in self.data_logger.detections:
            if event.doa_global is not None and not np.any(np.isnan(event.doa_global)):
                valid_positions.append(event.position)
                valid_doas.append(event.doa_global)

        if len(valid_positions) < self.sim_config.min_detection_points:
            if self.sim_config.verbose:
                print(f"\n  Triangulation: insufficient valid DOA rays "
                      f"({len(valid_positions)}/{self.sim_config.min_detection_points})")
            return None

        try:
            estimated_2d, residual = triangulate_2d(valid_positions, valid_doas)
            estimated_position = np.array([estimated_2d[0], estimated_2d[1], 0.0])
            gdop = calculate_geometric_dilution_of_precision(valid_positions, valid_doas)

            if self.sim_config.verbose:
                quality = "excellent" if gdop < 2 else "good" if gdop < 5 else "moderate" if gdop < 10 else "poor"
                print(f"\n  Triangulation:")
                print(f"    Rays used: {len(valid_positions)}")
                print(f"    Estimated: {estimated_position}")
                print(f"    Residual: {residual:.4f} m")
                print(f"    GDOP: {gdop:.2f} ({quality})")

            return estimated_position

        except Exception as e:
            if self.sim_config.verbose:
                print(f"  [WARN] Triangulation failed: {e}")
            return None

    def _compute_result(
        self,
        estimated_position: Optional[np.ndarray],
        detection_count: int,
        path: List[PathPoint]
    ) -> SimulationResult:
        """Compute final simulation result metrics."""
        if estimated_position is not None:
            loc_error = np.linalg.norm(estimated_position - self.target_position)
        else:
            loc_error = np.inf

        mission_time = path[-1].timestamp if path else 0.0
        path_length = self.path_planner.get_path_length()
        detection_rate = detection_count / len(path) if path else 0.0

        # Collect trajectory as array
        traj_data = self.data_logger.trajectory
        if traj_data:
            trajectory = np.array([[p['x'], p['y'], p['z']] for p in traj_data])
        else:
            trajectory = np.zeros((0, 3))

        return SimulationResult(
            estimated_position=estimated_position,
            true_position=self.target_position,
            localization_error=loc_error,
            num_detections=detection_count,
            mission_time=mission_time,
            path_length=path_length,
            detection_rate=detection_rate,
            false_alarm_rate=0.0,
            doa_errors_deg=self.data_logger.get_doa_errors(),
            mode_switches=self.mode_switches,
            detection_positions=self.data_logger.get_detection_positions(),
            doa_vectors=self.data_logger.get_doa_vectors(),
            trajectory=trajectory,
            mode_history=self.data_logger.mode_history,
        )

    def _print_result(self, result: SimulationResult):
        print(f"\n  Results:")
        print(f"    True position:  {result.true_position}")
        if result.estimated_position is not None:
            print(f"    Est. position:  {result.estimated_position}")
            print(f"    Loc. error:     {result.localization_error:.2f} m")
        else:
            print(f"    Est. position:  FAILED")
        print(f"    Detections:     {result.num_detections}")
        print(f"    Mode switches:  {result.mode_switches}")
        print(f"    Mission time:   {result.mission_time:.1f} s ({result.mission_time / 60:.1f} min)")
        print(f"    Path length:    {result.path_length:.1f} m")
        if result.doa_errors_deg:
            print(f"    DOA error mean: {np.mean(result.doa_errors_deg):.2f} deg")
            print(f"    DOA error max:  {np.max(result.doa_errors_deg):.2f} deg")
