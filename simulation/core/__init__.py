"""Core signal processing modules: acoustic simulation, TDOA, DOA, triangulation."""

# Lazy imports to avoid hard dependency on pyroomacoustics for scripts
# that only need propagation_model
from .path_planner import PathPlanner, PathPoint
from .ring_buffer import RetroactiveRingBuffer

try:
    from .acoustic_simulator import AcousticSimulator, AcousticEnvironment
except ImportError:
    AcousticSimulator = None
    AcousticEnvironment = None

try:
    from .tdoa_estimator import TDOAEstimator, gcc_phat, estimate_tdoa_array
    from .doa_calculator import DOACalculator, calculate_doa, get_9mic_array_positions
    from .triangulation import Triangulator, triangulate_source, triangulate_2d
except ImportError:
    pass
