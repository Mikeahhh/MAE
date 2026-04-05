"""
CLI entry point for UAV localization simulation.

Usage:
    # Single simulation (heuristic detection, no model needed)
    python -m SpecMae.simulation.run_simulation \
        --scenario desert --target-pos 25 30 0 \
        --output results/simulation/desert

    # With SpecMAE model
    python -m SpecMae.simulation.run_simulation \
        --scenario desert \
        --checkpoint results/train_desert/checkpoints/best_model.pth \
        --train-data data/generated/desert/train/normal \
        --target-pos 25 30 0 \
        --output results/simulation/desert

    # Monte Carlo batch (50 random source positions)
    python -m SpecMae.simulation.run_simulation \
        --scenario forest --mode batch --n-runs 50 \
        --output results/simulation/forest_mc

    # From YAML config
    python -m SpecMae.simulation.run_simulation \
        --config configs/simulation_desert.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is in path
_HERE = Path(__file__).resolve().parent
_SPEC = _HERE.parent                    # SpecMae
_PROJECT = _SPEC.parent                 # /Volumes/MIKE2T
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from SpecMae.simulation.config.simulation_config import SimulationConfig, EnvironmentType
from SpecMae.simulation.config.hardware_config import HardwareConfig
from SpecMae.simulation.engine.flight_simulator import FlightSimulator, SimulationResult
from SpecMae.simulation.visualization.scene_3d import visualize_mission_3d
from SpecMae.simulation.visualization.paper_figures import (
    plot_mission_overview,
    plot_localization_error_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="UAV Localization Simulation with SpecMAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Scenario
    parser.add_argument("--scenario", choices=["desert", "forest"], default="desert",
                        help="Environment scenario")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides --scenario)")

    # Model (optional)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="SpecMAE checkpoint path")
    parser.add_argument("--n-passes", type=int, default=100,
                        help="MC passes for reconstruction scoring")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Anomaly detection threshold")

    # Simulation
    parser.add_argument("--target-pos", nargs=3, type=float, default=[25.0, 30.0, 0.0],
                        help="Target source position (x y z)")
    parser.add_argument("--velocity", type=float, default=10.0,
                        help="UAV flight speed [m/s]")
    parser.add_argument("--area", nargs=4, type=float, default=[0, 35, 0, 35],
                        help="Search area bounds (x_min x_max y_min y_max)")

    # Mode
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                        help="single simulation or Monte Carlo batch")
    parser.add_argument("--n-runs", type=int, default=50,
                        help="Number of Monte Carlo runs (batch mode)")

    # Output
    parser.add_argument("--output", type=str, default="results/simulation",
                        help="Output directory")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display figures (just save)")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def build_config_from_args(args) -> SimulationConfig:
    """Build SimulationConfig from CLI arguments."""
    if args.config is not None:
        return load_config_yaml(args.config)

    env_type = EnvironmentType.DESERT if args.scenario == "desert" else EnvironmentType.FOREST

    config = SimulationConfig(
        env_type=env_type,
        area_bounds=tuple(args.area),
        velocity=args.velocity,
        output_dir=args.output,
        random_seed=args.seed,
        model_checkpoint=args.checkpoint or "",
        anomaly_threshold=args.threshold if args.threshold else 0.5,
        # Use simplified acoustics when no model is loaded (much faster)
        use_full_acoustic_sim=args.checkpoint is not None,
    )

    return config


def load_config_yaml(yaml_path: str) -> SimulationConfig:
    """Load SimulationConfig from YAML file."""
    import yaml
    with open(yaml_path) as f:
        d = yaml.safe_load(f)

    sim = d.get("simulation", d)
    env_str = sim.get("env_type", "desert")
    env_type = EnvironmentType.DESERT if env_str == "desert" else EnvironmentType.FOREST

    return SimulationConfig(
        env_type=env_type,
        area_bounds=tuple(sim.get("area_bounds", [0, 35, 0, 35])),
        flight_height=sim.get("flight_height", 10.0),
        velocity=sim.get("velocity", 10.0),
        coverage_radius=sim.get("coverage_radius", 15.0),
        sampling_interval=sim.get("sampling_interval", 0.1),
        model_checkpoint=sim.get("checkpoint", ""),
        anomaly_threshold=sim.get("anomaly_threshold", 0.5),
        recon_n_passes=sim.get("recon_n_passes", 100),
        recon_score_mode=sim.get("recon_score_mode", "top_k"),
        recon_top_k_ratio=sim.get("recon_top_k_ratio", 0.15),
        use_full_acoustic_sim=sim.get("use_full_acoustic_sim", True),
        output_dir=sim.get("output_dir", "results/simulation"),
        random_seed=sim.get("random_seed", 42),
        verbose=sim.get("verbose", True),
    )


def build_detector(config: SimulationConfig):
    """Build DetectorBridge if checkpoint is provided."""
    if not config.model_checkpoint or not Path(config.model_checkpoint).exists():
        return None

    from SpecMae.simulation.engine.detector_bridge import DetectorBridge

    return DetectorBridge(
        checkpoint_path=config.model_checkpoint,
        anomaly_threshold=config.anomaly_threshold,
        recon_n_passes=getattr(config, 'recon_n_passes', 100),
        score_mode=getattr(config, 'recon_score_mode', 'top_k'),
        top_k_ratio=getattr(config, 'recon_top_k_ratio', 0.15),
        verbose=config.verbose,
    )


def run_single(args, config: SimulationConfig):
    """Run a single simulation."""
    target_pos = np.array(args.target_pos)

    print(f"\n  Scenario: {config.env_type.value}")
    print(f"  Target:   {target_pos}")
    print(f"  Model:    {config.model_checkpoint or 'heuristic fallback'}")

    detector = build_detector(config)
    sim = FlightSimulator(sim_config=config, detector_bridge=detector)
    sim.setup_environment(target_pos)
    result = sim.run_mission()

    # Visualization
    if not args.no_viz:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract mode labels from trajectory
        mode_labels = [p['mode'] for p in sim.data_logger.trajectory]

        # 3D scene
        visualize_mission_3d(
            trajectory=result.trajectory,
            detection_points=result.detection_positions,
            doa_vectors=result.doa_vectors,
            estimated_position=result.estimated_position,
            true_position=result.true_position,
            area_bounds=config.area_bounds,
            detection_range=config.coverage_radius,
            mode_labels=mode_labels,
            save_path=str(out_dir / "mission_3d.png"),
            show=not args.no_show,
        )

        # Multi-panel overview
        scores = [d.anomaly_score for d in sim.data_logger.detections]
        plot_mission_overview(
            trajectory=result.trajectory,
            detection_points=result.detection_positions,
            doa_vectors=result.doa_vectors,
            estimated_position=result.estimated_position,
            true_position=result.true_position,
            anomaly_scores=scores,
            anomaly_threshold=config.anomaly_threshold,
            area_bounds=config.area_bounds,
            detection_range=config.coverage_radius,
            mode_labels=mode_labels,
            save_path=str(out_dir / "mission_overview.png"),
            show=not args.no_show,
        )

    return result


def run_batch(args, config: SimulationConfig):
    """Run Monte Carlo batch of simulations."""
    rng = np.random.RandomState(config.random_seed)
    n_runs = args.n_runs
    x_min, x_max, y_min, y_max = config.area_bounds

    print(f"\n  Monte Carlo: {n_runs} runs, scenario={config.env_type.value}")

    detector = build_detector(config)

    results = []
    errors = []

    for run_idx in range(n_runs):
        # Random source position (ground level)
        target_pos = np.array([
            rng.uniform(x_min + 5, x_max - 5),
            rng.uniform(y_min + 5, y_max - 5),
            0.0
        ])

        run_config = SimulationConfig(**{
            **config.__dict__,
            'output_dir': str(Path(config.output_dir) / f"run_{run_idx:03d}"),
            'verbose': False,
        })

        sim = FlightSimulator(sim_config=run_config, detector_bridge=detector)
        sim.setup_environment(target_pos)
        result = sim.run_mission()
        results.append(result)
        errors.append(result.localization_error)

        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            valid = [e for e in errors if e < np.inf]
            mean_err = np.mean(valid) if valid else np.inf
            print(f"  Run {run_idx + 1}/{n_runs}: "
                  f"loc_error={result.localization_error:.2f}m, "
                  f"running_mean={mean_err:.2f}m")

    # Summary
    valid_errors = [e for e in errors if e < np.inf]
    doa_all = []
    for r in results:
        doa_all.extend(r.doa_errors_deg)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'n_runs': n_runs,
        'scenario': config.env_type.value,
        'n_successful': len(valid_errors),
        'success_rate': len(valid_errors) / n_runs,
        'loc_error_mean': float(np.mean(valid_errors)) if valid_errors else None,
        'loc_error_std': float(np.std(valid_errors)) if valid_errors else None,
        'loc_error_median': float(np.median(valid_errors)) if valid_errors else None,
        'loc_error_max': float(np.max(valid_errors)) if valid_errors else None,
        'doa_error_mean': float(np.mean(doa_all)) if doa_all else None,
        'doa_error_std': float(np.std(doa_all)) if doa_all else None,
    }

    with open(out_dir / "batch_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  {'=' * 50}")
    print(f"  Monte Carlo Summary ({n_runs} runs)")
    print(f"  {'=' * 50}")
    print(f"    Success rate:    {summary['success_rate'] * 100:.1f}%")
    if valid_errors:
        print(f"    Loc error mean:  {summary['loc_error_mean']:.2f} m")
        print(f"    Loc error std:   {summary['loc_error_std']:.2f} m")
        print(f"    Loc error median:{summary['loc_error_median']:.2f} m")
    if doa_all:
        print(f"    DOA error mean:  {summary['doa_error_mean']:.2f} deg")
        print(f"    DOA error std:   {summary['doa_error_std']:.2f} deg")
    print(f"    Results saved:   {out_dir / 'batch_summary.json'}")

    # Visualization
    if not args.no_viz and valid_errors:
        plot_localization_error_distribution(
            errors={config.env_type.value: valid_errors},
            save_path=str(out_dir / "mc_error_distribution.png"),
            show=not args.no_show,
        )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    config = build_config_from_args(args)

    if args.mode == "single":
        run_single(args, config)
    elif args.mode == "batch":
        run_batch(args, config)


if __name__ == "__main__":
    main()
