"""Quick frequency band experiment for HPO.

Usage:
    cd /Volumes/MIKE2T
    python3 -m SpecMae.scripts.train.train_freq_experiment \
        --f_max 4000 --scenario desert --mask_ratio 0.10 --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPEC = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

import SpecMae.scripts.train.train_mask_ratio_sweep as tsm
from SpecMae.scripts.utils.feature_extraction import (
    AudioConfig, compute_dataset_stats,
)
from SpecMae.scripts.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_min", type=float, default=0.0)
    parser.add_argument("--f_max", type=float, default=4000.0)
    parser.add_argument("--n_mels", type=int, default=64,
                        help="Mel bins (reduce for narrow bands to avoid empty filters)")
    parser.add_argument("--scenario", default="desert", choices=["desert", "forest"])
    parser.add_argument("--mask_ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = get_device(verbose=True)

    sc_cfg = tsm.SCENARIO_CONFIGS[args.scenario]
    train_root = sc_cfg["data_dir"] / "train" / "normal"
    data_files = sorted(train_root.glob("*.wav"))
    print(f"\n  Training clips: {len(data_files)}")

    # Compute normalization stats for custom frequency range
    print(f"\n  Computing norm stats for f_min={args.f_min}, f_max={args.f_max}, "
          f"n_mels={args.n_mels}...")
    stats_cfg = AudioConfig(
        f_min=args.f_min, f_max=args.f_max, n_mels=args.n_mels,
    )
    norm_mean, norm_std = compute_dataset_stats(
        [str(f) for f in data_files], cfg=stats_cfg, n_samples=200,
    )
    print(f"  norm_mean={norm_mean:.3f}, norm_std={norm_std:.3f}")

    # Create AudioConfig
    cfg = AudioConfig(
        f_min=args.f_min,
        f_max=args.f_max,
        n_mels=args.n_mels,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    # Override RESULTS_ROOT to write to a separate directory
    fmin_part = f"fmin{int(args.f_min)}_" if args.f_min > 0 else ""
    tag = f"{fmin_part}fmax{int(args.f_max)}_mel{args.n_mels}"
    original_results_root = tsm.RESULTS_ROOT
    tsm.RESULTS_ROOT = original_results_root / f"freq_experiment_{tag}"
    tsm.RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    class FakeArgs:
        model_size = "base"
        batch_size = 256
        num_workers = 4
        lr = 1e-4
        force = True
        epochs = args.epochs
        amp = True

    print(f"\n  Training mr={args.mask_ratio:.2f} with f_max={args.f_max}Hz, "
          f"n_mels={args.n_mels}...")
    print(f"  Output: {tsm.RESULTS_ROOT}")

    summary = tsm.train_one_model(
        args.scenario, args.mask_ratio, cfg, data_files, device, FakeArgs(),
    )

    # Restore
    tsm.RESULTS_ROOT = original_results_root

    print(f"\n  Result: {summary}")
    ckpt = tsm.RESULTS_ROOT / f"freq_experiment_{tag}" / f"sweep_{args.scenario}" / f"mr_{args.mask_ratio:.2f}" / "model.pth"
    print(f"  Checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
