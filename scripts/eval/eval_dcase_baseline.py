"""
DCASE autoencoder baseline evaluation (Plan B).

If SpecMAE detection accuracy < 30%, this baseline replaces it.

Trains a simple autoencoder on normal data and computes reconstruction
error as the anomaly score, following the DCASE Task 2 methodology.

Usage:
    python -m SpecMae.scripts.eval.eval_dcase_baseline --scenario desert
    python -m SpecMae.scripts.eval.eval_dcase_baseline --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_SPEC = Path(__file__).resolve().parents[2]  # SpecMae/
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from SpecMae.scripts.utils.feature_extraction import LogMelExtractor
from SpecMae.scripts.utils.device import get_device

RESULTS_ROOT = _SPEC / "results"
DATA_ROOT = _SPEC / "data"

# ═════════════════════════════════════════════════════════════════════════
#  DCASE-style Autoencoder
# ═════════════════════════════════════════════════════════════════════════

class DcaseAutoencoder(nn.Module):
    """Simple fully-connected autoencoder (DCASE Task 2 baseline architecture)."""

    def __init__(self, input_dim: int = 640, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ═════════════════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════════════════

def load_features(data_dir: Path, extractor: LogMelExtractor) -> np.ndarray:
    """Load all WAV files from a directory and extract flattened log-mel features."""
    import soundfile as sf

    features = []
    wav_files = sorted(data_dir.glob("*.wav"))
    for wav_path in wav_files:
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        mel = extractor.extract(audio)  # (n_mels, n_frames)
        features.append(mel.flatten())
    if not features:
        return np.array([])
    # Truncate/pad to uniform length
    min_len = min(f.shape[0] for f in features)
    features = np.array([f[:min_len] for f in features], dtype=np.float32)
    return features


# ═════════════════════════════════════════════════════════════════════════
#  Training and evaluation
# ═════════════════════════════════════════════════════════════════════════

def train_dcase_ae(
    train_features: np.ndarray,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> DcaseAutoencoder:
    """Train DCASE autoencoder on normal training data."""
    input_dim = train_features.shape[1]
    model = DcaseAutoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.tensor(train_features, dtype=torch.float32)
    n_samples = len(dataset)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            batch = dataset[perm[i:i + batch_size]].to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={epoch_loss/n_batches:.6f}")

    return model


def compute_anomaly_scores(
    model: DcaseAutoencoder,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Compute per-sample reconstruction MSE (anomaly score)."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(device)
        recon = model(x)
        mse = ((x - recon) ** 2).mean(dim=1).cpu().numpy()
    return mse


def evaluate_dcase_baseline(
    scenario: str,
    device: torch.device,
    n_epochs: int = 100,
) -> dict:
    """Full DCASE baseline pipeline for one scenario."""
    print(f"\n  DCASE Baseline: {scenario.upper()}")

    extractor = LogMelExtractor()

    # Load training data
    train_dir = DATA_ROOT / scenario / scenario / "train" / "normal"
    if not train_dir.exists():
        print(f"    ERROR: Training dir not found: {train_dir}")
        return {}

    print(f"    Loading training features...")
    train_features = load_features(train_dir, extractor)
    if len(train_features) == 0:
        print(f"    ERROR: No training features loaded")
        return {}
    print(f"    Training features: {train_features.shape}")

    # Train
    print(f"    Training autoencoder...")
    model = train_dcase_ae(train_features, device, n_epochs=n_epochs)

    # Compute threshold from training data
    train_scores = compute_anomaly_scores(model, train_features, device)
    threshold = np.mean(train_scores) + 3 * np.std(train_scores)
    print(f"    Threshold: {threshold:.6f} (mean + 3*std)")

    # Evaluate on test data
    results = {
        "scenario": scenario,
        "threshold": float(threshold),
        "train_mean_mse": float(np.mean(train_scores)),
        "train_std_mse": float(np.std(train_scores)),
    }

    # Test normal
    test_normal_dir = DATA_ROOT / scenario / scenario / "test" / "normal"
    if test_normal_dir.exists():
        normal_features = load_features(test_normal_dir, extractor)
        if len(normal_features) > 0:
            normal_scores = compute_anomaly_scores(model, normal_features, device)
            fp = np.sum(normal_scores > threshold)
            results["normal_fp_rate"] = float(fp / len(normal_scores) * 100)
            print(f"    Normal FP rate: {results['normal_fp_rate']:.1f}%")

    # Test anomaly (per SNR)
    anomaly_base = DATA_ROOT / scenario / scenario / "test" / "anomaly"
    if anomaly_base.exists():
        per_snr = {}
        for snr_dir in sorted(anomaly_base.glob("snr_*")):
            anomaly_features = load_features(snr_dir, extractor)
            if len(anomaly_features) == 0:
                continue
            anomaly_scores = compute_anomaly_scores(model, anomaly_features, device)
            detected = np.sum(anomaly_scores > threshold)
            det_rate = float(detected / len(anomaly_scores) * 100)
            per_snr[snr_dir.name] = {
                "n_samples": len(anomaly_scores),
                "n_detected": int(detected),
                "detection_rate": det_rate,
            }
            print(f"    {snr_dir.name}: {det_rate:.1f}% ({detected}/{len(anomaly_scores)})")
        results["per_snr"] = per_snr

        # Overall detection accuracy
        all_det = sum(v["n_detected"] for v in per_snr.values())
        all_total = sum(v["n_samples"] for v in per_snr.values())
        if all_total > 0:
            results["overall_detection_accuracy"] = float(all_det / all_total * 100)
            print(f"    Overall detection: {results['overall_detection_accuracy']:.1f}%")

    return results


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DCASE autoencoder baseline evaluation (Plan B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default=None, choices=["desert", "forest"])
    parser.add_argument("--all", action="store_true", help="Run both scenarios")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.all:
        scenarios = ["desert", "forest"]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        parser.error("Specify --scenario or --all")

    device = get_device(verbose=True)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / "dcase_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        results = evaluate_dcase_baseline(scenario, device, n_epochs=args.n_epochs)
        if results:
            out_path = out_dir / f"dcase_baseline_{scenario}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    Results saved: {out_path}")

            # Plan B check
            acc = results.get("overall_detection_accuracy", 0)
            if acc < 30:
                print(f"\n    ⚠ WARNING: DCASE baseline accuracy {acc:.1f}% < 30%")
                print(f"    → Plan B: consider replacing SpecMAE with DCASE baseline")

    print("\nDone.")


if __name__ == "__main__":
    main()
