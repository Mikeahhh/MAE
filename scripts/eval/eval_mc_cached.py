"""
Monte Carlo cached evaluation script.

Loads pre-trained SpecMAE weights ONCE, then runs N MC passes with
different random masks for each test sample. Averages reconstruction
error across passes to produce stable anomaly scores.

This avoids re-loading/re-training per evaluation — MC 100 passes
takes minutes, not hours.

Outputs:
    - Per-sample anomaly scores
    - AUC / pAUC metrics (overall and per-SNR)
    - Score distribution plots
    - Comparison across different n_passes values (smoothing curve)

Usage:
    # Basic evaluation
    python SpecMae/scripts/eval/eval_mc_cached.py \
        --checkpoint results/sweep_desert/mr_0.75/model.pth \
        --data_dir   SpecMae/data/desert \
        --out_dir    results/mc_eval_desert

    # Sweep n_passes to see smoothing effect
    python SpecMae/scripts/eval/eval_mc_cached.py \
        --checkpoint results/sweep_desert/mr_0.75/model.pth \
        --data_dir   SpecMae/data/desert \
        --out_dir    results/mc_eval_desert \
        --sweep_passes 1 5 10 20 50 100

    # Sweep top_k_ratio
    python SpecMae/scripts/eval/eval_mc_cached.py \
        --checkpoint results/sweep_desert/mr_0.75/model.pth \
        --data_dir   SpecMae/data/desert \
        --out_dir    results/mc_eval_desert \
        --sweep_top_k 0.10 0.15 0.20 0.25 0.30

    # Use tiny model
    python SpecMae/scripts/eval/eval_mc_cached.py \
        --checkpoint results/train_desert_tiny/checkpoints/best_model.pth \
        --data_dir   SpecMae/data/desert \
        --model_size tiny
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from SpecMae.models.specmae import SpecMAE, get_model_factory
from SpecMae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor
from SpecMae.scripts.utils.data_loader import AnomalyTestDataset
from SpecMae.scripts.utils.device import get_device
from SpecMae.scripts.eval.compute_metrics import compute_metrics_per_snr, print_metrics_table


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(
    ckpt_path: Path,
    device: torch.device,
    model_size: str = "base",
) -> tuple[SpecMAE, float, AudioConfig]:
    """Load model weights once (cached for all MC passes)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mask_ratio = float(ckpt.get("mask_ratio", 0.75))
    cfg_dict = ckpt.get("audio_cfg", {})
    cfg = AudioConfig(
        sample_rate=cfg_dict.get("sample_rate", 48_000),
        n_mels=cfg_dict.get("n_mels", 128),
        n_fft=cfg_dict.get("n_fft", 1_024),
        hop_length=cfg_dict.get("hop_length", 480),
        norm_mean=cfg_dict.get("norm_mean", -6.0),
        norm_std=cfg_dict.get("norm_std", 5.0),
    )
    factory = get_model_factory(model_size)
    model = factory(mask_ratio=mask_ratio, norm_pix_loss=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, mask_ratio, cfg


# ═══════════════════════════════════════════════════════════════════════════
#  MC-averaged reconstruction scoring
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def mc_recon_scores(
    model: SpecMAE,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    n_passes: int = 50,
    score_mode: str = "top_k",
    top_k_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute MC-averaged reconstruction anomaly scores.

    Returns:
        scores: (N,) anomaly scores
        labels: (N,) ground truth labels
        snr_tags: list of SNR tags
    """
    model.eval()
    all_scores, all_labels, all_snr = [], [], []

    for specs, labels, snr_tags in loader:
        specs = specs.to(device, non_blocking=True)
        scores = model.compute_anomaly_score(
            specs, mask_ratio=mask_ratio, n_passes=n_passes,
            score_mode=score_mode, top_k_ratio=top_k_ratio,
        ).cpu().numpy()
        all_scores.extend(float(s) for s in scores)
        all_labels.extend(int(lb) for lb in labels)
        all_snr.extend(snr_tags)

    return np.array(all_scores), np.array(all_labels), all_snr


@torch.no_grad()
def mc_multiscale_scores(
    model: SpecMAE,
    loader: DataLoader,
    device: torch.device,
    mask_ratios: tuple[float, ...] = (0.3, 0.5, 0.7),
    n_passes: int = 50,
    score_mode: str = "top_k",
    top_k_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Multi-scale MC-averaged reconstruction scores."""
    model.eval()
    all_scores, all_labels, all_snr = [], [], []

    for specs, labels, snr_tags in loader:
        specs = specs.to(device, non_blocking=True)
        scores = model.compute_multiscale_anomaly_score(
            specs, mask_ratios=mask_ratios, n_passes=n_passes,
            score_mode=score_mode, top_k_ratio=top_k_ratio,
        ).cpu().numpy()
        all_scores.extend(float(s) for s in scores)
        all_labels.extend(int(lb) for lb in labels)
        all_snr.extend(snr_tags)

    return np.array(all_scores), np.array(all_labels), all_snr


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_mc_smoothing_curve(results: dict, out_dir: Path) -> None:
    """Plot AUC vs n_passes to show MC smoothing effect."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        passes = sorted(results.keys())
        aucs = [results[p]["overall"]["auc"] for p in passes]
        paucs = [results[p]["overall"]["pauc"] for p in passes]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(passes, aucs, "o-", label="AUC", linewidth=2, markersize=6)
        ax.plot(passes, paucs, "s--", label="pAUC (FPR≤0.1)", linewidth=2, markersize=6)
        ax.set_xlabel("MC passes (n_passes)")
        ax.set_ylabel("Score")
        ax.set_title("MC smoothing effect on detection performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / "mc_smoothing_curve.png", dpi=150)
        plt.close(fig)
        print(f"  Smoothing curve saved: {out_dir / 'mc_smoothing_curve.png'}")
    except Exception as e:
        print(f"  WARNING: Could not plot MC smoothing curve: {e}")


def plot_top_k_sweep(results: dict, out_dir: Path) -> None:
    """Plot AUC vs top_k_ratio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ratios = sorted(results.keys())
        aucs = [results[r]["overall"]["auc"] for r in ratios]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ratios, aucs, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("top_k_ratio")
        ax.set_ylabel("AUC")
        ax.set_title("Reconstruction AUC vs top_k_ratio")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "top_k_sweep.png", dpi=150)
        plt.close(fig)
        print(f"  top_k sweep plot saved: {out_dir / 'top_k_sweep.png'}")
    except Exception as e:
        print(f"  WARNING: Could not plot top_k sweep: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MC-cached evaluation (load weights once, run N MC passes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="SpecMae/data/desert",
                        help="Scenario data root")
    parser.add_argument("--out_dir", default="results/mc_eval",
                        help="Output directory")
    parser.add_argument("--model_size", default="base",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_passes", type=int, default=50,
                        help="Default MC passes")
    parser.add_argument("--score_mode", default="top_k",
                        choices=["mean", "max", "top_k"])
    parser.add_argument("--top_k_ratio", type=float, default=0.15)
    parser.add_argument("--multiscale", action="store_true",
                        help="Use multi-scale reconstruction")
    parser.add_argument("--max_fpr", type=float, default=0.1)
    parser.add_argument("--sweep_passes", nargs="+", type=int, default=None,
                        help="Sweep n_passes values (e.g. 1 5 10 20 50 100)")
    parser.add_argument("--sweep_top_k", nargs="+", type=float, default=None,
                        help="Sweep top_k_ratio values (e.g. 0.10 0.15 0.20 0.25 0.30)")
    args = parser.parse_args()

    device = get_device(verbose=True)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model (ONCE) ─────────────────────────────────────────────
    print(f"\n  Loading model ({args.model_size})...")
    model, mask_ratio, cfg = load_model(
        Path(args.checkpoint), device, model_size=args.model_size,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded. mask_ratio={mask_ratio}, params={n_params/1e6:.1f}M")

    extractor = LogMelExtractor(cfg=cfg)

    # ── Build test loader ─────────────────────────────────────────────
    test_ds = AnomalyTestDataset(
        normal_dir=data_dir / "test" / "normal",
        anomaly_dir=data_dir / "test" / "anomaly",
        extractor=extractor,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    n_normal = sum(1 for _, lb, _ in test_ds.samples if lb == 0)
    n_anomaly = sum(1 for _, lb, _ in test_ds.samples if lb == 1)
    print(f"  Test set: {n_normal} normal + {n_anomaly} anomaly = {len(test_ds)} total")

    all_results = {}

    # ── Mode 1: Sweep n_passes (MC smoothing analysis) ────────────────
    if args.sweep_passes:
        print(f"\n{'='*60}")
        print(f"  MC passes sweep: {args.sweep_passes}")
        print(f"{'='*60}")

        sweep_results = {}
        for np_ in args.sweep_passes:
            print(f"\n  n_passes={np_}...")
            t0 = time.time()

            if args.multiscale:
                scores, labels, snr_tags = mc_multiscale_scores(
                    model, test_loader, device,
                    n_passes=np_, score_mode=args.score_mode,
                    top_k_ratio=args.top_k_ratio,
                )
            else:
                scores, labels, snr_tags = mc_recon_scores(
                    model, test_loader, device,
                    mask_ratio=mask_ratio, n_passes=np_,
                    score_mode=args.score_mode,
                    top_k_ratio=args.top_k_ratio,
                )

            elapsed = time.time() - t0
            metrics = compute_metrics_per_snr(
                labels, scores, snr_tags, max_fpr=args.max_fpr,
            )
            sweep_results[np_] = metrics
            auc = metrics.get("overall", {}).get("auc", 0)
            pauc = metrics.get("overall", {}).get("pauc", 0)
            print(f"    AUC={auc:.4f}  pAUC={pauc:.4f}  ({elapsed:.1f}s)")

        all_results["mc_sweep"] = {str(k): v for k, v in sweep_results.items()}
        plot_mc_smoothing_curve(sweep_results, out_dir)

        # Print summary table
        print(f"\n  {'n_passes':>10s}  {'AUC':>8s}  {'pAUC':>8s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
        for np_ in sorted(sweep_results.keys()):
            m = sweep_results[np_]
            auc = m.get("overall", {}).get("auc", 0)
            pauc = m.get("overall", {}).get("pauc", 0)
            print(f"  {np_:>10d}  {auc:>8.4f}  {pauc:>8.4f}")

    # ── Mode 2: Sweep top_k_ratio ─────────────────────────────────────
    elif args.sweep_top_k:
        print(f"\n{'='*60}")
        print(f"  top_k_ratio sweep: {args.sweep_top_k}")
        print(f"{'='*60}")

        sweep_results = {}
        for tk in args.sweep_top_k:
            print(f"\n  top_k_ratio={tk}...")
            t0 = time.time()

            scores, labels, snr_tags = mc_recon_scores(
                model, test_loader, device,
                mask_ratio=mask_ratio, n_passes=args.n_passes,
                score_mode="top_k", top_k_ratio=tk,
            )

            elapsed = time.time() - t0
            metrics = compute_metrics_per_snr(
                labels, scores, snr_tags, max_fpr=args.max_fpr,
            )
            sweep_results[tk] = metrics
            auc = metrics.get("overall", {}).get("auc", 0)
            print(f"    AUC={auc:.4f}  ({elapsed:.1f}s)")

        all_results["top_k_sweep"] = {str(k): v for k, v in sweep_results.items()}
        plot_top_k_sweep(sweep_results, out_dir)

        # Print summary
        print(f"\n  {'top_k_ratio':>12s}  {'AUC':>8s}")
        print(f"  {'-'*12}  {'-'*8}")
        for tk in sorted(sweep_results.keys()):
            m = sweep_results[tk]
            auc = m.get("overall", {}).get("auc", 0)
            print(f"  {tk:>12.2f}  {auc:>8.4f}")

    # ── Mode 3: Single evaluation ─────────────────────────────────────
    else:
        print(f"\n  Running MC evaluation (n_passes={args.n_passes}, "
              f"score_mode={args.score_mode}, top_k_ratio={args.top_k_ratio})...")
        t0 = time.time()

        if args.multiscale:
            scores, labels, snr_tags = mc_multiscale_scores(
                model, test_loader, device,
                n_passes=args.n_passes, score_mode=args.score_mode,
                top_k_ratio=args.top_k_ratio,
            )
        else:
            scores, labels, snr_tags = mc_recon_scores(
                model, test_loader, device,
                mask_ratio=mask_ratio, n_passes=args.n_passes,
                score_mode=args.score_mode,
                top_k_ratio=args.top_k_ratio,
            )

        elapsed = time.time() - t0
        metrics = compute_metrics_per_snr(
            labels, scores, snr_tags, max_fpr=args.max_fpr,
        )
        all_results["evaluation"] = metrics
        print(f"\n  === MC Reconstruction (n_passes={args.n_passes}) ===")
        print_metrics_table(metrics)
        print(f"  Evaluation time: {elapsed:.1f}s")

        # Save scores
        scores_path = out_dir / "mc_scores.npz"
        np.savez(
            scores_path,
            scores=scores,
            labels=labels,
            snr_tags=np.array(snr_tags),
        )
        print(f"  Scores saved: {scores_path}")

    # ── Save all results ──────────────────────────────────────────────
    results_path = out_dir / "mc_eval_results.json"
    all_results["config"] = {
        "checkpoint": str(args.checkpoint),
        "model_size": args.model_size,
        "mask_ratio": mask_ratio,
        "n_params": n_params,
        "n_passes": args.n_passes,
        "score_mode": args.score_mode,
        "top_k_ratio": args.top_k_ratio,
        "multiscale": args.multiscale,
    }
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
