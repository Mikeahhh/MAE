"""
Batch training of SpecMAE models across mask ratios and scenarios.

Trains 17 mask_ratio values x 2 scenarios = 34 models total.
Each model is trained for a configurable number of epochs (default: 100).

Usage:
    # Train all 34 models
    python -m SpecMae.scripts.train.train_mask_ratio_sweep

    # Train only desert
    python -m SpecMae.scripts.train.train_mask_ratio_sweep --scenario desert

    # Quick sweep (fewer epochs)
    python -m SpecMae.scripts.train.train_mask_ratio_sweep --epochs 50

    # Train specific mask ratios
    python -m SpecMae.scripts.train.train_mask_ratio_sweep --mask_ratios 0.50 0.60 0.70

Output:
    results/sweep_{scenario}/mr_{ratio}/
        model.pth          best checkpoint
        train_log.csv      per-epoch metrics
        training_curve.png loss plot
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE    = Path(__file__).resolve().parent
_SPEC    = _HERE.parent.parent
_PROJECT = _SPEC.parent
sys.path.insert(0, str(_PROJECT))

from SpecMae.models.specmae import get_model_factory
from SpecMae.scripts.utils.feature_extraction import AudioConfig, LogMelExtractor
from SpecMae.scripts.utils.data_loader import make_kfold_loaders
from SpecMae.scripts.utils.device import (
    get_device, set_seed, supports_amp, make_grad_scaler,
    print_device_diagnostics,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Defaults
# ═══════════════════════════════════════════════════════════════════════════

MASK_RATIOS = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90

SCENARIO_CONFIGS = {
    "desert": {
        "data_dir": _SPEC / "data" / "generated" / "desert",
        "norm_mean": -10.507,
        "norm_std": 8.245,
    },
    "forest": {
        "data_dir": _SPEC / "data" / "generated" / "forest",
        "norm_mean": -10.489,
        "norm_std": 8.070,
    },
}

EPOCHS        = 200
BATCH_SIZE    = 256
LR            = 2.8e-4
MIN_LR        = 1e-6
WEIGHT_DECAY  = 0.05
WARMUP_EPOCHS = 15
GRAD_CLIP     = 1.0
VAL_FRAC      = 0.1
SAVE_EVERY    = 50
PATIENCE      = 60
SEED          = 42

RESULTS_ROOT = _SPEC / "results"


def adaptive_batch_size(mask_ratio: float, base_bs: int = 256) -> int:
    """Scale batch size down for low mask ratios to avoid VRAM OOM.

    Low mask_ratio → more visible tokens → O(N²) attention → more VRAM.
    RTX 5060 8GB safe limits (empirical):
      mr >= 0.60 → 256, mr >= 0.40 → 192, mr >= 0.20 → 128, else → 64
    """
    if mask_ratio >= 0.60:
        return min(base_bs, 256)
    if mask_ratio >= 0.40:
        return min(base_bs, 192)
    if mask_ratio >= 0.20:
        return min(base_bs, 128)
    return min(base_bs, 64)


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def cosine_lr(epoch: int, total_epochs: int, base_lr: float,
              min_lr: float, warmup: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════════════════════
#  GPU pre-loading (entire dataset fits in VRAM for maximum throughput)
# ═══════════════════════════════════════════════════════════════════════════

def preload_to_gpu(
    files: list[Path], device: torch.device,
) -> torch.Tensor:
    """Load all pre-computed .pt spectrograms into a single GPU tensor."""
    tensors = []
    for f in files:
        pt = f.with_suffix(".pt")
        if pt.exists():
            tensors.append(torch.load(pt, weights_only=True))
        else:
            raise FileNotFoundError(f"Cache miss: {pt} — run precompute_spectrograms first")
    stacked = torch.stack(tensors, dim=0)       # (N, 1, n_mels, T)
    return stacked.to(device, non_blocking=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Training logic — GPU-resident data path (no DataLoader overhead)
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, loader, optimizer, scaler, device,
    mask_ratio, epoch, total_epochs, base_lr, min_lr, warmup,
    *, gpu_data: torch.Tensor | None = None, batch_size: int = 256,
) -> tuple[float, float]:
    model.train()
    lr = cosine_lr(epoch, total_epochs, base_lr, min_lr, warmup)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    total_loss = 0.0
    n_batches = 0

    if gpu_data is not None:
        # Fast path: data already on GPU, shuffle and batch manually
        N = gpu_data.shape[0]
        perm = torch.randperm(N, device=gpu_data.device)
        for start in range(0, N - batch_size + 1, batch_size):
            idx = perm[start : start + batch_size]
            specs = gpu_data[idx]
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    loss, _, _ = model(specs, mask_ratio=mask_ratio)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _, _ = model(specs, mask_ratio=mask_ratio)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    else:
        # Fallback: DataLoader path
        for specs, _ in loader:
            specs = specs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    loss, _, _ = model(specs, mask_ratio=mask_ratio)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _, _ = model(specs, mask_ratio=mask_ratio)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1), lr


@torch.no_grad()
def evaluate(model, loader, device, mask_ratio,
             *, gpu_data: torch.Tensor | None = None, batch_size: int = 256) -> float:
    model.eval()
    losses = []

    if gpu_data is not None:
        N = gpu_data.shape[0]
        for start in range(0, N, batch_size):
            specs = gpu_data[start : start + batch_size]
            loss, _, _ = model(specs, mask_ratio=mask_ratio)
            losses.append(loss.item())
    else:
        for specs, _ in loader:
            specs = specs.to(device, non_blocking=True)
            loss, _, _ = model(specs, mask_ratio=mask_ratio)
            losses.append(loss.item())

    return float(np.mean(losses))


def save_checkpoint(path, epoch, model, optimizer, scaler,
                    val_loss, mask_ratio, cfg):
    payload = {
        "epoch": epoch,
        "mask_ratio": mask_ratio,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "audio_cfg": {
            "sample_rate": cfg.sample_rate,
            "n_mels": cfg.n_mels,
            "n_fft": cfg.n_fft,
            "hop_length": cfg.hop_length,
            "f_min": cfg.f_min,
            "f_max": cfg.f_max,
            "norm_mean": cfg.norm_mean,
            "norm_std": cfg.norm_std,
        },
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def save_training_curve(train_losses, val_records, mask_ratio, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = list(range(1, len(train_losses) + 1))
        ax.plot(epochs, train_losses, alpha=0.5, linewidth=0.8, label="train")
        if val_records:
            ep_v, v_vals = zip(*val_records)
            ax.plot(ep_v, v_vals, "o-", markersize=3, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"mask_ratio={mask_ratio:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "training_curve.png", dpi=100)
        plt.close(fig)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Train one model
# ═══════════════════════════════════════════════════════════════════════════

def train_one_model(
    scenario: str,
    mask_ratio: float,
    cfg: AudioConfig,
    data_files: list[Path],
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    """Train a single SpecMAE model. Returns summary dict."""

    size_suffix = f"_{args.model_size}" if args.model_size != "base" else ""
    out_dir = RESULTS_ROOT / f"sweep_{scenario}{size_suffix}" / f"mr_{mask_ratio:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pth"

    # Skip if already trained
    if model_path.exists() and not args.force:
        print(f"    [SKIP] {model_path} exists (use --force to retrain)")
        return {"mask_ratio": mask_ratio, "status": "skipped"}

    # Adaptive batch size: low mask_ratio → more visible tokens → more VRAM
    eff_bs = adaptive_batch_size(mask_ratio, args.batch_size)
    if eff_bs != args.batch_size:
        print(f"    [ADAPTIVE BS] {args.batch_size} → {eff_bs} (mr={mask_ratio:.2f})")

    # Free VRAM from previous model before loading new one
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Data split
    rng = np.random.default_rng(SEED)
    n_total = len(data_files)
    n_val = max(1, int(n_total * VAL_FRAC))
    idx = rng.permutation(n_total).tolist()
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    # Try GPU preloading (fastest); fall back to DataLoader
    gpu_train, gpu_val = None, None
    train_loader, val_loader = None, None
    sample_pt = data_files[0].with_suffix(".pt")
    if sample_pt.exists() and device.type == "cuda":
        try:
            gpu_train = preload_to_gpu([data_files[i] for i in trn_idx], device)
            gpu_val   = preload_to_gpu([data_files[i] for i in val_idx], device)
            print(f"    [GPU PRELOAD] train={gpu_train.shape[0]}, val={gpu_val.shape[0]}, "
                  f"VRAM ~{(gpu_train.nelement() + gpu_val.nelement())*4/1e6:.0f} MB")
        except Exception as exc:
            print(f"    [WARN] GPU preload failed ({exc}), falling back to DataLoader")
            gpu_train, gpu_val = None, None

    if gpu_train is None:
        train_loader, val_loader = make_kfold_loaders(
            data_files, trn_idx, val_idx, cfg=cfg,
            batch_size=eff_bs, num_workers=args.num_workers,
        )

    # Model
    set_seed(SEED)
    model_factory = get_model_factory(args.model_size)
    model = model_factory(
        mask_ratio=mask_ratio, norm_pix_loss=True, drop_path_rate=0.1,
        n_mels=cfg.n_mels,
    ).to(device)

    # Optimizer
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR, betas=(0.9, 0.95),
    )

    scaler = make_grad_scaler(device, enabled=args.amp)

    # CSV log
    log_path = out_dir / "train_log.csv"
    log_fields = ["epoch", "lr", "train_loss", "val_loss", "elapsed_s"]
    csv_file = open(log_path, "w", newline="")
    csv_wr = csv.DictWriter(csv_file, fieldnames=log_fields)
    csv_wr.writeheader()

    # Training loop
    best_val = float("inf")
    no_improve = 0
    train_losses = []
    val_records = []
    t_start = time.time()
    log_every = max(1, args.epochs // 10)

    for epoch in range(args.epochs):
        tr_loss, lr = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            mask_ratio, epoch, args.epochs, LR, MIN_LR, WARMUP_EPOCHS,
            gpu_data=gpu_train, batch_size=eff_bs,
        )
        train_losses.append(tr_loss)

        do_val = ((epoch + 1) % log_every == 0) or (epoch == args.epochs - 1)
        if do_val:
            val_loss = evaluate(model, val_loader, device, mask_ratio,
                                gpu_data=gpu_val, batch_size=eff_bs)
            val_records.append((epoch + 1, val_loss))

            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                no_improve = 0
                save_checkpoint(model_path, epoch + 1, model, optimizer,
                                scaler, val_loss, mask_ratio, cfg)
            else:
                no_improve += log_every

            csv_wr.writerow({
                "epoch": epoch + 1,
                "lr": f"{lr:.2e}",
                "train_loss": f"{tr_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "elapsed_s": f"{time.time() - t_start:.1f}",
            })
            csv_file.flush()

            if no_improve >= PATIENCE:
                break

        # Periodic snapshots
        if (epoch + 1) % SAVE_EVERY == 0 and (epoch + 1) != args.epochs:
            snap_path = out_dir / f"ckpt_epoch_{epoch+1:04d}.pth"
            save_checkpoint(snap_path, epoch + 1, model, optimizer,
                            scaler, best_val, mask_ratio, cfg)

    csv_file.close()
    elapsed = time.time() - t_start

    # Save training curve
    save_training_curve(train_losses, val_records, mask_ratio, out_dir)

    # If no best was ever saved (should not happen), save final
    if not model_path.exists():
        save_checkpoint(model_path, args.epochs, model, optimizer,
                        scaler, best_val, mask_ratio, cfg)

    summary = {
        "mask_ratio": mask_ratio,
        "best_val_loss": best_val,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "epochs_trained": len(train_losses),
        "elapsed_s": round(elapsed, 1),
        "status": "completed",
    }

    print(f"    mr={mask_ratio:.2f}  val={best_val:.5f}  "
          f"ep={len(train_losses)}  [{fmt_time(elapsed)}]")

    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SpecMAE mask-ratio sweep training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default=None,
                        choices=["desert", "forest"],
                        help="Scenario (default: both)")
    parser.add_argument("--model_size", default="base",
                        choices=["tiny", "small", "base", "large"],
                        help="Model variant: tiny (~5M), small (~22M), base (~98M), large")
    parser.add_argument("--mask_ratios", nargs="+", type=float, default=None,
                        help="Override mask ratios (default: 0.10 to 0.90 step 0.05)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use AMP (CUDA only, enabled by default)")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if checkpoint exists")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    mask_ratios = args.mask_ratios or MASK_RATIOS
    scenarios = [args.scenario] if args.scenario else ["desert", "forest"]

    device = get_device(verbose=True)
    print_device_diagnostics()

    print(f"\n  Mask ratios: {mask_ratios}")
    print(f"  Scenarios:   {scenarios}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Total models: {len(mask_ratios) * len(scenarios)}")

    all_summaries = {}

    for scenario in scenarios:
        sc_cfg = SCENARIO_CONFIGS[scenario]
        data_dir = sc_cfg["data_dir"]
        train_root = data_dir / "train" / "normal"

        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario}")
        print(f"  Data: {train_root}")
        print(f"{'='*60}")

        # Collect training files
        data_files = sorted(train_root.glob("*.wav"))
        if not data_files:
            print(f"  ERROR: No WAV files in {train_root}. Skipping.")
            continue

        print(f"  Training clips: {len(data_files)}")

        cfg = AudioConfig(
            norm_mean=sc_cfg["norm_mean"],
            norm_std=sc_cfg["norm_std"],
        )

        scenario_summaries = []
        for i, mr in enumerate(mask_ratios):
            print(f"\n  [{i+1}/{len(mask_ratios)}] Training mask_ratio={mr:.2f}")
            summary = train_one_model(
                scenario, mr, cfg, data_files, device, args,
            )
            scenario_summaries.append(summary)

        all_summaries[scenario] = scenario_summaries

        # Save per-scenario sweep summary
        size_suffix = f"_{args.model_size}" if args.model_size != "base" else ""
        sweep_dir = RESULTS_ROOT / f"sweep_{scenario}{size_suffix}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        summary_path = sweep_dir / "sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(scenario_summaries, f, indent=2)
        print(f"\n  Sweep summary saved: {summary_path}")

        # Find best
        completed = [s for s in scenario_summaries if s["status"] == "completed"]
        if completed:
            best = min(completed, key=lambda s: s["best_val_loss"])
            print(f"  Best mask_ratio for {scenario}: {best['mask_ratio']:.2f} "
                  f"(val_loss={best['best_val_loss']:.5f})")

    print(f"\nAll training complete.")


if __name__ == "__main__":
    main()
