"""
Centralized device management, seeding, and AMP utilities for SpecMAE.

This module is the **single source of truth** for all hardware-related decisions.
All training and evaluation scripts import from here instead of duplicating
device-detection logic.

Supported backends (in priority order):
    1. CUDA  — NVIDIA GPU via CUDA toolkit
    2. MPS   — Apple Silicon GPU via Metal Performance Shaders
    3. CPU   — Fallback (still viable for SpecMAE-Base ~86 M params)

Environment variable overrides:
    SPECMAE_DEVICE=cpu        Force CPU-only mode
    SPECMAE_DEVICE=mps        Force MPS mode (skip CUDA check)
    SPECMAE_DEVICE=cuda       Force CUDA mode
    SPECMAE_DEVICE=cuda:1     Force specific CUDA device index

Example:
    >>> from SpecMae.scripts.utils.device import get_device, set_seed
    >>> device = get_device(verbose=True)
    >>> set_seed(42)
"""
from __future__ import annotations

import os
import platform
import random
import sys
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch


# ═════════════════════════════════════════════════════════════════════════════
#  Device Selection
# ═════════════════════════════════════════════════════════════════════════════

_FORCE_DEVICE: str = os.environ.get("SPECMAE_DEVICE", "").strip().lower()


def get_device(verbose: bool = False) -> torch.device:
    """
    Select the best available compute device.

    Priority: CUDA > MPS > CPU, unless overridden by $SPECMAE_DEVICE.

    The MPS path includes a functional smoke-test (allocate + compute + read-back)
    to guard against edge cases where MPS reports available but fails at runtime.

    Args:
        verbose: Print device selection rationale to stdout.

    Returns:
        torch.device for model and tensor placement.
    """
    # ── 1. Explicit override ─────────────────────────────────────────────
    if _FORCE_DEVICE:
        dev = torch.device(_FORCE_DEVICE)
        if verbose:
            print(f"  Device: {dev} (forced via $SPECMAE_DEVICE)")
        return dev

    # ── 2. CUDA ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
            print(f"  Device: {dev} — {name} ({mem_gb:.1f} GB VRAM)")
        return dev

    # ── 3. MPS (Apple Silicon) ───────────────────────────────────────────
    if _mps_is_functional():
        dev = torch.device("mps")
        if verbose:
            chip = _get_apple_chip_name()
            print(f"  Device: {dev} — {chip} (Metal Performance Shaders)")
        return dev

    # ── 4. CPU fallback ──────────────────────────────────────────────────
    dev = torch.device("cpu")
    if verbose:
        print(f"  Device: {dev} ({torch.get_num_threads()} threads)")
    return dev


def _mps_is_functional() -> bool:
    """Verify MPS is not just advertised but actually operational."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    try:
        t = torch.tensor([1.0, 2.0], device="mps")
        result = (t * t).sum().item()
        return abs(result - 5.0) < 1e-4
    except Exception:
        return False


def _get_apple_chip_name() -> str:
    """Best-effort Apple Silicon chip identification via sysctl."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=2,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor() or "Apple Silicon"


# ═════════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds across all backends for reproducibility.

    ``torch.manual_seed()`` internally seeds CPU, CUDA (all devices),
    and MPS generators — no per-backend call is needed for basic seeding.

    Args:
        seed:          Integer seed value.
        deterministic: If True, additionally force deterministic algorithm
                       selection (may reduce speed; some ops may warn).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ═════════════════════════════════════════════════════════════════════════════
#  DataLoader Configuration
# ═════════════════════════════════════════════════════════════════════════════

def should_pin_memory(device: torch.device) -> bool:
    """
    Whether DataLoader should use ``pin_memory=True``.

    Pinned (page-locked) memory accelerates CPU-to-GPU DMA transfers via
    CUDA's asynchronous copy engine.  On MPS, CPU and GPU share a single
    unified memory pool — there is nothing to "pin", and the flag adds
    unnecessary allocation overhead.  On pure CPU, pinning is pointless.

    Returns True only for CUDA.
    """
    return device.type == "cuda"


# ═════════════════════════════════════════════════════════════════════════════
#  Memory Management
# ═════════════════════════════════════════════════════════════════════════════

def empty_device_cache(device: torch.device) -> None:
    """
    Release cached allocations for the active accelerator.

    CUDA: frees blocks held by the caching allocator.
    MPS:  frees Metal heap allocations (torch.mps.empty_cache, PyTorch 2.1+).
    CPU:  no-op.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def synchronize(device: torch.device) -> None:
    """
    Block until all pending device operations complete.

    Needed for accurate wall-clock timing on asynchronous backends.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


# ═════════════════════════════════════════════════════════════════════════════
#  Automatic Mixed Precision (AMP)
# ═════════════════════════════════════════════════════════════════════════════

def supports_amp(device: torch.device) -> bool:
    """
    Whether AMP (float16 autocast + GradScaler loss scaling) is production-ready.

    Currently returns True only for CUDA.  MPS supports ``autocast`` as of
    PyTorch 2.4, but ``GradScaler`` is CUDA-only and the MPS float16 path
    is not yet mature enough for unsupervised MAE reconstruction training.
    """
    return device.type == "cuda"


def make_grad_scaler(
    device:  torch.device,
    enabled: bool = True,
) -> Optional[torch.amp.GradScaler]:
    """
    Create a ``GradScaler`` if AMP is both requested **and** supported.

    Uses the modern ``torch.amp.GradScaler`` API (PyTorch >= 2.4).

    Returns None when AMP is disabled or unavailable on this device.
    """
    if enabled and supports_amp(device):
        return torch.amp.GradScaler("cuda")
    return None


def autocast_context(device: torch.device, enabled: bool = True):
    """
    Return the appropriate autocast context manager for *device*.

    CUDA  → ``torch.amp.autocast(device_type="cuda")``
    Other → ``contextlib.nullcontext()`` (full-precision pass-through)
    """
    if enabled and supports_amp(device):
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


# ═════════════════════════════════════════════════════════════════════════════
#  Diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def print_device_diagnostics() -> None:
    """Print comprehensive hardware and PyTorch backend information."""
    sep = "-" * 56
    print(f"\n  {sep}")
    print(f"  Device & Runtime Diagnostics")
    print(f"  {sep}")
    print(f"  Python          : {sys.version.split()[0]}")
    print(f"  Platform        : {platform.platform()}")
    print(f"  PyTorch         : {torch.__version__}")

    # CUDA ---
    if torch.cuda.is_available():
        print(f"  CUDA runtime    : {torch.version.cuda}")
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
        print(f"  cuDNN           : {cudnn}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {p.name} "
                  f"({p.total_memory / (1 << 30):.1f} GB, SM {p.major}.{p.minor})")
    else:
        print(f"  CUDA            : not available")

    # MPS ---
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_func  = _mps_is_functional() if mps_avail else False
    print(f"  MPS built       : {mps_built}")
    print(f"  MPS available   : {mps_avail}")
    print(f"  MPS functional  : {mps_func}")
    if mps_func:
        print(f"    Chip: {_get_apple_chip_name()}")

    # CPU ---
    print(f"  CPU threads     : {torch.get_num_threads()}")

    # Summary ---
    dev = get_device()
    print(f"  --------------------")
    print(f"  Selected device : {dev}")
    print(f"  AMP supported   : {supports_amp(dev)}")
    print(f"  pin_memory      : {should_pin_memory(dev)}")
    print(f"  {sep}\n")
