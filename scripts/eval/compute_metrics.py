"""
Anomaly-detection evaluation metrics for SpecMAE.

Computes:
    AUC-ROC   — area under the receiver-operating curve
    pAUC      — partial AUC for FPR in [0, max_fpr] (DCASE convention: max_fpr=0.1)
    F1 @ best threshold — maximised over the Youden index
    Precision / Recall / FPR @ best threshold

All metrics can be computed per-SNR or over the full test set.

Standalone usage
----------------
    python SpecMae/scripts/eval/compute_metrics.py \
        --scores_csv results/train_desert/anomaly_scores.csv

or imported as a module:
    from SpecMae.scripts.eval.compute_metrics import compute_all_metrics, load_scores_csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── optional sklearn dependency (graceful fallback for pure-numpy AUC) ──────
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════
#  Core metric functions
# ═══════════════════════════════════════════════════════════════════════════

def _trapezoidal_auc(fprs: np.ndarray, tprs: np.ndarray) -> float:
    """Trapezoidal rule AUC (sorted by fpr ascending)."""
    order = np.argsort(fprs)
    return float(np.trapezoid(tprs[order], fprs[order]))


def compute_roc(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (fprs, tprs, thresholds) with sklearn when available,
    otherwise a pure-numpy implementation.

    Labels: 0 = normal, 1 = anomaly.
    Scores: higher → more anomalous.
    """
    if _SKLEARN:
        fprs, tprs, thresholds = roc_curve(labels, scores)
        return fprs, tprs, thresholds

    # Pure-numpy fallback
    thresholds = np.unique(scores)[::-1]   # descending
    fprs, tprs = [1.0], [1.0]
    neg = (labels == 0).sum()
    pos = (labels == 1).sum()
    for t in thresholds:
        pred   = (scores >= t).astype(int)
        tp     = ((pred == 1) & (labels == 1)).sum()
        fp     = ((pred == 1) & (labels == 0)).sum()
        tprs.append(tp / max(pos, 1))
        fprs.append(fp / max(neg, 1))
    fprs.append(0.0)
    tprs.append(0.0)
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    order    = np.argsort(fprs_arr)
    return fprs_arr[order], tprs_arr[order], thresholds


def compute_pauc(
    fprs:    np.ndarray,
    tprs:    np.ndarray,
    max_fpr: float = 0.1,
) -> float:
    """
    Partial AUC (pAUC) for FPR ≤ max_fpr, normalised to [0, 1].

    Follows the DCASE 2023 Task 2 convention (max_fpr = 0.1).
    """
    mask     = fprs <= max_fpr
    if mask.sum() < 2:
        return float("nan")
    pauc_raw = float(np.trapezoid(tprs[mask], fprs[mask]))
    return pauc_raw / max_fpr      # normalise to [0, 1]


def best_f1_threshold(
    labels:     np.ndarray,
    scores:     np.ndarray,
    fprs:       np.ndarray,
    tprs:       np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    """
    Find the threshold that maximises the Youden index (TPR - FPR),
    then compute Precision, Recall, F1, FPR at that threshold.
    """
    youden   = tprs - fprs
    # thresholds may be 1 shorter than fprs/tprs (sklearn convention)
    n        = min(len(thresholds), len(youden))
    best_idx = int(np.argmax(youden[:n]))
    thr      = float(thresholds[best_idx])

    pred      = (scores >= thr).astype(int)
    tp        = int(((pred == 1) & (labels == 1)).sum())
    fp        = int(((pred == 1) & (labels == 0)).sum())
    fn        = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "threshold": thr,
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "fpr":       float(fprs[best_idx]),
        "tpr":       float(tprs[best_idx]),
    }


def compute_all_metrics(
    labels:  np.ndarray,
    scores:  np.ndarray,
    max_fpr: float = 0.1,
) -> dict[str, float]:
    """
    Full metric suite for one (label, score) array pair.

    Returns a flat dict with keys:
        auc, pauc, f1, threshold, precision, recall, fpr_at_best, tpr_at_best,
        n_normal, n_anomaly
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(labels)) < 2:
        return {"error": "Only one class present — cannot compute AUC."}

    # AUC via sklearn if available
    if _SKLEARN:
        auc = float(roc_auc_score(labels, scores))
    fprs, tprs, thresholds = compute_roc(labels, scores)
    if not _SKLEARN:
        auc = _trapezoidal_auc(fprs, tprs)

    pauc   = compute_pauc(fprs, tprs, max_fpr)
    f1_d   = best_f1_threshold(labels, scores, fprs, tprs, thresholds)

    return {
        "auc":          auc,
        "pauc":         pauc,
        "f1":           f1_d["f1"],
        "threshold":    f1_d["threshold"],
        "precision":    f1_d["precision"],
        "recall":       f1_d["recall"],
        "fpr_at_best":  f1_d["fpr"],
        "tpr_at_best":  f1_d["tpr"],
        "n_normal":     int((labels == 0).sum()),
        "n_anomaly":    int((labels == 1).sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Per-SNR breakdown
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics_per_snr(
    labels:  np.ndarray,
    scores:  np.ndarray,
    snr_tags: list[str],
    max_fpr: float = 0.1,
) -> dict[str, dict]:
    """
    Compute metrics for each unique SNR tag plus the overall set.

    Args:
        labels:   (N,) int array — 0=normal, 1=anomaly
        scores:   (N,) float array — higher = more anomalous
        snr_tags: (N,) list of strings, e.g. ["normal", "snr_-10dB", ...]

    Returns:
        dict keyed by SNR tag + "overall"
    """
    snr_tags_arr = np.array(snr_tags)
    results: dict[str, dict] = {}

    # Overall
    results["overall"] = compute_all_metrics(labels, scores, max_fpr)

    # Per-SNR (anomaly vs normal pool)
    anomaly_mask  = labels == 1
    normal_labels = labels[~anomaly_mask]
    normal_scores = scores[~anomaly_mask]

    for tag in sorted(set(snr_tags)):
        if tag == "normal":
            continue
        tag_mask  = (snr_tags_arr == tag) & anomaly_mask
        if tag_mask.sum() == 0:
            continue
        combined_labels = np.concatenate([normal_labels, labels[tag_mask]])
        combined_scores = np.concatenate([normal_scores, scores[tag_mask]])
        results[tag] = compute_all_metrics(combined_labels, combined_scores, max_fpr)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  CSV helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_scores_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a scores CSV with columns: label, score, snr_tag.

    Returns (labels, scores, snr_tags).
    """
    labels, scores, snr_tags = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
            snr_tags.append(row.get("snr_tag", "unknown"))
    return np.array(labels), np.array(scores), snr_tags


def print_metrics_table(results: dict[str, dict]) -> None:
    """Pretty-print metrics across SNR groups."""
    header = f"{'SNR tag':<20}  {'AUC':>6}  {'pAUC':>6}  {'F1':>6}  {'N_anom':>7}  {'N_norm':>7}"
    print(header)
    print("-" * len(header))
    for tag in ["overall"] + [k for k in sorted(results) if k != "overall"]:
        m = results[tag]
        if "error" in m:
            print(f"  {tag:<18}  {m['error']}")
            continue
        print(
            f"  {tag:<18}  "
            f"{m['auc']:>6.4f}  "
            f"{m['pauc']:>6.4f}  "
            f"{m['f1']:>6.4f}  "
            f"{m['n_anomaly']:>7d}  "
            f"{m['n_normal']:>7d}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpecMAE anomaly-detection metrics (AUC, pAUC, F1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scores_csv", required=True,
                   help="CSV with columns: label, score, snr_tag")
    p.add_argument("--out_json",   default=None,
                   help="Optional path to write metrics as JSON")
    p.add_argument("--max_fpr",    type=float, default=0.1,
                   help="FPR upper bound for pAUC (DCASE convention: 0.1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.scores_csv)
    if not path.exists():
        print(f"ERROR: scores CSV not found: {path}", file=sys.stderr)
        sys.exit(1)

    labels, scores, snr_tags = load_scores_csv(path)
    results = compute_metrics_per_snr(labels, scores, snr_tags, max_fpr=args.max_fpr)

    print_metrics_table(results)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {args.out_json}")


if __name__ == "__main__":
    main()
