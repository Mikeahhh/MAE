#!/bin/bash
# SpecMae Full Pipeline — DJI-based training + evaluation
# Expected total time: ~20-24h on Mac MPS (paper-grade 100-MC)
#
# Usage:
#   cd /Volumes/MIKE2T
#   nohup bash SpecMae/run_pipeline.sh > SpecMae/pipeline.log 2>&1 &

set -e
PYTHON=${SPECMAE_PYTHON:-python3}
cd /Volumes/MIKE2T

echo "=========================================="
echo "  SpecMae Pipeline"
echo "  Started: $(date)"
echo "=========================================="

# ── Step 0: Generate training data (~5 min) ────────────────────
echo ""
echo "[Step 0/5] Generating training data (DJI drone noise)..."
$PYTHON -m SpecMae.scripts.utils.generate_training_data
echo "Data generation complete: $(date)"

# ── Step 1: Train all 34 models (~8-10h MPS) ──────────────────
echo ""
echo "[Step 1/5] Training 17 mask_ratios x 2 scenarios..."
$PYTHON -m SpecMae.scripts.train.train_mask_ratio_sweep --force
echo "Training complete: $(date)"

# ── Step 2: Quick diagnostic (~5 min) ─────────────────────────
echo ""
echo "[Step 2/5] Running reconstruction error diagnostic..."
$PYTHON -m SpecMae.scripts.eval.plot_recon_distribution --mr 0.10 --scenario desert
$PYTHON -m SpecMae.scripts.eval.plot_recon_distribution --mr 0.60 --scenario forest
echo "Diagnostic complete: $(date)"

# ── Step 3: Height sweep evaluation (100-MC, ~14h) ────────────
echo ""
echo "[Step 3/5] Running height sweep (100 clips, 100 MC passes)..."
$PYTHON -m SpecMae.scripts.eval.eval_height_sweep --all --n_clips 100 --n_passes 100
echo "Height sweep complete: $(date)"

# ── Step 4: Generate figures ───────────────────────────────────
echo ""
echo "[Step 4/5] Generating figures..."
$PYTHON -m SpecMae.scripts.eval.plot_height_detection --metric presence_accuracy
$PYTHON -m SpecMae.scripts.eval.plot_height_detection --metric detection_accuracy
$PYTHON -m SpecMae.scripts.eval.plot_mask_ratio_detection
echo "Figures complete: $(date)"

echo ""
echo "=========================================="
echo "  Pipeline DONE: $(date)"
echo "=========================================="
