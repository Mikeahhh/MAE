# V²SDLS: UAV-enabled Victim Sound Detection and Localization System

A two-stage acoustic sensing system for UAV Search and Rescue (SAR). Uses a Masked Autoencoder (MAE) for anomaly-based victim sound detection (Sentinel stage) and GCC-PHAT TDOA triangulation for 3D localization (Responder stage).

Target venue: **IEEE SPAWC** (Signal Processing Advances in Wireless Communications).

---

## System Architecture

```
Stage 1: Sentinel (MAE-based Anomaly Detection)
  Single-channel audio → Log-Mel Spectrogram → Masked Autoencoder
    → Reconstruction Error (Top-K scoring) → D > Dth? → Trigger

Stage 2: Responder (Multi-Channel Localization)
  9-channel UCA audio → GCC-PHAT TDOA → DOA Estimation
    → Sliding-Window Triangulation → 3D Victim Position
```

The MAE is trained on **background noise only** (drone + ambient). At inference, victim sound (child cry / male rescue call) causes high reconstruction error because it is anomalous to the learned noise manifold.

---

## Performance (100 Monte Carlo simulations)

| Parameter | Desert | Forest |
|-----------|--------|--------|
| Flight height | h = 5 m | h = 15 m |
| Optimal mask ratio ρ | 0.10 | 0.10 |
| Detection threshold D_th | 1.57 | 1.33 |
| Source SPL | 120 dB @ 1m | 120 dB @ 1m |
| Drone noise | 75 dB | 75 dB |
| Path loss exponent α | 2.0 | 2.5 |
| Ambient noise | 25 dB | 35 dB |

Detection accuracy is computed as the percentage of test clips where the MAE reconstruction error exceeds D_th, averaged over 100 Monte Carlo simulations with 100 stochastic masked reconstructions per clip.

Localization accuracy is the Euclidean distance between the estimated and true victim position, computed via sliding-window triangulation from multiple DOA observations along the flight trajectory.

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Microphone array | 9-element UCA (M=9) |
| Anomaly scoring | Top-K (K=30% worst patches) |
| MC reconstruction passes | 100 |
| Training epochs | 60 |
| Optimizer | AdamW |
| Train/Val split | 90% / 10% |
| Flight path | L-shaped: Y-leg (-4→5m) + X-leg (-500→200m) |
| Sliding window (triangulation) | 10 observations |
| Detection heights (desert) | 5, 10, 15, 20 m |
| Detection heights (forest) | 15, 20, 35, 50 m |

---

## Project Structure

```
SpecMae/
├── models/specmae/              # ViT-Base Masked Autoencoder
│   ├── specmae_model.py         #   MAE model + anomaly scoring
│   ├── encoder.py               #   12-layer ViT encoder with random masking
│   ├── decoder.py               #   4-layer lightweight decoder
│   ├── patch_embed.py           #   16x16 patch embedding
│   └── pos_embed.py             #   2D sinusoidal positional embeddings
│
├── scripts/
│   ├── train/
│   │   ├── train_mask_ratio_sweep.py   # Train models across mask ratios
│   │   └── train_freq_experiment.py    # Frequency band experiments
│   │
│   ├── eval/
│   │   ├── eval_height_sweep.py        # Height-based MC evaluation
│   │   ├── eval_detection_timing.py    # Sliding-window anomaly detection
│   │   ├── plot_3d_snr_flyover.py      # 3D flyover + SNR + localization figure
│   │   ├── plot_height_detection.py    # Detection vs mask_ratio per height
│   │   ├── plot_mask_ratio_detection.py # Mask ratio selection figure
│   │   └── plot_recon_distribution.py  # Reconstruction error distribution
│   │
│   └── utils/
│       ├── generate_training_data.py   # Physics-based training data synthesis
│       ├── generate_long_test_audio.py # 12s test clips with voice onset
│       ├── feature_extraction.py       # LogMelExtractor (48kHz)
│       ├── data_loader.py             # PyTorch dataset
│       └── mix_audio.py               # SNR-calibrated waveform mixing
│
├── simulation/
│   ├── core/
│   │   ├── propagation_model.py       # ISO 9613-1 acoustic propagation
│   │   ├── tdoa_estimator.py          # GCC-PHAT TDOA estimation
│   │   ├── doa_calculator.py          # DOA from TDOA
│   │   ├── triangulation.py           # Multi-point 3D triangulation
│   │   └── ring_buffer.py            # Continuous raw audio buffer
│   └── engine/
│       ├── flight_simulator.py        # SAR mission simulation
│       └── detector_bridge.py         # MAE ↔ simulation bridge
│
├── data/
│   ├── drone/                  # DJI propeller recording (48kHz, 133s)
│   ├── ambient/{desert,forest}/ # Environmental noise
│   └── human_voice/            # Child cry + Male rescue (11,182 clips, not included)
│
└── configs/                    # Simulation YAML configs
```

---

## Data Flow

```
1. Raw Audio Sources
   data/drone/dji_sound.wav            (DJI propeller noise, 48kHz)
   data/ambient/{desert,forest}/       (environmental noise recordings)
   data/human_voice/                   (child cry + male rescue, 11,182 clips)

2. Physics-Based Data Generation       [generate_training_data.py]
   PropagationModel (ISO 9613-1) computes attenuation at each distance
   Training data = drone + ambient ONLY (no victim sound)
   Test data = drone + ambient + victim sound at physics-derived SNR

3. Training                            [train_mask_ratio_sweep.py]
   Normal-only training → MAE learns to reconstruct background noise
   60 epochs, AdamW optimizer, 90/10 split

4. Evaluation                          [eval_height_sweep.py]
   100 clips × 100 MC passes = 10,000 evaluations per condition
   Anomaly score = Top-30% patch MSE over 100 random masks
   Detection: score > Dth (desert=1.57, forest=1.33)

5. Flyover Simulation                  [plot_3d_snr_flyover.py]
   L-shaped flight path, physics-based SNR at each position
   Detection → GCC-PHAT TDOA → DOA → Sliding-window triangulation
```

---

## CLI Execution

All commands run from the parent directory of `SpecMae/`:

### Generate Training Data
```bash
python3 -m SpecMae.scripts.utils.generate_training_data
```

### Train Models
```bash
# Train across mask ratios for both scenarios
python3 -m SpecMae.scripts.train.train_mask_ratio_sweep --force

# Single scenario
python3 -m SpecMae.scripts.train.train_mask_ratio_sweep --scenario desert
```

### Evaluation
```bash
# Full 100-MC height sweep
python3 -m SpecMae.scripts.eval.eval_height_sweep --all --n_clips 100 --n_passes 100

# Flyover figure (paper-grade, ~2h)
python3 -m SpecMae.scripts.eval.plot_3d_snr_flyover \
    --combined --with_error --real --n_mc 100 --n_passes 100

# Quick verification (~5 min)
python3 -m SpecMae.scripts.eval.plot_3d_snr_flyover \
    --combined --with_error --real --n_mc 5 --n_passes 10

# Instant re-plot from cached data (no simulation)
python3 -m SpecMae.scripts.eval.plot_3d_snr_flyover \
    --combined --with_error --from-mat
```

---

## Quick Start (Verify Pipeline in ~30 min)

```bash
# 1. Generate data (5 min)
python3 -m SpecMae.scripts.utils.generate_training_data

# 2. Train one model (5 min, reduced epochs)
python3 -m SpecMae.scripts.train.train_mask_ratio_sweep \
    --scenario desert --mask_ratios 0.10 --epochs 30

# 3. Quick flyover eval (5 min)
python3 -m SpecMae.scripts.eval.plot_3d_snr_flyover \
    --terrain desert --with_error --real --n_mc 3 --n_passes 5
```

---

## Physics Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| SOURCE_SPL | 120 dB @ 1m | Human scream (literature) |
| DRONE_NOISE_SPL | 75 dB | DJI propeller at microphone |
| Horizontal offset | 5 m | UAV-to-person closest approach |
| Path loss model | Log-distance + ISO 9613-1 | Free-field acoustic propagation |
| Desert | α=2.0, ambient=25 dB, no canopy | Open terrain |
| Forest | α=2.5, ambient=35 dB, canopy=0.02 dB/m | Dense vegetation |
| Mic height | = UAV flight height | No suspension cable |

---

## Model Architecture

- **Encoder**: ViT-Base (12 layers, 768 dim, 12 heads, patch_size=16)
- **Decoder**: 4-layer lightweight transformer
- **Input**: Log-Mel spectrogram (128 mel bins, 48kHz, 1s clips)
- **Masking**: Random patch masking at ratio ρ
- **Scoring**: Top-K MSE of worst 30% patches over 100 MC random masks

---

## Raw Audio Data

**Included in repository:**
- `data/drone/dji_sound.wav` (25 MB, 48kHz mono, DJI propeller)
- `data/ambient/desert/` (10 MB, 3 files from Freesound.org)
  - [Sound 645305 — desert wind](https://freesound.org/people/DarkShroom/sounds/645305/)
  - [Sound 402710 — desert wind stereo](https://freesound.org/people/KasDonatov/sounds/402710/)
  - [Sound 383243 — wind](https://freesound.org/people/beautifuldaymonster1968/sounds/383243/)
- `data/ambient/forest/` (23 MB, 7 files from [Pixabay forest sound effects](https://pixabay.com/sound-effects/search/forest/))

**Not included (obtain separately):**
- `data/human_voice/Child_Cry_400_600Hz/` (8,639 clips)
- `data/human_voice/Male_Rescue_100_300Hz/` (2,543 clips)
- Source: [ASVP-ESD corpus on Kaggle](https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances). Total: 11,182 clips.

---

## Requirements

```bash
pip install -r requirements.txt
# Core: torch>=2.0, librosa>=0.10, numpy, matplotlib, scipy, soundfile, scikit-learn, tqdm
```

Hardware: GPU with 8GB+ VRAM (NVIDIA CUDA or Apple M-series MPS), 16GB+ RAM.
