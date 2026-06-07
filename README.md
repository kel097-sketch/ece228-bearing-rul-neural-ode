# Bearing RUL Prediction under Operating Condition Shift

This repository contains code, experiment results, and report assets for bearing Remaining Useful Life (RUL) prediction using the XJTU-SY rolling bearing dataset.

The project studies whether vibration-derived features, especially wavelet features, improve RUL prediction when the test operating condition is unseen during training.

## Overview

We compare feature-based and sequence-based models for normalized RUL prediction:

- Ridge regression
- LSTM
- TCN
- Transformer
- Latent ODE

The main evaluation uses a leave-one-condition-out cross-condition protocol:

- Train on C1 + C2, test on C3
- Train on C1 + C3, test on C2
- Train on C2 + C3, test on C1

All splits are performed at the bearing trajectory level.

## Dataset

The raw XJTU-SY dataset is not included in this repository.

Download it from:

- https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets

Expected structure:

```text
data/
  XJTU-SY_Bearing_Datasets/
    35Hz12kN/
    37.5Hz11kN/
    40Hz10kN/
```

## Feature Settings

The experiments use vibration features extracted from horizontal and vertical acceleration channels.

| Feature setting | Description |
|---|---|
| Raw signal | Downsampled horizontal and vertical vibration signals |
| Time | Time-domain statistical features |
| Frequency | Frequency-domain spectral features |
| Original | Time + frequency features |
| Wavelet-only | db4 WPT and DWT wavelet features |
| All-expanded | Original + wavelet features |
| Top-k | Training-only selected features from all-expanded features |

## Main Results

The final report focuses on cross-condition prediction.

Key findings:

- Cross-condition RUL prediction is harder than random window-level evaluation.
- Wavelet features improve point prediction accuracy.
- TCN achieves the best MAE and RMSE in the main wavelet setting.
- Compact feature selection reduces redundancy, but more selected features do not always improve performance.

## Repository Structure

```text
src/                         Source code
results/tables/              Result tables
RUL_XJTU_NeurIPS_Final/      Final report source and figures
RUL_Paper_Results_Assets/    Curated result figures and tables
data/                        Raw dataset folder, not included
processed/                   Generated features and sequences, mostly not tracked
```

## Setup

This project uses `uv` for dependency management.

```bash
uv sync
```

## Reproduce

Generate wavelet features:

```bash
uv run python src/extract_features.py --metadata processed/metadata.csv --out processed/features_wavelet.csv --include_wavelet
```

Run the main tuned wavelet cross-condition experiment:

```bash
uv run python src/tune_wavelet_k40.py
```

Run feature ablation:

```bash
uv run python src/feature_analysis.py --features processed/features_wavelet.csv --split_dir processed/splits
```

## Notes

Large files such as raw data, generated sequences, model checkpoints, and prediction files are excluded from GitHub.
