# Bearing RUL Prediction under Operating Condition Shift

This repository contains code and final report assets for bearing Remaining Useful Life (RUL) prediction on the XJTU-SY rolling bearing dataset.

The project studies whether vibration based representations, especially wavelet features, improve RUL prediction when the test operating condition is unseen during training.

## Included

- `src/`: feature extraction, split generation, training, and evaluation scripts
- `RUL_XJTU_NeurIPS_Final/`: final LaTeX report, figures, references, and style file
- `results/final_tables/`: selected CSV results used in the final report

The raw dataset, processed sequences, checkpoints, slide decks, and large intermediate outputs are not included.

## Dataset

Download the XJTU-SY bearing dataset from:

https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets

Expected local layout:

```text
data/
  XJTU-SY_Bearing_Datasets/
    35Hz12kN/
    37.5Hz11kN/
    40Hz10kN/
```

## Setup

```bash
uv sync
```

## Main Protocol

The main experiment uses leave one condition out cross condition evaluation:

- train on C1 and C2, test on C3
- train on C1 and C3, test on C2
- train on C2 and C3, test on C1

All splits are performed at the bearing trajectory level.

## Models and Inputs

Models:

- Ridge
- LSTM
- TCN
- Transformer
- Latent ODE

Input representations:

- raw signal
- time domain features
- frequency domain features
- wavelet domain features

## Useful Scripts

```bash
uv run python src/build_metadata.py
uv run python src/extract_features.py
uv run python src/extract_raw_downsample_features.py
uv run python src/strict_method_v2.py
uv run python src/tune_wavelet_k40.py
uv run python src/fixed_wavelet_params_feature_settings_k40.py
```

## Final Report

The final LaTeX report is:

```text
RUL_XJTU_NeurIPS_Final/final.tex
```
