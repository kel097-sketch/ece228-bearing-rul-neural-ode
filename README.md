# Bearing RUL Prediction with Neural ODE Models

This repository contains the code and experiment results for bearing remaining useful life (RUL) prediction. The project compares classical machine learning, deep sequence models, and Neural ODE-based models under different train/test settings.

## Models

- Ridge regression
- LSTM
- TCN
- Transformer
- Latent ODE

## Methodology Additions

The current version adds a stricter evaluation layer inspired by recent bearing RUL papers on wavelet feature fusion, attention-based RUL estimation, uncertainty-aware prediction, and small-sample robustness:

- Wavelet packet and DWT features using `db4` decomposition, alongside the original time/frequency features.
- Feature scoring with correlation, monotonicity, and robustness, plus feature-group ablation.
- Cross-condition uncertainty estimates using validation-calibrated conformal intervals.
- Robustness checks under sparse observation and missing-feature perturbations.
- Sequence length sensitivity and multi-seed stability checks for stronger reproducibility.

## Project Structure

```text
src/                 Source code for preprocessing, training, evaluation, and plotting
processed/           Processed features and train/test split files
results/tables/      Evaluation result tables
results/figures/     Generated plots and visualizations
data/                Raw dataset folder, not included in this repository
```

## Setup

This project uses `uv` for dependency management.

```bash
uv sync
```

## Data Setup

The raw XJTU-SY Bearing Dataset is not included in this repository because it is large. To reproduce the full pipeline, download the dataset from one of the official links:

- Dataset page: https://biaowang.tech/xjtu-sy-bearing-datasets/
- Dataset GitHub page: https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets

After downloading, unzip the dataset under `data/` so the folder structure looks like this:

```text
data/
  XJTU-SY_Bearing_Datasets/
    35Hz12kN/
      Bearing1_1/
        1.csv
        2.csv
        ...
    37.5Hz11kN/
      Bearing2_1/
        1.csv
        2.csv
        ...
    40Hz10kN/
      Bearing3_1/
        1.csv
        2.csv
        ...
```

The code searches recursively under `data/`, so the important part is keeping the condition folders named `35Hz12kN`, `37.5Hz11kN`, and `40Hz10kN`.

## Run Experiments

Run the full experiment pipeline:

```bash
uv run python src/run_all.py
```

Run the stricter analysis layer after the main models are available:

```bash
uv run python src/run_all.py --skip_features --skip_ml --skip_deep --run_advanced
```

If the raw dataset is not available, the committed feature file can still be used to rerun the models:

```bash
uv run python src/run_all.py --skip_features
```

For a faster check that skips deep learning models:

```bash
uv run python src/run_all.py --skip_features --skip_deep
```

Generate plots from existing results:

```bash
uv run python src/plot_results.py --all
```

Key standalone advanced commands:

```bash
uv run python src/extract_features.py --metadata processed/metadata.csv --out processed/features.csv
uv run python src/extract_features.py --metadata processed/metadata.csv --out processed/features_wavelet.csv --include_wavelet
uv run python src/feature_analysis.py --features processed/features_wavelet.csv --split_dir processed/splits
uv run python src/conformal_uncertainty.py --seq_dir processed/sequences --features processed/features.csv
uv run python src/missing_feature_robustness.py --seq_dir processed/sequences
uv run python src/sparse_observation.py --seq_dir processed/sequences --keep_ratios 0.3
uv run python src/k_sensitivity.py --features processed/features.csv --split_dir processed/splits --k_values 5 10 20 --models TCN Transformer latent_ode --epochs 30 --patience 5
uv run python src/multiseed_experiment.py --seq_dir processed/sequences --models TCN latent_ode --seeds 42 43 44 --epochs 30 --patience 5
uv run python src/feature_setting_retraining.py --protocol cross_condition --top_k 30 --k 10 --epochs 50 --patience 8 --rebuild_sequences
```

## Results Summary

Metrics are reported on normalized RUL, where lower MAE and RMSE indicate better prediction performance.

- In the final cross-condition setting, TCN achieved the best average MAE of 0.230.
- Transformer was very close in cross-condition MAE at 0.232, while Latent ODE reached 0.238.
- In the final mixed-condition setting, Transformer achieved the best average MAE of 0.218.
- Mixed-condition results are reported only as an in-distribution reference; the main conclusion uses cross-condition splits.

Overall, sequence-based models performed strongly in most settings, while the simpler Ridge baseline remained competitive in the mixed-condition experiment. The results suggest that generalization across operating conditions is still challenging for bearing RUL prediction.

## Strict Evaluation Results

Additional cross-condition analyses make the evaluation more paper-like:

- Wavelet-only features improved the Ridge cross-condition MAE from 0.316 to 0.258, while using all expanded features without selection degraded performance, confirming the need for feature selection or regularization.
- Validation-calibrated conformal intervals provide coverage-width reliability information under distribution shift.
- Sparse observation robustness now includes 30% observed samples. Transformer, TCN, Latent ODE, and LSTM all remain in roughly the 0.215-0.242 MAE range at 30% observation.
- Missing-feature robustness was evaluated at 0%, 10%, 20%, and 30% random test-time missing features with train-mean imputation.
- K sensitivity under compact retraining showed K=20 produced the best average MAE among tested sequence lengths for TCN, Transformer, and Latent ODE.
- Multi-seed compact retraining on cross-condition splits gave TCN average MAE 0.222 with std 0.048, and Latent ODE average MAE 0.240 with std 0.038.
- Complete feature-setting retraining was run for 4 feature settings and 5 active models under cross-condition splits as a supplementary comparison. Wavelet-only features improved several models, including Ridge (Original MAE 0.316 to Wavelet-only MAE 0.258). Selected-top30 also improved LSTM in this compact retraining setting, but it is treated as supplementary evidence rather than the main model conclusion.

The conformal intervals are calibrated on the existing validation split. This is useful for uncertainty reporting, but a fully independent calibration split would be needed for strict split-conformal claims.

A Chinese write-up for the stricter experiment package is available at `docs/strict_experiment_report.md`.

## Outputs

Main outputs are saved under:

```text
results/tables/
results/figures/
```

Important advanced outputs:

```text
results/tables/feature_scores.csv
results/tables/feature_group_sensitivity_average_results.csv
results/tables/conformal_interval_average_results.csv
results/tables/missing_feature_robustness_average_results.csv
results/tables/sparse_observation_average_results.csv
results/tables/k_sensitivity_average_results.csv
results/tables/multiseed_model_average_results.csv
results/tables/selected_feature_retraining_average_results.csv
results/tables/selected_feature_retraining_mae_pivot.csv
results/tables/selected_feature_retraining_selected_features.csv
results/figures/bar_charts/selected_feature_retraining_mae_heatmap.png
results/figures/uncertainty/
results/figures/robustness/
```

Large files such as raw data, generated sequences, model checkpoints, latent outputs, and prediction files are excluded from GitHub using `.gitignore`.
