# Bearing RUL Prediction with Neural ODE Models

This repository contains the code and experiment results for bearing remaining useful life (RUL) prediction. The project compares classical machine learning, deep sequence models, and Neural ODE-based models under different train/test settings.

## Models

- Ridge regression
- LSTM
- TCN
- Transformer
- Latent ODE
- Condition-aware Neural ODE

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

## Results Summary

Metrics are reported on normalized RUL, where lower MAE and RMSE indicate better prediction performance.

- In the within-condition setting, TCN achieved the best average MAE of 0.246.
- In the cross-condition setting, TCN also achieved the best average MAE of 0.227.
- In the literature-aligned leave-one-bearing-out setting, Transformer and LSTM performed best, with average MAE around 0.260.
- In the mixed-condition setting, Ridge regression achieved the best MAE of 0.199.
- The C3 within-condition split was the most difficult case, with the best MAE around 0.503.

Overall, sequence-based models performed strongly in most settings, while the simpler Ridge baseline remained competitive in the mixed-condition experiment. The results suggest that generalization across operating conditions is still challenging for bearing RUL prediction.

## Outputs

Main outputs are saved under:

```text
results/tables/
results/figures/
```

Large files such as raw data, generated sequences, model checkpoints, latent outputs, and prediction files are excluded from GitHub using `.gitignore`.
