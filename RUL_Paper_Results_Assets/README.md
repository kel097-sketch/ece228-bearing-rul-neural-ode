# RUL Paper Results Assets

This folder contains the figures, tables, and LaTeX section draft for the paper Results and Analysis section.

## Main LaTeX file

- `results_and_analysis_section.tex`

This file can be pasted into the paper after the Method section.

## Figures

Important figures for the Results section:

- `figures/ridge_feature_ablation_bar.png`  
  Feature setting ablation with Ridge regression.

- `figures/k_selection_v2_accuracy_coverage.png`  
  Sequence length selection. This supports choosing `K=40`.

- `figures/tuned_wavelet_k40_cross_mae_bar.png`  
  Main cross condition MAE comparison after hyperparameter tuning.

- `figures/tuned_wavelet_k40_per_split_heatmap.png`  
  Per split cross condition MAE heatmap.

- `figures/tuned_wavelet_k40_cross_table.png`  
  Rendered table version of the tuned cross condition results.

Optional qualitative figure:

- `figures/tuned_representative_prediction_trajectory_no_target.png`  
  Representative prediction trajectories on held out C3. Use as qualitative support, not as the main result.

Method or dataset support figures:

- `figures/pipeline.png`
- `figures/example_vibration_signals.png`

## Tables

- `tables/tuned_wavelet_k40_protocol_summary.csv`
- `tables/tuned_wavelet_k40_split_summary.csv`
- `tables/ridge_feature_ablation_average.csv`
- `tables/k_selection_recommendation_v2.csv`

## Suggested paper figure priority

1. Main result table in LaTeX.
2. Per split cross condition heatmap.
3. Feature setting ablation.
4. Sequence length selection.
5. Optional representative trajectory.

