import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import METADATA_COLUMNS
from make_sequences import make_sequence_file
from strict_wavelet_experiment import (
    aggregate_bearing_metrics,
    per_bearing_metrics,
    prediction_frame,
    summarize_final,
)
from tune_wavelet_k40 import (
    MODEL_COLORS,
    MODEL_ORDER,
    cross_split_paths,
    load_sequence,
    model_grid,
    split_df,
    train_one_config,
)
from utils import ensure_dir, get_feature_columns, load_split, set_seed


SETTING_LABELS = {
    "raw_waveform": "Raw waveform",
    "original": "Original",
    "wavelet_only": "Wavelet-only",
    "selected_top10": "Selected-top10",
    "selected_top20": "Selected-top20",
    "selected_top30": "Selected-top30",
    "selected_top60": "Selected-top60",
    "selected_top90": "Selected-top90",
}


def fixed_model_grid() -> dict[str, list[dict]]:
    return {
        "LSTM": [
            {"hidden_dim": 64, "num_layers": 1, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
        ],
        "TCN": [
            {"hidden_dim": 64, "levels": 2, "kernel_size": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4},
        ],
        "Transformer": [
            {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1, "lr": 5e-4, "weight_decay": 1e-4},
        ],
        "latent_ode": [
            {"latent_dim": 8, "lr": 1e-3, "weight_decay": 1e-4, "smooth_weight": 1e-4},
        ],
    }


def args_for_model(args, model_name: str):
    if model_name != "latent_ode":
        return args
    model_args = copy.copy(args)
    model_args.epochs = args.latent_ode_epochs
    model_args.patience = args.latent_ode_patience
    return model_args


def feature_subset(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    meta_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:5]}")
    return df[meta_cols + feature_cols].copy()


def selected_features_for_split(split_name: str, selected_dir: Path, top_k: int) -> list[str]:
    path = selected_dir / f"{split_name}_selected_top{top_k}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Selected feature file not found: {path}")
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        raise ValueError(f"Selected feature file has no 'feature' column: {path}")
    features = frame["feature"].head(top_k).astype(str).tolist()
    if len(features) != top_k:
        raise ValueError(f"Expected {top_k} features in {path}, found {len(features)}")
    return features


def load_setting_features(setting: str, split: dict, args) -> tuple[pd.DataFrame, list[str]]:
    if setting == "raw_waveform":
        df = pd.read_csv(args.raw_features)
        cols = get_feature_columns(df)
        return feature_subset(df, cols), cols
    if setting == "original":
        df = pd.read_csv(args.original_features)
        cols = get_feature_columns(df)
        return feature_subset(df, cols), cols
    if setting == "wavelet_only":
        df = pd.read_csv(args.wavelet_features)
        cols = get_feature_columns(df)
        return feature_subset(df, cols), cols
    if setting.startswith("selected_top"):
        top_k = int(setting.replace("selected_top", ""))
        df = pd.read_csv(args.expanded_features)
        cols = selected_features_for_split(split["split_name"], Path(args.selected_feature_dir), top_k)
        return feature_subset(df, cols), cols
    raise ValueError(setting)


def train_ridge(
    features_df: pd.DataFrame,
    split: dict,
    feature_cols: list[str],
    setting: str,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = split_df(features_df, split["train_bearings"])
    val_df = split_df(features_df, split["val_bearings"])
    test_df = split_df(features_df, split["test_bearings"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_val = val_df["normalized_rul"].to_numpy(dtype=np.float32)

    best = None
    trial_rows = []
    for alpha in args.ridge_alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred_val = np.clip(model.predict(X_val), 0.0, 1.0)
        val_metrics = aggregate_bearing_metrics(
            per_bearing_metrics(
                val_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object),
                y_val,
                pred_val,
                args.epsilon,
            )
        )
        trial_rows.append({"alpha": alpha, **val_metrics})
        if best is None or val_metrics["mae"] < best["mae"]:
            best = {"alpha": alpha, "model": model, **val_metrics}

    y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)
    pred_test = np.clip(best["model"].predict(X_test), 0.0, 1.0)
    test_meta = test_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object)
    pred = prediction_frame(test_meta, y_test, pred_test, split, "Ridge", setting, None)
    per_bearing = per_bearing_metrics(test_meta, y_test, pred_test, args.epsilon)
    per_bearing["model"] = "Ridge"
    per_bearing["feature_setting"] = setting
    per_bearing["K"] = "NA"
    per_bearing["protocol"] = split["protocol"]
    per_bearing["split_name"] = split["split_name"]
    per_bearing["best_params"] = json.dumps({"alpha": best["alpha"]}, sort_keys=True)
    per_bearing["num_features"] = len(feature_cols)
    return pred, per_bearing, {"best": best, "trials": trial_rows}


def append_frame(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def result_exists(path: Path, setting: str, split_name: str, model: str) -> bool:
    if not path.exists():
        return False
    existing = pd.read_csv(path, usecols=["feature_setting", "split_name", "model"])
    return (
        (existing["feature_setting"] == setting)
        & (existing["split_name"] == split_name)
        & (existing["model"] == model)
    ).any()


def run(args) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    pred_dir = ensure_dir(args.pred_dir)
    seq_root = ensure_dir(args.seq_dir)
    ckpt_dir = ensure_dir(args.ckpt_dir)
    split_paths = cross_split_paths(Path(args.split_dir))
    grid = fixed_model_grid() if args.fixed_grid else model_grid()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trials_path = out_dir / "three_feature_settings_k40_tuning_trials.csv"
    best_path = out_dir / "three_feature_settings_k40_best_configs.csv"
    per_bearing_path = out_dir / "three_feature_settings_k40_per_bearing_metrics.csv"
    predictions_path = out_dir / "three_feature_settings_k40_predictions.csv"

    if args.reset:
        for path in [trials_path, best_path, per_bearing_path, predictions_path]:
            if path.exists():
                path.unlink()

    for setting in args.feature_settings:
        for split_path in split_paths:
            split = load_split(split_path)
            split_name = split["split_name"]
            features_df, feature_cols = load_setting_features(setting, split, args)
            print(
                f"\n=== setting={setting} split={split_name} features={len(feature_cols)} device={device} ===",
                flush=True,
            )

            seq_dir = ensure_dir(seq_root / setting / f"k{args.k}")
            seq_path = seq_dir / f"{split_name}_k{args.k}.npz"
            if args.rebuild_sequences or not seq_path.exists():
                seq_path = make_sequence_file(features_df, split_path, args.k, seq_dir)
            data, _ = load_sequence(seq_path)

            if not result_exists(per_bearing_path, setting, split_name, "Ridge"):
                ridge_pred, ridge_metrics, ridge_info = train_ridge(features_df, split, feature_cols, setting, args)
                append_frame(predictions_path, ridge_pred)
                append_frame(per_bearing_path, ridge_metrics)
                ridge_trials = pd.DataFrame(
                    [
                        {
                            "feature_setting": setting,
                            "split_name": split_name,
                            "model": "Ridge",
                            "config_id": f"ridge_alpha_{trial['alpha']}",
                            "params": json.dumps({"alpha": trial["alpha"]}, sort_keys=True),
                            "val_mae": trial["mae"],
                            "val_rmse": trial["rmse"],
                            "val_spearman": trial["spearman"],
                            "best_epoch": None,
                            "num_features": len(feature_cols),
                        }
                        for trial in ridge_info["trials"]
                    ]
                )
                append_frame(trials_path, ridge_trials)
                append_frame(
                    best_path,
                    pd.DataFrame(
                        [
                            {
                                "feature_setting": setting,
                                "split_name": split_name,
                                "model": "Ridge",
                                "params": json.dumps({"alpha": ridge_info["best"]["alpha"]}, sort_keys=True),
                                "val_mae": ridge_info["best"]["mae"],
                                "val_rmse": ridge_info["best"]["rmse"],
                                "val_spearman": ridge_info["best"]["spearman"],
                                "best_epoch": None,
                                "num_features": len(feature_cols),
                            }
                        ]
                    ),
                )
                print(f"Ridge selected alpha={ridge_info['best']['alpha']}", flush=True)

            for model_name in ["LSTM", "TCN", "Transformer", "latent_ode"]:
                if result_exists(per_bearing_path, setting, split_name, model_name):
                    print(f"Skipping existing {setting} {split_name} {model_name}", flush=True)
                    continue
                best_result = None
                best_params = None
                best_config_id = None
                for config_id, params in enumerate(grid[model_name], start=1):
                    seed = args.seed + sum(ord(ch) for ch in f"{setting}:{split_name}:{model_name}:{config_id}")
                    print(
                        f"Tuning setting={setting} split={split_name} model={model_name} "
                        f"config={config_id}/{len(grid[model_name])}",
                        flush=True,
                    )
                    result = train_one_config(model_name, params, data, args_for_model(args, model_name), device, seed)
                    val = result["val_metrics"]
                    append_frame(
                        trials_path,
                        pd.DataFrame(
                            [
                                {
                                    "feature_setting": setting,
                                    "split_name": split_name,
                                    "model": model_name,
                                    "config_id": config_id,
                                    "params": json.dumps(params, sort_keys=True),
                                    "val_mae": val["mae"],
                                    "val_rmse": val["rmse"],
                                    "val_spearman": val["spearman"],
                                    "val_late_mae": val["late_mae"],
                                    "best_epoch": result["best_epoch"],
                                    "num_features": len(feature_cols),
                                }
                            ]
                        ),
                    )
                    if best_result is None or val["mae"] < best_result["val_metrics"]["mae"]:
                        best_result = result
                        best_params = copy.deepcopy(params)
                        best_config_id = config_id

                assert best_result is not None and best_params is not None
                print(
                    f"Selected {setting} {split_name} {model_name}: "
                    f"config={best_config_id}, val_MAE={best_result['val_metrics']['mae']:.4f}",
                    flush=True,
                )
                torch.save(
                    {
                        "model_name": model_name,
                        "feature_setting": setting,
                        "params": best_params,
                        "model_state_dict": best_result["model_state"],
                        "split": split,
                        "K": args.k,
                        "feature_names": data["feature_names"],
                        "best_epoch": best_result["best_epoch"],
                    },
                    ckpt_dir / f"{setting}_{split_name}_{model_name}_K{args.k}_tuned.pt",
                )
                test_metrics = best_result["test_per_bearing"].copy()
                test_metrics["model"] = model_name
                test_metrics["feature_setting"] = setting
                test_metrics["K"] = args.k
                test_metrics["protocol"] = split["protocol"]
                test_metrics["split_name"] = split_name
                test_metrics["best_params"] = json.dumps(best_params, sort_keys=True)
                test_metrics["num_features"] = len(feature_cols)
                append_frame(per_bearing_path, test_metrics)
                pred_frame = prediction_frame(
                    data["meta_test"],
                    data["y_test"],
                    best_result["test_pred"],
                    split,
                    model_name,
                    setting,
                    args.k,
                )
                pred_frame.to_csv(pred_dir / f"{setting}_{split_name}_{model_name}_K{args.k}_predictions.csv", index=False)
                append_frame(predictions_path, pred_frame)
                append_frame(
                    best_path,
                    pd.DataFrame(
                        [
                            {
                                "feature_setting": setting,
                                "split_name": split_name,
                                "model": model_name,
                                "params": json.dumps(best_params, sort_keys=True),
                                "val_mae": best_result["val_metrics"]["mae"],
                                "val_rmse": best_result["val_metrics"]["rmse"],
                                "val_spearman": best_result["val_metrics"]["spearman"],
                                "best_epoch": best_result["best_epoch"],
                                "num_features": len(feature_cols),
                            }
                        ]
                    ),
                )

    per_bearing = pd.read_csv(per_bearing_path)
    split_summary, protocol_summary = summarize_final(per_bearing)
    split_summary.to_csv(out_dir / "three_feature_settings_k40_split_summary.csv", index=False)
    protocol_summary.to_csv(out_dir / "three_feature_settings_k40_protocol_summary.csv", index=False)
    save_figures(split_summary, protocol_summary, fig_dir, out_dir)
    print(f"\nSaved three-feature-setting results to {out_dir}", flush=True)


def save_figures(split_summary: pd.DataFrame, protocol_summary: pd.DataFrame, fig_dir: Path, out_dir: Path) -> None:
    cross = protocol_summary[protocol_summary["protocol"] == "cross_condition"].copy()
    cross["feature_label"] = cross["feature_setting"].map(SETTING_LABELS).fillna(cross["feature_setting"])
    preferred_order = [
        "Raw waveform",
        "Original",
        "Wavelet-only",
        "Selected-top10",
        "Selected-top20",
        "Selected-top30",
        "Selected-top60",
        "Selected-top90",
    ]
    available_labels = set(cross["feature_label"])
    setting_order = [label for label in preferred_order if label in available_labels]
    row_order = [m for m in MODEL_ORDER if m in set(cross["model"])]

    pivot = cross.pivot_table(index="model", columns="feature_label", values="mae_mean", aggfunc="mean")
    pivot = pivot.reindex(index=row_order, columns=setting_order)
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=240)
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(setting_order)))
    ax.set_xticklabels(setting_order, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_title("Cross-condition MAE across feature settings")
    values = pivot.to_numpy(dtype=float)
    mean_value = np.nanmean(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            color = "white" if value < mean_value else "#0b1f3d"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9, weight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("MAE")
    fig.tight_layout()
    fig.savefig(fig_dir / "three_feature_settings_k40_model_feature_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    def save_subset_heatmap(labels: list[str], filename: str, title: str) -> None:
        sub_pivot = pivot.reindex(columns=labels)
        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=240)
        values = sub_pivot.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="YlGnBu_r", aspect="auto")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order)
        ax.set_title(title)
        mean_value = np.nanmean(values)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isnan(value):
                    continue
                color = "white" if value < mean_value else "#0b1f3d"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9, weight="bold", color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_label("MAE")
        fig.tight_layout()
        fig.savefig(fig_dir / filename, bbox_inches="tight")
        plt.close(fig)

    subset_specs = [
        (
            ["Raw waveform", "Wavelet-only"],
            "raw_waveform_vs_wavelet_k40_heatmap.png",
            "Raw waveform vs. Wavelet-only",
        ),
        (
            ["Original", "Wavelet-only", "Selected-top10"],
            "original_wavelet_top10_k40_heatmap.png",
            "Original vs. Wavelet-only vs. Selected-top10",
        ),
        (
            ["Selected-top10", "Selected-top20", "Selected-top30"],
            "selected_topk_k40_heatmap.png",
            "Selected feature set size comparison",
        ),
        (
            ["Selected-top30", "Selected-top60", "Selected-top90"],
            "selected_top30_top60_top90_k40_heatmap.png",
            "Selected feature set size comparison",
        ),
    ]
    for labels, filename, title in subset_specs:
        labels = [label for label in labels if label in available_labels]
        if labels:
            save_subset_heatmap(labels, filename, title)

    def save_big_table(labels: list[str], stem: str, title: str) -> None:
        rows = []
        for model in row_order:
            row = {"Model": model}
            for label in labels:
                sub = cross[(cross["model"] == model) & (cross["feature_label"] == label)]
                if sub.empty:
                    row[f"{label} MAE"] = ""
                    row[f"{label} RMSE"] = ""
                    row[f"{label} Spearman"] = ""
                    continue
                item = sub.iloc[0]
                row[f"{label} MAE"] = f"{item['mae_mean']:.3f} +/- {item['mae_std']:.3f}"
                row[f"{label} RMSE"] = f"{item['rmse_mean']:.3f} +/- {item['rmse_std']:.3f}"
                row[f"{label} Spearman"] = f"{item['spearman_mean']:.3f} +/- {item['spearman_std']:.3f}"
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(out_dir / f"{stem}.csv", index=False)

        fig_width = 3.0 + 2.3 * len(labels) * 3
        fig, ax = plt.subplots(figsize=(fig_width, 2.4 + 0.38 * len(table)), dpi=220)
        ax.axis("off")
        tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.2)
        tbl.scale(1.0, 1.45)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if r == 0:
                cell.set_facecolor("#e8f0fb")
                cell.set_text_props(weight="bold", color="#0b1f3d")
            elif r == 1:
                cell.set_facecolor("#fff7cc")
                cell.set_text_props(weight="bold")
        ax.set_title(title, fontsize=14, weight="bold", pad=14)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{stem}.png", bbox_inches="tight")
        plt.close(fig)

    table_specs = [
        (
            ["Raw waveform", "Wavelet-only"],
            "table_raw_waveform_vs_wavelet",
            "Raw waveform and Wavelet-only comparison",
        ),
        (
            ["Original", "Wavelet-only", "Selected-top10"],
            "table_original_wavelet_selected_top10",
            "Original, Wavelet-only, and Selected-top10 comparison",
        ),
        (
            ["Selected-top10", "Selected-top20", "Selected-top30"],
            "table_selected_top10_top20_top30",
            "Selected feature set size comparison",
        ),
        (
            ["Selected-top30", "Selected-top60", "Selected-top90"],
            "table_selected_top30_top60_top90",
            "Selected feature set size comparison",
        ),
    ]
    for labels, stem, title in table_specs:
        labels = [label for label in labels if label in available_labels]
        if labels:
            save_big_table(labels, stem, title)

    best = cross.sort_values("mae_mean").copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=240)
    labels = [f"{row.model}\n{row.feature_label}" for row in best.itertuples()]
    colors = [MODEL_COLORS.get(row.model, "#64748b") for row in best.itertuples()]
    bars = ax.bar(labels, best["mae_mean"], color=colors)
    ax.set_title("Best cross-condition MAE by model and feature setting")
    ax.set_ylabel("Bearing-level MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    for bar, value in zip(bars, best["mae_mean"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8, weight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "three_feature_settings_k40_sorted_mae_bar.png", bbox_inches="tight")
    plt.close(fig)

    for setting, label in SETTING_LABELS.items():
        heat = split_summary[
            (split_summary["protocol"] == "cross_condition")
            & (split_summary["feature_setting"] == setting)
        ].copy()
        if heat.empty:
            continue
        heat["test_condition"] = heat["split_name"].str.extract(r"test_(C\d)")
        split_pivot = heat.pivot_table(index="model", columns="test_condition", values="mae", aggfunc="mean")
        col_order = [c for c in ["C1", "C2", "C3"] if c in split_pivot.columns]
        split_pivot = split_pivot.reindex(index=row_order, columns=col_order)
        fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=240)
        im = ax.imshow(split_pivot.to_numpy(), cmap="YlGnBu_r", aspect="auto")
        ax.set_xticks(np.arange(len(col_order)))
        ax.set_xticklabels([f"Test {c}" for c in col_order])
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order)
        ax.set_title(f"{label}: per-split cross-condition MAE")
        values = split_pivot.to_numpy(dtype=float)
        mean_value = np.nanmean(values)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if np.isnan(value):
                    continue
                color = "white" if value < mean_value else "#0b1f3d"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8.5, weight="bold", color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("MAE")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{setting}_k40_per_split_heatmap.png", bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Strict cross-condition retraining for three feature settings.")
    parser.add_argument("--raw_features", default="processed/features_raw_downsample_256.csv")
    parser.add_argument("--original_features", default="processed/features.csv")
    parser.add_argument("--wavelet_features", default="processed/features_wavelet_only.csv")
    parser.add_argument("--expanded_features", default="processed/features_wavelet.csv")
    parser.add_argument("--selected_feature_dir", default="results/tables/strict_method_v2/selected_features")
    parser.add_argument("--split_dir", default="processed/splits_final")
    parser.add_argument("--out_dir", default="results/tables/three_feature_settings_k40")
    parser.add_argument("--fig_dir", default="results/figures/three_feature_settings_k40")
    parser.add_argument("--pred_dir", default="results/predictions_three_feature_settings_k40")
    parser.add_argument("--seq_dir", default="processed/sequences_three_feature_settings_k40")
    parser.add_argument("--ckpt_dir", default="results/checkpoints_three_feature_settings_k40")
    parser.add_argument(
        "--feature_settings",
        nargs="+",
        default=["original", "wavelet_only", "selected_top10", "selected_top20", "selected_top30"],
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge_alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--rebuild_sequences", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--fixed_grid", action="store_true")
    parser.add_argument("--latent_ode_epochs", type=int, default=12)
    parser.add_argument("--latent_ode_patience", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
