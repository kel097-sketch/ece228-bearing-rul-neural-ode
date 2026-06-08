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
from feature_analysis import feature_group
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
    split_df,
    train_one_config,
)
from utils import ensure_dir, get_feature_columns, load_split, set_seed


SETTING_LABELS = {
    "time": "Time",
    "frequency": "Frequency",
    "wavelet_only": "Wavelet",
    "original": "Original",
}


def feature_subset(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    meta_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:5]}")
    return df[meta_cols + feature_cols].copy()


def load_setting_features(setting: str, args) -> tuple[pd.DataFrame, list[str]]:
    if setting in {"time", "frequency", "original"}:
        df = pd.read_csv(args.original_features)
        all_cols = get_feature_columns(df)
        if setting == "time":
            cols = [col for col in all_cols if feature_group(col) == "time"]
        elif setting == "frequency":
            cols = [col for col in all_cols if feature_group(col) == "frequency"]
        else:
            cols = all_cols
        return feature_subset(df, cols), cols
    if setting == "wavelet_only":
        df = pd.read_csv(args.wavelet_features)
        cols = get_feature_columns(df)
        return feature_subset(df, cols), cols
    raise ValueError(setting)


def load_wavelet_best_params(path: Path) -> dict[tuple[str, str], dict]:
    frame = pd.read_csv(path)
    params = {}
    for row in frame.itertuples(index=False):
        params[(row.split_name, row.model)] = json.loads(row.params)
    return params


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


def train_ridge_fixed(
    features_df: pd.DataFrame,
    split: dict,
    feature_cols: list[str],
    setting: str,
    params: dict,
    args,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    alpha = float(params["alpha"])
    train_df = split_df(features_df, split["train_bearings"])
    test_df = split_df(features_df, split["test_bearings"])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    x_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    test_pred = np.clip(model.predict(x_test), 0.0, 1.0)
    test_meta = test_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object)
    pred = prediction_frame(test_meta, y_test, test_pred, split, "Ridge", setting, None)
    per_bearing = per_bearing_metrics(test_meta, y_test, test_pred, args.epsilon)
    per_bearing["model"] = "Ridge"
    per_bearing["feature_setting"] = setting
    per_bearing["K"] = "NA"
    per_bearing["protocol"] = split["protocol"]
    per_bearing["split_name"] = split["split_name"]
    per_bearing["best_params"] = json.dumps({"alpha": alpha}, sort_keys=True)
    per_bearing["num_features"] = len(feature_cols)
    return pred, per_bearing, {"alpha": alpha}


def save_figures(split_summary: pd.DataFrame, protocol_summary: pd.DataFrame, fig_dir: Path) -> None:
    cross = protocol_summary[protocol_summary["protocol"] == "cross_condition"].copy()
    cross["feature_label"] = cross["feature_setting"].map(SETTING_LABELS).fillna(cross["feature_setting"])
    setting_order = ["Time", "Frequency", "Wavelet", "Original"]
    row_order = [model for model in MODEL_ORDER if model in set(cross["model"])]

    pivot = cross.pivot_table(index="model", columns="feature_label", values="mae_mean", aggfunc="mean")
    pivot = pivot.reindex(index=row_order, columns=[s for s in setting_order if s in pivot.columns])
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=240)
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Fixed wavelet-tuned parameters: cross-condition MAE")
    mean_value = np.nanmean(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            color = "white" if value < mean_value else "#0b1f3d"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9, weight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE")
    fig.tight_layout()
    fig.savefig(fig_dir / "fixed_wavelet_params_feature_setting_mae_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    best_by_setting = (
        cross.sort_values("mae_mean")
        .groupby("feature_label", as_index=False, sort=False)
        .first()
        .sort_values("mae_mean")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=240)
    bars = ax.bar(
        best_by_setting["feature_label"],
        best_by_setting["mae_mean"],
        color=[MODEL_COLORS.get(model, "#64748b") for model in best_by_setting["model"]],
    )
    ax.set_title("Best model for each feature setting")
    ax.set_ylabel("Bearing-level MAE")
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, best_by_setting.itertuples(index=False), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{row.mae_mean:.3f}\n{row.model}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(fig_dir / "fixed_wavelet_params_best_feature_setting_bar.png", bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for setting in setting_order:
        sub = cross[cross["feature_label"] == setting].sort_values("mae_mean")
        for row in sub.itertuples(index=False):
            table_rows.append(
                {
                    "Feature": setting,
                    "Model": row.model,
                    "MAE": f"{row.mae_mean:.3f} +/- {row.mae_std:.3f}",
                    "RMSE": f"{row.rmse_mean:.3f} +/- {row.rmse_std:.3f}",
                    "Spearman": f"{row.spearman_mean:.3f} +/- {row.spearman_std:.3f}",
                }
            )
    table = pd.DataFrame(table_rows)
    table.to_csv(fig_dir.parent.parent / "tables" / "fixed_wavelet_params_k40_feature_settings" / "fixed_wavelet_params_feature_setting_table_for_paper.csv", index=False)


def run(args) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    pred_dir = ensure_dir(args.pred_dir)
    seq_root = ensure_dir(args.seq_dir)
    split_paths = cross_split_paths(Path(args.split_dir))
    best_params = load_wavelet_best_params(Path(args.wavelet_best_configs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_bearing_path = out_dir / "fixed_wavelet_params_k40_per_bearing_metrics.csv"
    predictions_path = out_dir / "fixed_wavelet_params_k40_predictions.csv"
    configs_path = out_dir / "fixed_wavelet_params_k40_used_configs.csv"
    split_summary_path = out_dir / "fixed_wavelet_params_k40_split_summary.csv"
    protocol_summary_path = out_dir / "fixed_wavelet_params_k40_protocol_summary.csv"
    if args.reset:
        for path in [per_bearing_path, predictions_path, configs_path, split_summary_path, protocol_summary_path]:
            if path.exists():
                path.unlink()

    for setting in args.feature_settings:
        features_df, feature_cols = load_setting_features(setting, args)
        print(f"\n=== Feature setting={setting}, dim={len(feature_cols)}, device={device} ===", flush=True)
        for split_path in split_paths:
            split = load_split(split_path)
            split_name = split["split_name"]
            seq_dir = ensure_dir(seq_root / setting / f"k{args.k}")
            seq_path = seq_dir / f"{split_name}_k{args.k}.npz"
            if args.rebuild_sequences or not seq_path.exists():
                seq_path = make_sequence_file(features_df, split_path, args.k, seq_dir)
            data, _ = load_sequence(seq_path)

            for model_name in ["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"]:
                if result_exists(per_bearing_path, setting, split_name, model_name):
                    print(f"Skipping existing setting={setting} split={split_name} model={model_name}", flush=True)
                    continue
                params = copy.deepcopy(best_params[(split_name, model_name)])
                print(f"Running setting={setting} split={split_name} model={model_name}", flush=True)
                if model_name == "Ridge":
                    pred_frame, metrics, info = train_ridge_fixed(features_df, split, feature_cols, setting, params, args)
                    best_epoch = None
                    val_metrics = {}
                    params_json = json.dumps({"alpha": info["alpha"]}, sort_keys=True)
                else:
                    seed = args.seed + sum(ord(ch) for ch in f"{setting}:{split_name}:{model_name}")
                    result = train_one_config(model_name, params, data, args, device, seed)
                    metrics = result["test_per_bearing"].copy()
                    metrics["model"] = model_name
                    metrics["feature_setting"] = setting
                    metrics["K"] = args.k
                    metrics["protocol"] = split["protocol"]
                    metrics["split_name"] = split_name
                    metrics["best_params"] = json.dumps(params, sort_keys=True)
                    metrics["num_features"] = len(feature_cols)
                    pred_frame = prediction_frame(
                        data["meta_test"],
                        data["y_test"],
                        result["test_pred"],
                        split,
                        model_name,
                        setting,
                        args.k,
                    )
                    best_epoch = result["best_epoch"]
                    val_metrics = result["val_metrics"]
                    params_json = json.dumps(params, sort_keys=True)

                append_frame(per_bearing_path, metrics)
                append_frame(predictions_path, pred_frame)
                append_frame(
                    configs_path,
                    pd.DataFrame(
                        [
                            {
                                "feature_setting": setting,
                                "split_name": split_name,
                                "model": model_name,
                                "params": params_json,
                                "best_epoch": best_epoch,
                                "val_mae": val_metrics.get("mae"),
                                "val_rmse": val_metrics.get("rmse"),
                                "val_spearman": val_metrics.get("spearman"),
                                "num_features": len(feature_cols),
                            }
                        ]
                    ),
                )

    if not per_bearing_path.exists():
        raise RuntimeError("No results were written.")
    per_bearing = pd.read_csv(per_bearing_path)
    split_summary, protocol_summary = summarize_final(per_bearing)
    split_summary.to_csv(split_summary_path, index=False)
    protocol_summary.to_csv(protocol_summary_path, index=False)
    save_figures(split_summary, protocol_summary, fig_dir)
    print(f"\nSaved fixed-parameter feature-setting results to {out_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare feature settings with fixed wavelet-tuned K=40 parameters.")
    parser.add_argument("--original_features", default="processed/features.csv")
    parser.add_argument("--wavelet_features", default="processed/features_wavelet_only.csv")
    parser.add_argument("--split_dir", default="processed/splits_final")
    parser.add_argument("--wavelet_best_configs", default="results/tables/tuned_wavelet_k40/best_configs_by_split.csv")
    parser.add_argument("--out_dir", type=Path, default=Path("results/tables/fixed_wavelet_params_k40_feature_settings"))
    parser.add_argument("--fig_dir", type=Path, default=Path("results/figures/fixed_wavelet_params_k40_feature_settings"))
    parser.add_argument("--pred_dir", type=Path, default=Path("results/predictions_fixed_wavelet_params_k40_feature_settings"))
    parser.add_argument("--seq_dir", type=Path, default=Path("processed/sequences_fixed_wavelet_params_k40_feature_settings"))
    parser.add_argument("--feature_settings", nargs="+", default=["time", "frequency", "wavelet_only", "original"])
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild_sequences", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
