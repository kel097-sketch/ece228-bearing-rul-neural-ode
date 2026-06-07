import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import ACTIVE_MODEL_ORDER
from utils import ensure_dir


MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "LSTM": "#54A24B",
    "TCN": "#F58518",
    "Transformer": "#72B7B2",
    "latent_ode": "#B279A2",
}

STAGES = [
    ("early", 0.7, 1.000001),
    ("middle", 0.3, 0.7),
    ("late", -0.000001, 0.3),
]


def model_prediction_path(prediction_dir: Path, split_name: str, model: str) -> Path:
    suffix = {
        "Ridge": "Ridge",
        "LSTM": "LSTM",
        "TCN": "TCN",
        "Transformer": "Transformer",
        "latent_ode": "latent_ode",
    }[model]
    return prediction_dir / f"{split_name}_{suffix}.csv"


def load_predictions(prediction_dir: Path, split_dir: Path, models: list[str], min_time_index: int, clip: bool) -> pd.DataFrame:
    frames = []
    for split_path in sorted(split_dir.glob("*.json")):
        split = pd.read_json(split_path, typ="series").to_dict()
        split_name = split["split_name"]
        for model in models:
            path = model_prediction_path(prediction_dir, split_name, model)
            if not path.exists():
                print(f"Skipping missing prediction file: {path}")
                continue
            frame = pd.read_csv(path)
            frame = frame[frame["time_index"] >= min_time_index].copy()
            if clip:
                frame["y_pred_raw"] = frame["y_pred"]
                frame["y_pred"] = frame["y_pred"].clip(0.0, 1.0)
            frame["model"] = model
            frame["protocol"] = split["protocol"]
            frame["split_name"] = split_name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir} for splits in {split_dir}")
    return pd.concat(frames, ignore_index=True)


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def monotonic_violation_rate(y_pred: np.ndarray, epsilon: float) -> float:
    if len(y_pred) < 2:
        return float("nan")
    return float(np.mean(np.diff(y_pred) > epsilon))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> dict:
    error = y_pred - y_true
    late_mask = y_true <= 0.3
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "spearman": spearman_corr(y_true, y_pred),
        "late_mae": float(np.mean(np.abs(error[late_mask]))) if np.any(late_mask) else float("nan"),
        "monotonic_violation_rate": monotonic_violation_rate(y_pred, epsilon),
    }


def compute_per_bearing_metrics(predictions: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    rows = []
    group_cols = ["protocol", "split_name", "model", "condition_id", "bearing_id"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        group = group.sort_values("time_index")
        y_true = group["normalized_rul"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_cols, keys, strict=False)),
                "n_points": len(group),
                **metric_dict(y_true, y_pred, epsilon),
            }
        )
    return pd.DataFrame(rows)


def compute_stage_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["protocol", "split_name", "model", "condition_id", "bearing_id"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        y_true_all = group["normalized_rul"].to_numpy(dtype=float)
        y_pred_all = group["y_pred"].to_numpy(dtype=float)
        for stage, lower, upper in STAGES:
            mask = (y_true_all >= lower) & (y_true_all < upper)
            if not np.any(mask):
                continue
            rows.append(
                {
                    **dict(zip(group_cols, keys, strict=False)),
                    "stage": stage,
                    "n_points": int(np.sum(mask)),
                    "mae": float(np.mean(np.abs(y_pred_all[mask] - y_true_all[mask]))),
                    "rmse": float(np.sqrt(np.mean((y_pred_all[mask] - y_true_all[mask]) ** 2))),
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(per_bearing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["mae", "rmse", "r2", "spearman", "late_mae", "monotonic_violation_rate"]
    split_summary = (
        per_bearing.groupby(["protocol", "split_name", "model"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["protocol", "split_name", "mae"])
    )
    protocol_summary = (
        split_summary.groupby(["protocol", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            late_mae_mean=("late_mae", "mean"),
            late_mae_std=("late_mae", "std"),
            monotonic_violation_rate_mean=("monotonic_violation_rate", "mean"),
            monotonic_violation_rate_std=("monotonic_violation_rate", "std"),
            num_splits=("split_name", "nunique"),
        )
        .sort_values(["protocol", "mae_mean"])
    )
    return split_summary, protocol_summary


def aggregate_stage_metrics(stage_metrics: pd.DataFrame) -> pd.DataFrame:
    if stage_metrics.empty:
        return stage_metrics
    bearing_stage = (
        stage_metrics.groupby(["protocol", "split_name", "model", "stage"], as_index=False)[["mae", "rmse"]]
        .mean()
    )
    return (
        bearing_stage.groupby(["protocol", "model", "stage"], as_index=False)
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"), num_splits=("split_name", "nunique"))
        .sort_values(["protocol", "stage", "mae_mean"])
    )


def model_order(models: list[str]) -> list[str]:
    return [model for model in ACTIVE_MODEL_ORDER if model in models] + [
        model for model in models if model not in ACTIVE_MODEL_ORDER
    ]


def save_bar_cross(protocol_summary: pd.DataFrame, out_dir: Path, models: list[str]) -> None:
    data = protocol_summary[protocol_summary["protocol"] == "cross_condition"].copy()
    if data.empty:
        return
    order = model_order(models)
    data["model"] = pd.Categorical(data["model"], categories=order, ordered=True)
    data = data.sort_values("model")
    plt.figure(figsize=(7.2, 4.2))
    x = np.arange(len(data))
    colors = [MODEL_COLORS.get(model, "#777777") for model in data["model"].astype(str)]
    plt.bar(x, data["mae_mean"], yerr=data["mae_std"].fillna(0.0), color=colors, capsize=4)
    plt.xticks(x, data["model"].astype(str), rotation=25, ha="right")
    plt.ylabel("Bearing-level MAE")
    plt.title("Cross-condition MAE (mean over held-out conditions)")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = out_dir / "final_cross_condition_mae_bar.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def split_to_test_condition(split_name: str) -> str:
    match = re.search(r"_test_(C\d)$", split_name)
    return f"Test {match.group(1)}" if match else split_name


def save_per_split_heatmap(split_summary: pd.DataFrame, out_dir: Path, models: list[str]) -> None:
    cross = split_summary[split_summary["protocol"] == "cross_condition"].copy()
    if cross.empty:
        return
    cross["test_condition"] = cross["split_name"].map(split_to_test_condition)
    pivot = cross.pivot_table(index="model", columns="test_condition", values="mae", aggfunc="mean")
    row_order = [model for model in model_order(models) if model in pivot.index]
    col_order = [col for col in ["Test C1", "Test C2", "Test C3"] if col in pivot.columns]
    pivot = pivot.loc[row_order, col_order]
    plt.figure(figsize=(7.0, 4.5))
    image = plt.imshow(pivot.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            plt.text(j, i, f"{value:.3f}", ha="center", va="center", color="#102033", fontsize=10)
    plt.colorbar(image, label="Bearing-level MAE")
    plt.title("Per-split cross-condition MAE")
    plt.tight_layout()
    path = out_dir / "final_cross_condition_per_split_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def save_cross_vs_mixed(protocol_summary: pd.DataFrame, out_dir: Path, models: list[str]) -> None:
    data = protocol_summary[protocol_summary["protocol"].isin(["cross_condition", "mixed_condition"])].copy()
    if data.empty:
        return
    order = model_order(models)
    protocols = ["cross_condition", "mixed_condition"]
    x = np.arange(len(order))
    width = 0.36
    plt.figure(figsize=(8.0, 4.2))
    for offset, protocol in [(-width / 2, "cross_condition"), (width / 2, "mixed_condition")]:
        values = []
        errors = []
        for model in order:
            row = data[(data["protocol"] == protocol) & (data["model"] == model)]
            values.append(float(row["mae_mean"].iloc[0]) if not row.empty else np.nan)
            errors.append(float(row["mae_std"].fillna(0.0).iloc[0]) if not row.empty else 0.0)
        plt.bar(x + offset, values, width, yerr=errors, capsize=3, label=protocol.replace("_", "-"), alpha=0.9)
    plt.xticks(x, order, rotation=25, ha="right")
    plt.ylabel("Bearing-level MAE")
    plt.title("Cross-condition vs mixed-condition")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = out_dir / "final_cross_vs_mixed_mae.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def save_stage_mae(stage_summary: pd.DataFrame, out_dir: Path, models: list[str]) -> None:
    data = stage_summary[stage_summary["protocol"] == "cross_condition"].copy()
    if data.empty:
        return
    order = model_order(models)
    stages = ["early", "middle", "late"]
    x = np.arange(len(stages))
    width = 0.14
    plt.figure(figsize=(8.6, 4.4))
    for i, model in enumerate(order):
        values = []
        for stage in stages:
            row = data[(data["model"] == model) & (data["stage"] == stage)]
            values.append(float(row["mae_mean"].iloc[0]) if not row.empty else np.nan)
        plt.bar(x + (i - (len(order) - 1) / 2) * width, values, width, label=model, color=MODEL_COLORS.get(model, None))
    plt.xticks(x, ["Early\nRUL 0.7-1.0", "Middle\nRUL 0.3-0.7", "Late\nRUL 0-0.3"])
    plt.ylabel("MAE")
    plt.title("Cross-condition stage-wise MAE")
    plt.legend(frameon=False, ncols=3, fontsize=8)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    path = out_dir / "final_cross_condition_stage_mae.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def representative_subset(predictions: pd.DataFrame, preferred_split: str | None, preferred_bearing: str | None) -> pd.DataFrame:
    data = predictions[predictions["protocol"] == "cross_condition"].copy()
    if preferred_split:
        candidate = data[data["split_name"] == preferred_split]
        if not candidate.empty:
            data = candidate
    if preferred_bearing:
        candidate = data[data["bearing_id"].astype(str) == preferred_bearing]
        if not candidate.empty:
            data = candidate
    split_name = str(data["split_name"].iloc[0])
    bearing_id = str(data[data["split_name"] == split_name]["bearing_id"].iloc[0])
    return data[(data["split_name"] == split_name) & (data["bearing_id"].astype(str) == bearing_id)].copy()


def save_prediction_curve(predictions: pd.DataFrame, out_dir: Path, models: list[str], preferred_split: str | None, preferred_bearing: str | None, show_target: bool) -> None:
    data = representative_subset(predictions, preferred_split, preferred_bearing)
    order = model_order(models)
    plt.figure(figsize=(8.0, 4.4))
    if show_target:
        target = data.drop_duplicates("time_index").sort_values("time_index")
        plt.plot(
            target["time_index"],
            target["normalized_rul"],
            color="#333333",
            linewidth=1.5,
            linestyle="--",
            alpha=0.55,
            label="constructed normalized RUL target",
        )
    for model in order:
        one = data[data["model"] == model].sort_values("time_index")
        if one.empty:
            continue
        plt.plot(one["time_index"], one["y_pred"], label=model, color=MODEL_COLORS.get(model), linewidth=1.4)
    title_bits = [str(data["split_name"].iloc[0]), str(data["bearing_id"].iloc[0])]
    plt.title("Prediction trajectories: " + " / ".join(title_bits))
    plt.xlabel("Time index")
    plt.ylabel("Predicted normalized RUL")
    plt.ylim(-0.03, 1.03)
    plt.legend(frameon=False, ncols=2, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    path = out_dir / "final_representative_prediction_trajectory.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def save_absolute_error_curve(predictions: pd.DataFrame, out_dir: Path, models: list[str], preferred_split: str | None, preferred_bearing: str | None) -> None:
    data = representative_subset(predictions, preferred_split, preferred_bearing)
    order = model_order(models)
    plt.figure(figsize=(8.0, 4.4))
    for model in order:
        one = data[data["model"] == model].sort_values("time_index").copy()
        if one.empty:
            continue
        one["abs_error"] = (one["y_pred"] - one["normalized_rul"]).abs()
        one["abs_error_smooth"] = one["abs_error"].rolling(window=25, min_periods=1, center=True).mean()
        plt.plot(one["time_index"], one["abs_error_smooth"], label=model, color=MODEL_COLORS.get(model), linewidth=1.4)
    title_bits = [str(data["split_name"].iloc[0]), str(data["bearing_id"].iloc[0])]
    plt.title("Smoothed absolute error: " + " / ".join(title_bits))
    plt.xlabel("Time index")
    plt.ylabel("Absolute error")
    plt.legend(frameon=False, ncols=2, fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    path = out_dir / "final_representative_absolute_error.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def save_outputs(
    predictions: pd.DataFrame,
    per_bearing: pd.DataFrame,
    split_summary: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    stage_metrics: pd.DataFrame,
    stage_summary: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
    models: list[str],
    args,
) -> None:
    ensure_dir(out_dir)
    ensure_dir(fig_dir)
    predictions.to_csv(out_dir / "final_predictions_used.csv", index=False)
    per_bearing.to_csv(out_dir / "final_per_bearing_metrics.csv", index=False)
    split_summary.to_csv(out_dir / "final_split_metrics.csv", index=False)
    protocol_summary.to_csv(out_dir / "final_protocol_summary.csv", index=False)
    stage_metrics.to_csv(out_dir / "final_stage_metrics_per_bearing.csv", index=False)
    stage_summary.to_csv(out_dir / "final_stage_metrics_summary.csv", index=False)
    print(f"Saved final metric tables to {out_dir}")

    save_bar_cross(protocol_summary, fig_dir, models)
    save_per_split_heatmap(split_summary, fig_dir, models)
    save_cross_vs_mixed(protocol_summary, fig_dir, models)
    save_stage_mae(stage_summary, fig_dir, models)
    save_prediction_curve(predictions, fig_dir, models, args.preferred_split, args.preferred_bearing, args.show_target)
    save_absolute_error_curve(predictions, fig_dir, models, args.preferred_split, args.preferred_bearing)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute bearing-level final metrics and publication-ready figures.")
    parser.add_argument("--prediction_dir", default="results/predictions")
    parser.add_argument("--split_dir", default="processed/splits_final")
    parser.add_argument("--out_dir", default="results/tables/final")
    parser.add_argument("--fig_dir", default="results/figures/final")
    parser.add_argument("--models", nargs="+", default=ACTIVE_MODEL_ORDER)
    parser.add_argument("--min_time_index", type=int, default=19)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--show_target", action="store_true")
    parser.add_argument("--preferred_split", default="cross_train_C1_C2_test_C3")
    parser.add_argument("--preferred_bearing", default="C3_Bearing3_1")
    return parser.parse_args()


def main():
    args = parse_args()
    models = args.models
    predictions = load_predictions(
        Path(args.prediction_dir),
        Path(args.split_dir),
        models,
        min_time_index=args.min_time_index,
        clip=not args.no_clip,
    )
    per_bearing = compute_per_bearing_metrics(predictions, args.epsilon)
    split_summary, protocol_summary = aggregate_metrics(per_bearing)
    stage_metrics = compute_stage_metrics(predictions)
    stage_summary = aggregate_stage_metrics(stage_metrics)
    save_outputs(
        predictions,
        per_bearing,
        split_summary,
        protocol_summary,
        stage_metrics,
        stage_summary,
        Path(args.out_dir),
        Path(args.fig_dir),
        models,
        args,
    )


if __name__ == "__main__":
    main()
