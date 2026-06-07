import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from feature_analysis import feature_group, score_features
from make_sequences import build_samples
from utils import ensure_dir, get_feature_columns, load_split, set_seed


META_COLUMNS = [
    "bearing_id",
    "condition_id",
    "speed_rpm",
    "load_kn",
    "file_path",
    "file_index",
    "time_index",
    "failure_time",
    "rul",
    "normalized_rul",
]

MODEL_COLORS = {
    "Ridge": "#8C8C8C",
    "LSTM": "#4C78A8",
    "TCN": "#00A676",
    "Transformer": "#0057B8",
    "latent_ode": "#7E57C2",
}


def subset_by_bearings(df: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    return df[df["bearing_id"].isin(bearings)].copy()


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    value = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def monotonic_violation_rate(y_pred: np.ndarray, epsilon: float = 0.01) -> float:
    if len(y_pred) < 2:
        return 0.0
    return float(np.mean(np.diff(y_pred) > epsilon))


def per_bearing_metrics(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    frame = meta.copy()
    frame["y_true"] = np.asarray(y_true, dtype=float)
    frame["y_pred"] = np.asarray(y_pred, dtype=float)
    rows = []
    for (bearing_id, condition_id), group in frame.groupby(["bearing_id", "condition_id"], sort=True):
        group = group.sort_values("time_index")
        yt = group["y_true"].to_numpy(dtype=float)
        yp = group["y_pred"].to_numpy(dtype=float)
        err = yp - yt
        late = yt <= 0.3
        rows.append(
            {
                "bearing_id": bearing_id,
                "condition_id": condition_id,
                "n_points": len(group),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "spearman": spearman_corr(yt, yp),
                "late_mae": float(np.mean(np.abs(err[late]))) if np.any(late) else np.nan,
                "monotonic_violation_rate": monotonic_violation_rate(yp),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics: pd.DataFrame) -> dict:
    cols = ["mae", "rmse", "spearman", "late_mae", "monotonic_violation_rate"]
    return {col: float(metrics[col].mean()) for col in cols}


def feature_columns_for_setting(
    setting: str,
    all_feature_cols: list[str],
    train_df: pd.DataFrame,
    selected_dir: Path,
    split_name: str,
) -> tuple[list[str], pd.DataFrame]:
    if setting == "time":
        return [col for col in all_feature_cols if feature_group(col) == "time"], pd.DataFrame()
    if setting == "frequency":
        return [col for col in all_feature_cols if feature_group(col) == "frequency"], pd.DataFrame()
    if setting == "wavelet":
        return [col for col in all_feature_cols if feature_group(col) == "wavelet"], pd.DataFrame()
    if setting == "original":
        return [col for col in all_feature_cols if feature_group(col) != "wavelet"], pd.DataFrame()
    if setting == "all_expanded":
        return list(all_feature_cols), pd.DataFrame()
    if setting.startswith("selected_top"):
        k = int(setting.replace("selected_top", ""))
        scores = score_features(train_df, all_feature_cols)
        selected = scores.head(k)["feature"].tolist()
        out = scores.head(k).copy()
        out.insert(0, "split_name", split_name)
        out.insert(1, "feature_setting", setting)
        out.to_csv(selected_dir / f"{split_name}_{setting}_features.csv", index=False)
        return selected, scores
    raise ValueError(f"Unknown feature setting: {setting}")


def tune_and_test_ridge(
    features_df: pd.DataFrame,
    split: dict,
    feature_cols: list[str],
    alphas: list[float],
) -> tuple[dict, pd.DataFrame]:
    train_df = subset_by_bearings(features_df, split["train_bearings"])
    val_df = subset_by_bearings(features_df, split["val_bearings"])
    test_df = subset_by_bearings(features_df, split["test_bearings"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_val = val_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)

    val_meta = val_df[["bearing_id", "condition_id", "time_index"]].copy()
    best = None
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred_val = np.clip(model.predict(X_val), 0.0, 1.0)
        val_summary = aggregate_metrics(per_bearing_metrics(val_meta, y_val, pred_val))
        if best is None or val_summary["mae"] < best["val_mae"]:
            best = {"model": model, "alpha": alpha, "val_mae": val_summary["mae"], "val_rmse": val_summary["rmse"]}

    pred_test = np.clip(best["model"].predict(X_test), 0.0, 1.0)
    test_meta = test_df[["bearing_id", "condition_id", "time_index"]].copy()
    per_bearing = per_bearing_metrics(test_meta, y_test, pred_test)
    summary = aggregate_metrics(per_bearing)
    summary.update({"alpha": best["alpha"], "val_mae": best["val_mae"], "val_rmse": best["val_rmse"]})
    return summary, per_bearing


def run_ridge_feature_ablation(args) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    selected_dir = ensure_dir(out_dir / "selected_features")
    features_df = pd.read_csv(args.expanded_features)
    all_feature_cols = get_feature_columns(features_df)
    split_paths = [
        path
        for path in sorted(Path(args.split_dir).glob("*.json"))
        if load_split(path).get("protocol") == "cross_condition"
    ]
    settings = ["time", "frequency", "wavelet", "original", "all_expanded"] + [
        f"selected_top{k}" for k in args.top_k_values
    ]

    result_rows = []
    per_bearing_rows = []
    composition_rows = []
    for split_path in split_paths:
        split = load_split(split_path)
        split_name = split["split_name"]
        train_df = subset_by_bearings(features_df, split["train_bearings"])
        for setting in settings:
            feature_cols, scores = feature_columns_for_setting(setting, all_feature_cols, train_df, selected_dir, split_name)
            if not feature_cols:
                continue
            group_counts = pd.Series([feature_group(col) for col in feature_cols]).value_counts().to_dict()
            composition_rows.append(
                {
                    "split_name": split_name,
                    "feature_setting": setting,
                    "num_features": len(feature_cols),
                    "time": group_counts.get("time", 0),
                    "frequency": group_counts.get("frequency", 0),
                    "wavelet": group_counts.get("wavelet", 0),
                    "selected_features": json.dumps(feature_cols, ensure_ascii=True),
                }
            )
            print(f"Ridge feature ablation: split={split_name}, setting={setting}, features={len(feature_cols)}", flush=True)
            summary, per_bearing = tune_and_test_ridge(features_df, split, feature_cols, args.alphas)
            result_rows.append(
                {
                    "protocol": split["protocol"],
                    "split_name": split_name,
                    "test_condition": ",".join(split.get("test_conditions", [])),
                    "model": "Ridge",
                    "feature_setting": setting,
                    "num_features": len(feature_cols),
                    **summary,
                }
            )
            per_bearing["protocol"] = split["protocol"]
            per_bearing["split_name"] = split_name
            per_bearing["test_condition"] = ",".join(split.get("test_conditions", []))
            per_bearing["model"] = "Ridge"
            per_bearing["feature_setting"] = setting
            per_bearing["num_features"] = len(feature_cols)
            per_bearing_rows.append(per_bearing)

    results = pd.DataFrame(result_rows)
    per_bearing_all = pd.concat(per_bearing_rows, ignore_index=True)
    composition = pd.DataFrame(composition_rows)
    results.to_csv(out_dir / "ridge_feature_ablation_split_summary.csv", index=False)
    per_bearing_all.to_csv(out_dir / "ridge_feature_ablation_per_bearing.csv", index=False)
    composition.to_csv(out_dir / "feature_setting_composition_by_split.csv", index=False)
    avg = (
        results.groupby(["feature_setting", "model"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            spearman_mean=("spearman", "mean"),
            late_mae_mean=("late_mae", "mean"),
            n_features=("num_features", "mean"),
            n_splits=("split_name", "nunique"),
        )
        .sort_values("mae_mean")
    )
    avg.to_csv(out_dir / "ridge_feature_ablation_average.csv", index=False)
    plot_feature_ablation(results, avg, composition, fig_dir)


def run_k_coverage(args) -> None:
    out_dir = ensure_dir(args.out_dir)
    features_df = pd.read_csv(args.wavelet_features)
    feature_cols = get_feature_columns(features_df)
    split_paths = [
        path
        for path in sorted(Path(args.split_dir).glob("*.json"))
        if load_split(path).get("protocol") == "cross_condition"
    ]
    rows = []
    for split_path in split_paths:
        split = load_split(split_path)
        for k in args.k_values:
            for partition, bearings in [
                ("train", split["train_bearings"]),
                ("val", split["val_bearings"]),
                ("test", split["test_bearings"]),
            ]:
                part_df = subset_by_bearings(features_df, bearings)
                _, y, _, _, meta = build_samples(part_df, feature_cols, k)
                present = set(pd.DataFrame(meta, columns=["bearing_id", "condition_id", "time_index"])["bearing_id"]) if len(meta) else set()
                rows.append(
                    {
                        "split_name": split["split_name"],
                        "test_condition": ",".join(split.get("test_conditions", [])),
                        "K": k,
                        "partition": partition,
                        "expected_bearings": len(bearings),
                        "covered_bearings": len(present),
                        "coverage": len(present) / len(bearings) if bearings else 0.0,
                        "num_sequences": int(len(y)),
                    }
                )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(out_dir / "k_coverage_by_split.csv", index=False)
    summary = (
        coverage.groupby(["K", "partition"], as_index=False)
        .agg(
            avg_coverage=("coverage", "mean"),
            min_coverage=("coverage", "min"),
            avg_sequences=("num_sequences", "mean"),
        )
        .sort_values(["K", "partition"])
    )
    summary.to_csv(out_dir / "k_coverage_summary.csv", index=False)


def plot_feature_ablation(results: pd.DataFrame, avg: pd.DataFrame, composition: pd.DataFrame, fig_dir: Path) -> None:
    ensure_dir(fig_dir)
    selected_settings = sorted(
        [setting for setting in results["feature_setting"].unique() if str(setting).startswith("selected_top")],
        key=lambda name: int(str(name).replace("selected_top", "")),
    )
    setting_order = ["time", "frequency", "wavelet", "original", "all_expanded"] + selected_settings
    label_map = {
        "time": "Time",
        "frequency": "Frequency",
        "wavelet": "Wavelet",
        "original": "Original",
        "all_expanded": "All-expanded",
    }
    for setting in selected_settings:
        label_map[setting] = f"Top{setting.replace('selected_top', '')}"
    pivot = results.pivot_table(index="feature_setting", columns="test_condition", values="mae", aggfunc="mean").reindex(setting_order)
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=220)
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([label_map.get(x, x) for x in pivot.index])
    ax.set_title("Ridge feature-setting ablation (cross-condition MAE)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE")
    fig.tight_layout()
    fig.savefig(fig_dir / "ridge_feature_ablation_heatmap.png", bbox_inches="tight")
    fig.savefig(fig_dir / "ridge_feature_ablation_heatmap.svg", bbox_inches="tight")
    plt.close(fig)

    avg_plot = avg.set_index("feature_setting").reindex(setting_order).dropna(subset=["mae_mean"])
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=220)
    colors = ["#4C78A8", "#72B7B2", "#00A676", "#7E57C2", "#8C8C8C", "#F6C85F", "#F58518", "#E45756"]
    bars = ax.bar(np.arange(len(avg_plot)), avg_plot["mae_mean"], color=colors[: len(avg_plot)])
    ax.errorbar(np.arange(len(avg_plot)), avg_plot["mae_mean"], yerr=avg_plot["mae_std"].fillna(0), fmt="none", ecolor="#263238", capsize=4, lw=1)
    ax.set_xticks(np.arange(len(avg_plot)))
    ax.set_xticklabels([label_map.get(x, x) for x in avg_plot.index], rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Average cross-condition MAE by feature setting")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, avg_plot["mae_mean"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "ridge_feature_ablation_bar.png", bbox_inches="tight")
    fig.savefig(fig_dir / "ridge_feature_ablation_bar.svg", bbox_inches="tight")
    plt.close(fig)

    top_comp = composition[composition["feature_setting"].isin(selected_settings)]
    if not top_comp.empty:
        comp_avg = top_comp.groupby("feature_setting", as_index=True)[["time", "frequency", "wavelet"]].mean().reindex(selected_settings)
        fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=220)
        bottom = np.zeros(len(comp_avg))
        for group, color in [("time", "#4C78A8"), ("frequency", "#72B7B2"), ("wavelet", "#00A676")]:
            vals = comp_avg[group].to_numpy(dtype=float)
            ax.bar(np.arange(len(comp_avg)), vals, bottom=bottom, label=group.capitalize(), color=color)
            bottom += vals
        ax.set_xticks(np.arange(len(comp_avg)))
        ax.set_xticklabels([label_map.get(x, x) for x in comp_avg.index])
        ax.set_ylabel("Average number of selected features")
        ax.set_title("Composition of train-only selected top-k features")
        ax.legend(frameon=True)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_dir / "selected_topk_feature_composition.png", bbox_inches="tight")
        fig.savefig(fig_dir / "selected_topk_feature_composition.svg", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Method v2 experiment utilities.")
    parser.add_argument("--mode", nargs="+", default=["feature_ablation", "k_coverage"], choices=["feature_ablation", "k_coverage"])
    parser.add_argument("--expanded_features", default="processed/features_wavelet.csv")
    parser.add_argument("--wavelet_features", default="processed/features_wavelet_only.csv")
    parser.add_argument("--split_dir", default="processed/splits_final")
    parser.add_argument("--out_dir", type=Path, default=Path("results/tables/strict_method_v2"))
    parser.add_argument("--fig_dir", type=Path, default=Path("results/figures/strict_method_v2"))
    parser.add_argument("--top_k_values", nargs="+", type=int, default=[10, 20, 30])
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.fig_dir)
    if "feature_ablation" in args.mode:
        run_ridge_feature_ablation(args)
    if "k_coverage" in args.mode:
        run_k_coverage(args)


if __name__ == "__main__":
    main()
