import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import ACTIVE_MODEL_ORDER
from utils import ensure_dir


MODEL_ORDER = ACTIVE_MODEL_ORDER
MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "LSTM": "#54A24B",
    "TCN": "#F58518",
    "Transformer": "#72B7B2",
    "latent_ode": "#B279A2",
    "condition_aware_ode": "#E45756",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_current_figure(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if path.suffix.lower() != ".pdf":
        plt.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"Saved {path}")


def plot_feature_curves(features_path: str) -> None:
    path = Path(features_path)
    if not path.exists():
        print(f"Skipping feature curves; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("Skipping feature curves; features file is empty.")
        return

    bearing_id = df.groupby("bearing_id").size().sort_values(ascending=False).index[0]
    bearing_df = df[df["bearing_id"] == bearing_id].sort_values("time_index")
    entropy_candidates = [c for c in bearing_df.columns if c.startswith("h_") and "spectral_entropy" in c]
    entropy_col = entropy_candidates[0] if entropy_candidates else None
    columns = [("h_rms", "Horizontal RMS"), ("h_kurtosis", "Horizontal kurtosis")]
    if entropy_col:
        columns.append((entropy_col, "Horizontal spectral entropy"))
    columns.append(("normalized_rul", "Normalized RUL"))

    plt.figure(figsize=(10, 7))
    for i, (col, label) in enumerate(columns, start=1):
        if col not in bearing_df.columns:
            continue
        ax = plt.subplot(len(columns), 1, i)
        ax.plot(bearing_df["time_index"], bearing_df[col], linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if i == len(columns):
            ax.set_xlabel("Time index")
    save_current_figure(Path("results/figures/feature_curves") / f"{bearing_id}_feature_curves.png")


def prediction_files_for_split(split_name: str) -> dict[str, Path]:
    pred_dir = Path("results/predictions")
    return {
        model: pred_dir / f"{split_name}_{model}.csv"
        for model in MODEL_ORDER
        if (pred_dir / f"{split_name}_{model}.csv").exists()
    }


def available_prediction_splits() -> list[str]:
    pred_dir = Path("results/predictions")
    if not pred_dir.exists():
        return []
    split_names = set()
    for path in pred_dir.glob("*.csv"):
        stem = path.stem
        for model in sorted(MODEL_ORDER, key=len, reverse=True):
            suffix = f"_{model}"
            if stem.endswith(suffix):
                split_names.add(stem[: -len(suffix)])
                break
    return sorted(split_names)


def plot_prediction_curve() -> None:
    splits = available_prediction_splits()
    if not splits:
        print("Skipping prediction curve; no prediction CSV files found.")
        return

    selected_split = None
    files = {}
    for split_name in splits:
        candidate = prediction_files_for_split(split_name)
        if len(candidate) > len(files):
            selected_split = split_name
            files = candidate
    if selected_split is None or not files:
        print("Skipping prediction curve; no usable prediction files found.")
        return

    frames = {model: pd.read_csv(path) for model, path in files.items()}
    common_bearings = None
    for frame in frames.values():
        bearings = set(frame["bearing_id"].astype(str).unique())
        common_bearings = bearings if common_bearings is None else common_bearings & bearings
    if not common_bearings:
        print("Skipping prediction curve; prediction files share no test bearing.")
        return

    bearing_id = sorted(common_bearings)[0]
    plt.figure(figsize=(10, 5))
    first_frame = next(iter(frames.values()))
    true_df = first_frame[first_frame["bearing_id"].astype(str) == bearing_id].sort_values("time_index")
    plt.plot(true_df["time_index"], true_df["normalized_rul"], color="#222222", linewidth=2.2, label="True")
    for model in MODEL_ORDER:
        if model not in frames:
            continue
        model_df = frames[model][frames[model]["bearing_id"].astype(str) == bearing_id].sort_values("time_index")
        if model_df.empty:
            continue
        plt.plot(
            model_df["time_index"],
            model_df["y_pred"],
            linewidth=1.4,
            label=model,
            color=MODEL_COLORS.get(model),
        )
    plt.xlabel("Time index")
    plt.ylabel("Normalized RUL")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    save_current_figure(
        Path("results/figures/prediction_curves") / f"{selected_split}_{bearing_id}_prediction_curve.png"
    )


def plot_bar_chart(table_path: str, out_path: str, title: str) -> None:
    path = Path(table_path)
    if not path.exists():
        print(f"Skipping bar chart; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        print(f"Skipping bar chart; {path} is empty.")
        return
    grouped = (
        df.groupby("model", as_index=False)
        .agg(mae=("mae", "mean"), mae_std=("mae", "std"), n=("mae", "count"))
    )
    grouped["mae_std"] = grouped["mae_std"].fillna(0.0)
    grouped["order"] = grouped["model"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    grouped = grouped.sort_values(["mae", "order"])
    colors = [MODEL_COLORS.get(model, "#777777") for model in grouped["model"]]
    plt.figure(figsize=(6.8, 3.8))
    x = np.arange(len(grouped))
    yerr = grouped["mae_std"].to_numpy() if grouped["n"].max() > 1 else None
    plt.bar(x, grouped["mae"], yerr=yerr, color=colors, capsize=3, edgecolor="none")
    plt.ylabel("MAE")
    plt.title(title)
    plt.xticks(x, grouped["model"], rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(Path(out_path))


def plot_line_table(
    table_path: str,
    x_col: str,
    out_path: str,
    title: str,
    xlabel: str,
    metric: str = "mae",
) -> None:
    path = Path(table_path)
    if not path.exists():
        print(f"Skipping line plot; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty or x_col not in df.columns:
        print(f"Skipping line plot; {path} is empty or lacks {x_col}.")
        return
    plt.figure(figsize=(6.4, 3.8))
    for model, model_df in df.groupby("model", sort=False):
        model_df = model_df.sort_values(x_col)
        plt.plot(
            model_df[x_col],
            model_df[metric],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=model,
            color=MODEL_COLORS.get(model),
        )
    plt.xlabel(xlabel)
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(alpha=0.18)
    plt.legend(frameon=False, ncol=2)
    save_current_figure(Path(out_path))


def plot_conformal_tradeoff() -> None:
    path = Path("results/tables/conformal_interval_average_results.csv")
    if not path.exists():
        print(f"Skipping conformal plot; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("Skipping conformal plot; table is empty.")
        return
    plt.figure(figsize=(5.8, 4.0))
    for _, row in df.iterrows():
        model = row["model"]
        plt.scatter(
            row["avg_interval_length"],
            row["coverage"],
            s=80,
            color=MODEL_COLORS.get(model, "#777777"),
            label=model,
            alpha=0.9,
        )
        plt.text(row["avg_interval_length"], row["coverage"] + 0.006, model, fontsize=8, ha="center")
    nominal = float(df["coverage"].mean() * 0 + 0.9)
    plt.axhline(nominal, color="#222222", linewidth=1.0, linestyle="--", label="Nominal 90%")
    plt.xlabel("Average interval length")
    plt.ylabel("Empirical coverage")
    plt.ylim(0.0, 1.05)
    plt.title("Conformal interval tradeoff")
    plt.grid(alpha=0.18)
    save_current_figure(Path("results/figures/uncertainty/conformal_coverage_length_tradeoff.png"))


def plot_top_feature_scores() -> None:
    path = Path("results/tables/feature_scores.csv")
    if not path.exists():
        print(f"Skipping feature score plot; missing {path}")
        return
    df = pd.read_csv(path).head(20)
    if df.empty:
        return
    group_colors = {"time": "#4C78A8", "frequency": "#72B7B2", "wavelet": "#F58518"}
    plt.figure(figsize=(7.0, 5.0))
    ordered = df.iloc[::-1]
    colors = [group_colors.get(group, "#777777") for group in ordered["group"]]
    plt.barh(ordered["feature"], ordered["total_score"], color=colors)
    plt.xlabel("Feature score")
    plt.title("Top degradation-sensitive features")
    plt.grid(axis="x", alpha=0.18)
    save_current_figure(Path("results/figures/feature_curves/top_feature_scores.png"))


def plot_feature_group_sensitivity() -> None:
    path = Path("results/tables/feature_group_sensitivity_average_results.csv")
    if not path.exists():
        print(f"Skipping feature group plot; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    if "protocol" in df.columns:
        cross_df = df[df["protocol"] == "cross_condition"].copy()
        if not cross_df.empty:
            df = cross_df
    df = df.sort_values("mae")
    plt.figure(figsize=(6.2, 3.8))
    plt.bar(df["feature_group"], df["mae"], color="#4C78A8")
    plt.ylabel("MAE")
    plt.title("Feature group sensitivity")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.18)
    save_current_figure(Path("results/figures/bar_charts/feature_group_sensitivity_mae.png"))


def plot_feature_setting_retraining() -> None:
    path = Path("results/tables/selected_feature_retraining_average_results.csv")
    if not path.exists():
        print(f"Skipping feature-setting retraining plot; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    settings = ["original", "wavelet_only", "all_expanded", "selected_top"]
    setting_labels = ["Original", "Wavelet-only", "All-expanded", "Selected-top30"]
    models = [model for model in MODEL_ORDER if model in set(df["model"])]
    pivot = df.pivot_table(index="model", columns="feature_setting", values="mae", aggfunc="mean")
    pivot = pivot.reindex(index=models, columns=settings)

    values = pivot.to_numpy(dtype=float)
    plt.figure(figsize=(7.2, 4.4))
    image = plt.imshow(values, cmap="YlGnBu_r", aspect="auto")
    plt.colorbar(image, label="MAE")
    plt.xticks(np.arange(len(settings)), setting_labels, rotation=20, ha="right")
    plt.yticks(np.arange(len(models)), models)
    plt.title("Feature setting retraining MAE")
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                plt.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="#172033")
    save_current_figure(Path("results/figures/bar_charts/selected_feature_retraining_mae_heatmap.png"))


def parse_latent_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.endswith("_condition_aware_ode_latent"):
        return stem[: -len("_condition_aware_ode_latent")], "condition_aware_ode"
    if stem.endswith("_latent_ode_latent"):
        return stem[: -len("_latent_ode_latent")], "latent_ode"
    return stem.replace("_latent", ""), "latent"


def plot_latent_pca() -> None:
    latent_dir = Path("results/latent")
    if not latent_dir.exists():
        print("Skipping latent PCA; no latent directory found.")
        return
    latent_files = sorted(latent_dir.glob("*_latent.npz"))
    if not latent_files:
        print("Skipping latent PCA; no latent npz files found.")
        return

    for latent_file in latent_files:
        data = np.load(latent_file, allow_pickle=True)
        if "z_test" not in data or len(data["z_test"]) < 2:
            continue
        z = data["z_test"]
        if z.shape[1] < 2:
            coords = np.column_stack([z[:, 0], np.zeros(len(z))])
        else:
            coords = PCA(n_components=2, random_state=42).fit_transform(z)
        color = data["y_test"] if "y_test" in data else np.arange(len(coords))
        split_name, model = parse_latent_filename(latent_file)
        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=color, cmap="viridis", s=14, alpha=0.85)
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.title(f"{split_name} {model}")
        plt.grid(alpha=0.2)
        cbar = plt.colorbar(scatter)
        cbar.set_label("Normalized RUL")
        save_current_figure(Path("results/figures/latent_pca") / f"{split_name}_{model}_latent_pca.png")


def plot_all(features_path: str) -> None:
    setup_style()
    plot_feature_curves(features_path)
    plot_prediction_curve()
    plot_bar_chart(
        "results/tables/within_condition_results.csv",
        "results/figures/bar_charts/within_condition_mae.png",
        "Within-condition MAE",
    )
    plot_bar_chart(
        "results/tables/mixed_condition_results.csv",
        "results/figures/bar_charts/mixed_condition_mae.png",
        "Mixed-condition MAE",
    )
    plot_bar_chart(
        "results/tables/cross_condition_average_results.csv",
        "results/figures/bar_charts/cross_condition_average_mae.png",
        "Average cross-condition MAE",
    )
    plot_bar_chart(
        "results/tables/literature_aligned_average_results.csv",
        "results/figures/bar_charts/literature_aligned_average_mae.png",
        "Literature-aligned leave-one-bearing-out MAE",
    )
    plot_bar_chart(
        "results/tables/sparse_observation_average_results.csv",
        "results/figures/bar_charts/sparse_observation_average_mae.png",
        "Sparse-observation average MAE",
    )
    plot_line_table(
        "results/tables/sparse_observation_average_results.csv",
        "keep_ratio",
        "results/figures/robustness/sparse_observation_mae_curve.png",
        "Sparse-observation robustness",
        "Kept observation ratio",
    )
    plot_line_table(
        "results/tables/missing_feature_robustness_average_results.csv",
        "missing_ratio",
        "results/figures/robustness/missing_feature_mae_curve.png",
        "Missing-feature robustness",
        "Missing feature ratio",
    )
    plot_line_table(
        "results/tables/k_sensitivity_average_results.csv",
        "k",
        "results/figures/robustness/k_sensitivity_mae_curve.png",
        "Window-length sensitivity",
        "Sequence length K",
    )
    plot_conformal_tradeoff()
    plot_top_feature_scores()
    plot_feature_group_sensitivity()
    plot_feature_setting_retraining()
    plot_latent_pca()


def parse_args():
    parser = argparse.ArgumentParser(description="Create experiment figures.")
    parser.add_argument("--all", action="store_true", help="Create all required plots.")
    parser.add_argument("--features", default="processed/features.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all:
        plot_all(args.features)
    else:
        plot_all(args.features)


if __name__ == "__main__":
    main()
