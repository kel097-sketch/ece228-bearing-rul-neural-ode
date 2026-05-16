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


def save_current_figure(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
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
    grouped = df.groupby("model", as_index=False)["mae"].mean()
    grouped["order"] = grouped["model"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    grouped = grouped.sort_values(["mae", "order"])
    colors = [MODEL_COLORS.get(model, "#777777") for model in grouped["model"]]
    plt.figure(figsize=(8, 4.5))
    plt.bar(grouped["model"], grouped["mae"], color=colors)
    plt.ylabel("MAE")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    save_current_figure(Path(out_path))


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
