import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from config import ACTIVE_MODEL_ORDER
from utils import ensure_dir


MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "LSTM": "#54A24B",
    "TCN": "#F58518",
    "Transformer": "#72B7B2",
    "latent_ode": "#B279A2",
}


def ordered_models(values: list[str]) -> list[str]:
    return [model for model in ACTIVE_MODEL_ORDER if model in values] + [
        model for model in values if model not in ACTIVE_MODEL_ORDER
    ]


def save_feature_ablation(table_dir: Path, fig_dir: Path) -> None:
    path = table_dir / "final_feature_ablation" / "feature_group_sensitivity_average_results.csv"
    if not path.exists():
        print(f"Skipping feature ablation plot; missing {path}")
        return
    df = pd.read_csv(path)
    df = df[df["protocol"] == "cross_condition"].copy()
    if df.empty:
        return
    order = ["wavelet", "time", "selected_top", "frequency", "original", "all"]
    df["feature_group"] = pd.Categorical(df["feature_group"], categories=order, ordered=True)
    df = df.sort_values("feature_group")
    plt.figure(figsize=(7.0, 4.0))
    plt.bar(df["feature_group"].astype(str), df["mae"], color="#4C78A8")
    plt.ylabel("Ridge MAE")
    plt.title("Feature group ablation under cross-condition")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out = fig_dir / "final_feature_group_ablation.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def save_split_diagram(split_dir: Path, fig_dir: Path) -> None:
    split_paths = sorted(split_dir.glob("*.json"))
    if not split_paths:
        print(f"Skipping split diagram; missing split files in {split_dir}")
        return
    rows = []
    for path in split_paths:
        split = pd.read_json(path, typ="series").to_dict()
        rows.append(
            [
                split["split_name"].replace("cross_train_", "").replace("_test_", " -> "),
                ", ".join(split["train_conditions"]),
                ", ".join(split["test_conditions"]),
                f"{len(split['train_bearings'])}/{len(split['val_bearings'])}/{len(split['test_bearings'])}",
            ]
        )
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Split", "Train conditions", "Test conditions", "Bearings train/val/test"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DFEF")
        if r == 0:
            cell.set_text_props(weight="bold", color="#102033")
            cell.set_facecolor("#EAF0FB")
        else:
            cell.set_facecolor("#FFFFFF")
    plt.title("Final bearing-level split protocol", fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    out = fig_dir / "final_dataset_split_diagram.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def draw_box(ax, xy, text, width=1.8, height=0.72, face="#FFFFFF", edge="#4C78A8"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9, color="#102033")
    return box


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color="#60708A",
        )
    )


def save_feature_pipeline(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    draw_box(ax, (0.35, 1.65), "Raw vibration\nH/V channels", face="#F4F7FB")
    draw_box(ax, (2.5, 2.55), "Original features\n52 time/frequency", face="#EDF7ED", edge="#54A24B")
    draw_box(ax, (2.5, 0.75), "Wavelet features\n72 WPT/DWT", face="#FFF5E6", edge="#F58518")
    draw_box(ax, (5.0, 2.55), "Original", width=1.45, face="#FFFFFF")
    draw_box(ax, (5.0, 1.65), "Wavelet-only", width=1.45, face="#FFFFFF")
    draw_box(ax, (5.0, 0.75), "All-expanded\n124", width=1.45, face="#FFFFFF")
    draw_box(ax, (7.15, 1.65), "Train-only\nSelected-top30", width=1.75, face="#F4F7FB", edge="#B279A2")
    draw_box(ax, (9.35, 1.65), "Ridge / LSTM / TCN\nTransformer / Latent ODE", width=2.15, face="#FFFFFF", edge="#1F5EFF")
    arrow(ax, (2.15, 2.0), (2.5, 2.9))
    arrow(ax, (2.15, 2.0), (2.5, 1.1))
    arrow(ax, (4.3, 2.9), (5.0, 2.9))
    arrow(ax, (4.3, 1.1), (5.0, 1.1))
    arrow(ax, (6.45, 2.9), (9.35, 2.25))
    arrow(ax, (6.45, 2.0), (9.35, 2.0))
    arrow(ax, (6.45, 1.1), (7.15, 2.0))
    arrow(ax, (8.9, 2.0), (9.35, 2.0))
    ax.text(6.1, 3.55, "Feature settings", ha="center", fontsize=12, fontweight="bold")
    ax.text(10.42, 2.72, "RUL prediction", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = fig_dir / "final_feature_extraction_pipeline.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def save_feature_setting_heatmap(table_dir: Path, fig_dir: Path) -> None:
    path = table_dir / "final_feature_setting" / "selected_feature_retraining_average_results.csv"
    if not path.exists():
        print(f"Skipping feature setting heatmap; missing {path}")
        return
    df = pd.read_csv(path)
    df = df[df["protocol"] == "cross_condition"].copy()
    if df.empty:
        return
    settings = ["original", "wavelet_only", "all_expanded", "selected_top"]
    models = ordered_models(df["model"].unique().tolist())
    pivot = df.pivot_table(index="model", columns="feature_setting", values="mae", aggfunc="mean")
    pivot = pivot.loc[[model for model in models if model in pivot.index], [s for s in settings if s in pivot.columns]]
    plt.figure(figsize=(8.0, 4.6))
    image = plt.imshow(pivot.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    plt.xticks(np.arange(len(pivot.columns)), ["Original", "Wavelet-only", "All-expanded", "Selected-top30"], rotation=20, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.iat[i, j]:.3f}", ha="center", va="center", color="#102033", fontsize=9)
    plt.colorbar(image, label="MAE")
    plt.title("Feature-setting retraining MAE")
    plt.tight_layout()
    out = fig_dir / "final_feature_setting_retraining_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def save_uncertainty_tradeoff(table_dir: Path, fig_dir: Path) -> None:
    path = table_dir / "final_uncertainty" / "conformal_interval_average_results.csv"
    if not path.exists():
        print(f"Skipping uncertainty plot; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    plt.figure(figsize=(6.6, 4.2))
    for _, row in df.iterrows():
        model = row["model"]
        plt.scatter(
            row["avg_interval_length"],
            row["coverage"],
            s=90,
            color=MODEL_COLORS.get(model, "#777777"),
            label=model,
            alpha=0.9,
        )
        plt.text(row["avg_interval_length"], row["coverage"] + 0.006, model, ha="center", fontsize=8)
    plt.axhline(0.9, color="#333333", linestyle="--", linewidth=1.0, label="90% target")
    plt.xlabel("Average interval length")
    plt.ylabel("Empirical coverage")
    plt.title("Conformal interval coverage-width tradeoff")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out = fig_dir / "final_conformal_coverage_width.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def save_missing_robustness(table_dir: Path, fig_dir: Path) -> None:
    path = table_dir / "final_robustness" / "missing_feature_robustness_average_results.csv"
    if not path.exists():
        print(f"Skipping missing-feature plot; missing {path}")
        return
    df = pd.read_csv(path)
    df = df[df["protocol"] == "cross_condition"].copy() if "protocol" in df.columns else df
    if df.empty:
        return
    plt.figure(figsize=(7.0, 4.0))
    for model in ordered_models(df["model"].unique().tolist()):
        one = df[df["model"] == model].sort_values("missing_ratio")
        plt.plot(one["missing_ratio"], one["mae"], marker="o", label=model, color=MODEL_COLORS.get(model))
    plt.xlabel("Missing feature ratio")
    plt.ylabel("MAE")
    plt.title("Missing-feature robustness")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False, ncols=2, fontsize=8)
    plt.tight_layout()
    out = fig_dir / "final_missing_feature_robustness.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def save_sparse_robustness(table_dir: Path, fig_dir: Path) -> None:
    path = table_dir / "final_robustness" / "sparse_observation_average_results.csv"
    if not path.exists():
        print(f"Skipping sparse-observation plot; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    plt.figure(figsize=(7.0, 4.0))
    for model in ordered_models(df["model"].unique().tolist()):
        one = df[df["model"] == model].sort_values("keep_ratio")
        plt.plot(one["keep_ratio"], one["mae"], marker="o", label=model, color=MODEL_COLORS.get(model))
    plt.xlabel("Kept observation ratio")
    plt.ylabel("MAE")
    plt.title("Sparse-observation robustness")
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.2)
    plt.legend(frameon=False, ncols=2, fontsize=8)
    plt.tight_layout()
    out = fig_dir / "final_sparse_observation_robustness.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def save_multiseed(table_dir: Path, fig_dir: Path) -> None:
    path = table_dir / "final_multiseed" / "multiseed_model_average_results.csv"
    if not path.exists():
        print(f"Skipping multi-seed plot; missing {path}")
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    models = ordered_models(df["model"].unique().tolist())
    df["model"] = pd.Categorical(df["model"], categories=models, ordered=True)
    df = df.sort_values("model")
    plt.figure(figsize=(6.8, 4.0))
    x = np.arange(len(df))
    colors = [MODEL_COLORS.get(str(model), "#777777") for model in df["model"]]
    plt.bar(x, df["mae_mean"], yerr=df["mae_std"].fillna(0.0), color=colors, capsize=4)
    plt.xticks(x, df["model"].astype(str), rotation=25, ha="right")
    plt.ylabel("MAE")
    plt.title("Multi-seed compact stability check")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out = fig_dir / "final_multiseed_mae.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final supplementary figures from completed experiment tables.")
    parser.add_argument("--table_dir", default="results/tables")
    parser.add_argument("--fig_dir", default="results/figures/final")
    parser.add_argument("--split_dir", default="processed/splits_final")
    return parser.parse_args()


def main():
    args = parse_args()
    table_dir = Path(args.table_dir)
    fig_dir = ensure_dir(args.fig_dir)
    save_split_diagram(Path(args.split_dir), fig_dir)
    save_feature_pipeline(fig_dir)
    save_feature_ablation(table_dir, fig_dir)
    save_feature_setting_heatmap(table_dir, fig_dir)
    save_uncertainty_tradeoff(table_dir, fig_dir)
    save_missing_robustness(table_dir, fig_dir)
    save_sparse_robustness(table_dir, fig_dir)
    save_multiseed(table_dir, fig_dir)


if __name__ == "__main__":
    main()
