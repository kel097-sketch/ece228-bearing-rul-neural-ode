import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


def ordered_models(models: list[str]) -> list[str]:
    return [m for m in ACTIVE_MODEL_ORDER if m in models] + [m for m in models if m not in ACTIVE_MODEL_ORDER]


def bearing_sort_key(bearing_id: str) -> tuple[int, int]:
    # C3_Bearing3_1 -> (3, 1)
    tail = str(bearing_id).split("Bearing")[-1]
    condition, bearing = tail.split("_")
    return int(condition), int(bearing)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def save_table_image(df: pd.DataFrame, path: Path, title: str, font_size: int = 8, scale_y: float = 1.35) -> None:
    fig_height = max(2.6, 0.38 * len(df) + 1.35)
    fig_width = max(8.0, 1.22 * len(df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, scale_y)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DFEF")
        if row == 0:
            cell.set_text_props(weight="bold", color="#102033")
            cell.set_facecolor("#EAF0FB")
        else:
            cell.set_facecolor("#FFFFFF")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#102033", pad=14)
    plt.tight_layout()
    ensure_dir(path.parent)
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def mean_std_text(mean: float, std: float | None) -> str:
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def make_protocol_summary_tables(protocol_summary: pd.DataFrame, table_dir: Path, fig_dir: Path) -> None:
    rows = []
    for protocol in ["cross_condition", "mixed_condition"]:
        sub = protocol_summary[protocol_summary["protocol"] == protocol].copy()
        if sub.empty:
            continue
        sub["model"] = pd.Categorical(sub["model"], categories=ordered_models(sub["model"].unique().tolist()), ordered=True)
        sub = sub.sort_values("model")
        for _, row in sub.iterrows():
            rows.append(
                {
                    "Protocol": protocol.replace("_", "-"),
                    "Model": row["model"],
                    "MAE": mean_std_text(row["mae_mean"], row["mae_std"]),
                    "RMSE": mean_std_text(row["rmse_mean"], row["rmse_std"]),
                    "R2": mean_std_text(row["r2_mean"], row["r2_std"]),
                    "Spearman": mean_std_text(row["spearman_mean"], row["spearman_std"]),
                    "Late MAE": mean_std_text(row["late_mae_mean"], row["late_mae_std"]),
                    "Violation": mean_std_text(
                        row["monotonic_violation_rate_mean"], row["monotonic_violation_rate_std"]
                    ),
                }
            )
    summary = pd.DataFrame(rows)
    write_csv(summary, table_dir / "paper_style_protocol_summary.csv")
    save_table_image(
        summary,
        fig_dir / "paper_style_protocol_summary_table.png",
        "Protocol-level model comparison",
        font_size=7,
        scale_y=1.25,
    )


def make_tcn_per_bearing_table(per_bearing: pd.DataFrame, table_dir: Path, fig_dir: Path) -> None:
    data = per_bearing[(per_bearing["protocol"] == "cross_condition") & (per_bearing["model"] == "TCN")].copy()
    data = data.sort_values("bearing_id", key=lambda s: s.map(bearing_sort_key))
    table = data[
        ["bearing_id", "condition_id", "mae", "rmse", "r2", "spearman", "late_mae", "monotonic_violation_rate"]
    ].rename(
        columns={
            "bearing_id": "Test bearing",
            "condition_id": "Condition",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R2",
            "spearman": "Spearman",
            "late_mae": "Late MAE",
            "monotonic_violation_rate": "Violation",
        }
    )
    for col in ["MAE", "RMSE", "R2", "Spearman", "Late MAE", "Violation"]:
        table[col] = table[col].map(lambda x: f"{x:.3f}")
    write_csv(table, table_dir / "paper_style_tcn_per_bearing_metrics.csv")
    save_table_image(
        table,
        fig_dir / "paper_style_tcn_per_bearing_table.png",
        "Per-bearing metrics for the main TCN model",
        font_size=7,
        scale_y=1.22,
    )


def make_metric_panel(protocol_summary: pd.DataFrame, fig_dir: Path) -> None:
    data = protocol_summary[protocol_summary["protocol"] == "cross_condition"].copy()
    if data.empty:
        return
    order = ordered_models(data["model"].unique().tolist())
    data["model"] = pd.Categorical(data["model"], categories=order, ordered=True)
    data = data.sort_values("model")
    metrics = [
        ("mae_mean", "MAE", "lower is better"),
        ("rmse_mean", "RMSE", "lower is better"),
        ("r2_mean", "R2", "higher is better"),
        ("spearman_mean", "Spearman", "higher is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for ax, (metric, label, subtitle) in zip(axes.ravel(), metrics, strict=False):
        values = data[metric].to_numpy(dtype=float)
        colors = [MODEL_COLORS.get(str(m), "#777777") for m in data["model"]]
        ax.bar(np.arange(len(data)), values, color=colors)
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(data["model"].astype(str), rotation=25, ha="right")
        ax.set_title(f"{label} ({subtitle})", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        if metric == "r2_mean":
            ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
        for idx, value in enumerate(values):
            va = "bottom" if value >= 0 else "top"
            ax.text(idx, value, f"{value:.3f}", ha="center", va=va, fontsize=8)
    fig.suptitle("Cross-condition metric comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = fig_dir / "paper_style_cross_condition_metric_panels.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_prediction_band_grid(predictions: pd.DataFrame, fig_dir: Path, model: str = "TCN") -> None:
    cases = [
        ("cross_train_C2_C3_test_C1", "C1_Bearing1_1", "Held-out C1"),
        ("cross_train_C1_C3_test_C2", "C2_Bearing2_1", "Held-out C2"),
        ("cross_train_C1_C2_test_C3", "C3_Bearing3_1", "Held-out C3"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), sharey=True)
    for ax, (split_name, bearing_id, title) in zip(axes, cases, strict=False):
        one = predictions[
            (predictions["protocol"] == "cross_condition")
            & (predictions["split_name"] == split_name)
            & (predictions["bearing_id"] == bearing_id)
            & (predictions["model"] == model)
        ].sort_values("time_index")
        if one.empty:
            ax.set_axis_off()
            continue
        x = np.linspace(0, 100, len(one))
        target = one["normalized_rul"].to_numpy(dtype=float)
        pred = one["y_pred"].to_numpy(dtype=float)
        lower = np.clip(target - 0.15, 0.0, 1.0)
        upper = np.clip(target + 0.15, 0.0, 1.0)
        ax.fill_between(x, lower, upper, color="#DDEBFF", alpha=0.8, label="+/- 15% band")
        ax.plot(x, target, color="#222222", linewidth=1.4, linestyle="--", label="constructed target")
        ax.plot(x, pred, color=MODEL_COLORS.get(model, "#F58518"), linewidth=1.3, label=f"{model} prediction")
        ax.set_title(f"{title}\n{bearing_id}", fontsize=10)
        ax.set_xlabel("Life percentage (%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized RUL")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Paper-style representative prediction curves", fontsize=15, fontweight="bold", y=1.20)
    plt.tight_layout()
    path = fig_dir / "paper_style_prediction_curves_with_error_band.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def make_r2_heatmap(per_bearing: pd.DataFrame, fig_dir: Path) -> None:
    data = per_bearing[per_bearing["protocol"] == "cross_condition"].copy()
    data["bearing_order"] = data["bearing_id"].map(bearing_sort_key)
    bearings = sorted(data["bearing_id"].unique(), key=bearing_sort_key)
    models = ordered_models(data["model"].unique().tolist())
    pivot = data.pivot_table(index="model", columns="bearing_id", values="r2", aggfunc="mean")
    pivot = pivot.loc[[m for m in models if m in pivot.index], bearings]
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    vmax = max(abs(np.nanmin(pivot.to_numpy())), abs(np.nanmax(pivot.to_numpy())), 0.5)
    image = ax.imshow(pivot.to_numpy(), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(bearings)))
    ax.set_xticklabels(bearings, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#102033")
    fig.colorbar(image, ax=ax, label="R2")
    ax.set_title("Per-bearing R2 under cross-condition shift", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "paper_style_per_bearing_r2_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def make_feature_setting_heatmap(table_dir: Path, fig_dir: Path) -> None:
    path = table_dir / "final_feature_setting" / "selected_feature_retraining_average_results.csv"
    if not path.exists():
        print(f"Skipping feature-setting table; missing {path}")
        return
    df = pd.read_csv(path)
    df = df[df["protocol"] == "cross_condition"].copy()
    settings = ["original", "wavelet_only", "all_expanded", "selected_top"]
    labels = ["Original", "Wavelet-only", "All-expanded", "Selected-top30"]
    models = ordered_models(df["model"].unique().tolist())
    pivot = df.pivot_table(index="model", columns="feature_setting", values="mae", aggfunc="mean")
    pivot = pivot.loc[[m for m in models if m in pivot.index], [s for s in settings if s in pivot.columns]]
    pivot.columns = labels[: len(pivot.columns)]
    pivot.to_csv(table_dir / "paper_aligned" / "paper_style_feature_setting_mae_pivot.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j]:.3f}", ha="center", va="center", fontsize=9, color="#102033")
    fig.colorbar(image, ax=ax, label="MAE")
    ax.set_title("Feature-setting retraining MAE", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = fig_dir / "paper_style_feature_setting_retraining_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def make_outputs(args) -> None:
    table_dir = ensure_dir(Path(args.table_dir))
    fig_dir = ensure_dir(Path(args.fig_dir))
    final_dir = Path(args.final_table_dir)
    protocol_summary = pd.read_csv(final_dir / "final_protocol_summary.csv")
    per_bearing = pd.read_csv(final_dir / "final_per_bearing_metrics.csv")
    predictions = pd.read_csv(final_dir / "final_predictions_used.csv")

    paper_table_dir = ensure_dir(table_dir / "paper_aligned")
    make_protocol_summary_tables(protocol_summary, paper_table_dir, fig_dir)
    make_tcn_per_bearing_table(per_bearing, paper_table_dir, fig_dir)
    make_metric_panel(protocol_summary, fig_dir)
    make_prediction_band_grid(predictions, fig_dir, model=args.main_model)
    make_r2_heatmap(per_bearing, fig_dir)
    make_feature_setting_heatmap(Path(args.table_dir), fig_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-aligned RUL tables and figures.")
    parser.add_argument("--final_table_dir", default="results/tables/final")
    parser.add_argument("--table_dir", default="results/tables")
    parser.add_argument("--fig_dir", default="results/figures/paper_aligned")
    parser.add_argument("--main_model", default="TCN")
    return parser.parse_args()


def main():
    make_outputs(parse_args())


if __name__ == "__main__":
    main()
