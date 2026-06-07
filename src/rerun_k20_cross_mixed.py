import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from make_sequences import make_sequence_file
from mixed5fold_experiment import (
    MODEL_COLORS,
    compute_metrics,
    ordered_models,
    select_feature_frame,
    train_neural_predictions,
    train_ridge_predictions,
)
from utils import ensure_dir, load_split, set_seed


def load_protocol_splits(split_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(split_dir.glob("*.json")):
        split = load_split(path)
        if split.get("protocol") in {"cross_condition", "mixed_condition"}:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No cross_condition or mixed_condition splits found in {split_dir}")
    return paths


def save_protocol_bar(summary: pd.DataFrame, fig_dir: Path, variant: str) -> None:
    data = summary[summary["prediction_variant"] == variant].copy()
    if data.empty:
        return
    models = ordered_models(data["model"].unique().tolist())
    protocols = ["cross_condition", "mixed_condition"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=180, sharey=True)
    for ax, protocol in zip(axes, protocols, strict=False):
        sub = data[data["protocol"] == protocol].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=models, ordered=True)
        sub = sub.sort_values("model")
        x = np.arange(len(sub))
        colors = [MODEL_COLORS.get(str(model), "#64748b") for model in sub["model"]]
        yerr = sub["mae_std"].fillna(0.0).to_numpy()
        ax.bar(x, sub["mae_mean"], yerr=yerr, color=colors, capsize=4)
        ax.set_title("Cross-condition" if protocol == "cross_condition" else "Mixed-condition")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].astype(str), rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0.0, max(0.34, float(data["mae_mean"].max() + data["mae_std"].fillna(0).max() + 0.04)))
    axes[0].set_ylabel("Bearing-level MAE")
    fig.suptitle(f"K=20 rerun MAE ({variant})", fontsize=14)
    fig.tight_layout()
    path = fig_dir / f"rerun_k20_cross_mixed_mae_{variant}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_key_table(summary: pd.DataFrame, out_dir: Path, fig_dir: Path, variant: str) -> None:
    data = summary[summary["prediction_variant"] == variant].copy()
    rows = []
    for protocol in ["cross_condition", "mixed_condition"]:
        sub = data[data["protocol"] == protocol].sort_values("mae_mean")
        for _, row in sub.iterrows():
            rows.append(
                {
                    "Protocol": "Cross" if protocol == "cross_condition" else "Mixed",
                    "Model": row["model"],
                    "MAE": f"{row['mae_mean']:.3f}",
                    "RMSE": f"{row['rmse_mean']:.3f}",
                    "Spearman": f"{row['spearman_mean']:.3f}",
                    "Late MAE": f"{row['late_mae_mean']:.3f}",
                }
            )
    table = pd.DataFrame(rows)
    table_path = out_dir / f"rerun_k20_summary_for_ppt_{variant}.csv"
    table.to_csv(table_path, index=False)

    fig_h = max(4.5, 0.38 * len(table) + 1.3)
    fig, ax = plt.subplots(figsize=(12.5, fig_h), dpi=180)
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#e8f0fb")
            cell.set_text_props(weight="bold", color="#0b1f3d")
        elif c == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold")
    ax.set_title(f"K=20 cross/mixed rerun summary ({variant})", fontsize=15, weight="bold", pad=16)
    fig.tight_layout()
    fig_path = fig_dir / f"rerun_k20_key_results_table_{variant}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {table_path}")
    print(f"Saved {fig_path}")


def run(args) -> None:
    set_seed(args.seed)
    features_df = pd.read_csv(args.features)
    split_paths = load_protocol_splits(Path(args.split_dir))
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    pred_dir = ensure_dir(args.prediction_dir)
    seq_dir = ensure_dir(args.sequence_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_args = SimpleNamespace(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        smooth_weight=args.smooth_weight,
    )

    all_predictions = []
    train_info = []
    for split_path in split_paths:
        split = load_split(split_path)
        setting_df, feature_cols = select_feature_frame(features_df, args.feature_setting, split, args.top_k)
        print(f"Running {split['protocol']} / {split['split_name']} with {len(feature_cols)} {args.feature_setting} features")

        ridge_pred, ridge_info = train_ridge_predictions(setting_df, split, feature_cols, clip=not args.no_clip)
        ridge_pred.to_csv(pred_dir / f"{split['split_name']}_Ridge.csv", index=False)
        all_predictions.append(ridge_pred)
        train_info.extend([{**item, "split_name": split["split_name"], "protocol": split["protocol"]} for item in ridge_info])

        seq_path = seq_dir / f"{split['split_name']}_k{args.k}.npz"
        if args.rebuild_sequences or not seq_path.exists():
            seq_path = make_sequence_file(setting_df, split_path, args.k, seq_dir)

        for model_name in args.models:
            if model_name == "Ridge":
                continue
            seed = args.seed + sum(ord(ch) for ch in f"{split['split_name']}:{model_name}:{args.feature_setting}:k{args.k}")
            print(f"Training {model_name} on {split['split_name']} using {device}")
            pred_frame, info = train_neural_predictions(seq_path, model_name, train_args, device, seed, clip=not args.no_clip)
            pred_frame.to_csv(pred_dir / f"{split['split_name']}_{model_name}.csv", index=False)
            all_predictions.append(pred_frame)
            train_info.extend(
                [{**item, "split_name": split["split_name"], "protocol": split["protocol"], "seed": seed} for item in info]
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out_dir / "rerun_k20_predictions.csv", index=False)
    pd.DataFrame(train_info).to_csv(out_dir / "rerun_k20_training_info.csv", index=False)
    per_bearing, split_summary, protocol_summary = compute_metrics(predictions, args.epsilon)
    per_bearing.to_csv(out_dir / "rerun_k20_per_bearing_metrics.csv", index=False)
    split_summary.to_csv(out_dir / "rerun_k20_split_metrics.csv", index=False)
    protocol_summary.to_csv(out_dir / "rerun_k20_protocol_summary.csv", index=False)
    for variant in ["raw", "val_calibrated"]:
        save_protocol_bar(protocol_summary, fig_dir, variant)
        save_key_table(protocol_summary, out_dir, fig_dir, variant)
    print(f"Saved rerun outputs to {out_dir} and {fig_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun K=20 cross-condition and mixed-condition training.")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--feature_setting", default="original", choices=["original", "wavelet_only", "all_expanded", "selected_top"])
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--split_dir", default="processed/splits_final")
    parser.add_argument("--sequence_dir", default="processed/sequences_rerun_k20_cross_mixed")
    parser.add_argument("--prediction_dir", default="results/predictions_rerun_k20_cross_mixed")
    parser.add_argument("--out_dir", default="results/tables/rerun_k20_cross_mixed")
    parser.add_argument("--fig_dir", default="results/figures/rerun_k20_cross_mixed")
    parser.add_argument("--models", nargs="+", default=["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--rebuild_sequences", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
