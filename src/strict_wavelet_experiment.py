import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn

from k_sensitivity import model_instance, predict_discrete, predict_ode, tensor_loader
from make_sequences import make_sequence_file
from utils import ensure_dir, get_feature_columns, load_split, set_seed


MODEL_ORDER = ["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"]
MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "LSTM": "#54A24B",
    "TCN": "#F58518",
    "Transformer": "#72B7B2",
    "latent_ode": "#B279A2",
}


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    value = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def monotonic_violation_rate(y_pred: np.ndarray, epsilon: float) -> float:
    if len(y_pred) < 2:
        return 0.0
    return float(np.mean(np.diff(y_pred) > epsilon))


def per_bearing_metrics(meta: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(meta, dtype=object), columns=["bearing_id", "condition_id", "time_index"])
    frame["time_index"] = frame["time_index"].astype(int)
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
                "monotonic_violation_rate": monotonic_violation_rate(yp, epsilon),
            }
        )
    return pd.DataFrame(rows)


def aggregate_bearing_metrics(metrics: pd.DataFrame) -> dict:
    cols = ["mae", "rmse", "spearman", "late_mae", "monotonic_violation_rate"]
    return {col: float(metrics[col].mean()) for col in cols}


def load_sequence(path: Path) -> tuple[dict, dict]:
    npz = np.load(path, allow_pickle=True)
    split = json.loads(npz["split_json"].item())
    data = {
        "X_train": npz["X_train"].astype(np.float32),
        "y_train": npz["y_train"].astype(np.float32),
        "X_val": npz["X_val"].astype(np.float32),
        "y_val": npz["y_val"].astype(np.float32),
        "X_test": npz["X_test"].astype(np.float32),
        "y_test": npz["y_test"].astype(np.float32),
        "c_train": (npz["c_norm_train"] if "c_norm_train" in npz else npz["c_train"]).astype(np.float32),
        "c_val": (npz["c_norm_val"] if "c_norm_val" in npz else npz["c_val"]).astype(np.float32),
        "c_test": (npz["c_norm_test"] if "c_norm_test" in npz else npz["c_test"]).astype(np.float32),
        "tau_train": npz["tau_train"].astype(np.float32),
        "tau_val": npz["tau_val"].astype(np.float32),
        "tau_test": npz["tau_test"].astype(np.float32),
        "meta_val": npz["meta_val"],
        "meta_test": npz["meta_test"],
        "feature_names": npz["feature_names"],
    }
    return data, split


@torch.no_grad()
def predict_model(model: nn.Module, model_name: str, data: dict, split_name: str, batch_size: int, device: torch.device):
    if split_name == "val":
        X = data["X_val"]
        c = data["c_val"]
        tau = data["tau_val"]
    elif split_name == "test":
        X = data["X_test"]
        c = data["c_test"]
        tau = data["tau_test"]
    else:
        raise ValueError(split_name)
    if model_name == "latent_ode":
        return predict_ode(model, X, c, tau, batch_size, device)
    return predict_discrete(model, X, batch_size, device)


def train_sequence_model(seq_path: Path, model_name: str, train_args, device: torch.device, seed: int) -> tuple[dict, pd.DataFrame, np.ndarray]:
    data, split = load_sequence(seq_path)
    if len(data["X_train"]) == 0 or len(data["X_val"]) == 0 or len(data["X_test"]) == 0:
        raise RuntimeError(f"Empty train/val/test sequence set: {seq_path}")
    set_seed(seed)
    model = model_instance(model_name, data["X_train"].shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)
    loss_fn = nn.MSELoss()
    if model_name == "latent_ode":
        train_loader = tensor_loader(
            (data["X_train"], data["y_train"], data["c_train"], data["tau_train"]),
            train_args.batch_size,
            True,
            device,
            seed,
        )
    else:
        train_loader = tensor_loader((data["X_train"], data["y_train"]), train_args.batch_size, True, device, seed)

    best_state = copy.deepcopy(model.state_dict())
    best_val_mae = float("inf")
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, train_args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if model_name == "latent_ode":
                xb, yb, cb, tb = batch
                pred, z_traj = model(xb, cb, tb)
                smooth_loss = torch.mean((z_traj[1:] - z_traj[:-1]) ** 2)
                loss = loss_fn(pred, yb) + train_args.smooth_weight * smooth_loss
            else:
                xb, yb = batch
                loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_pred = predict_model(model, model_name, data, "val", train_args.batch_size, device)
        val_metrics = aggregate_bearing_metrics(per_bearing_metrics(data["meta_val"], data["y_val"], val_pred, train_args.epsilon))
        val_mae = val_metrics["mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_args.patience:
                break

    model.load_state_dict(best_state)
    val_pred = predict_model(model, model_name, data, "val", train_args.batch_size, device)
    val_per_bearing = per_bearing_metrics(data["meta_val"], data["y_val"], val_pred, train_args.epsilon)
    val_summary = aggregate_bearing_metrics(val_per_bearing)
    val_summary.update(
        {
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "n_train_sequences": int(len(data["y_train"])),
            "n_val_sequences": int(len(data["y_val"])),
            "n_test_sequences": int(len(data["y_test"])),
        }
    )
    test_pred = predict_model(model, model_name, data, "test", train_args.batch_size, device)
    return val_summary, val_per_bearing, test_pred


def prediction_frame(meta: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, split: dict, model: str, feature_setting: str, k_value) -> pd.DataFrame:
    out = pd.DataFrame(np.asarray(meta, dtype=object), columns=["bearing_id", "condition_id", "time_index"])
    out["time_index"] = out["time_index"].astype(int)
    out["y_true"] = np.asarray(y_true, dtype=float)
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out["protocol"] = split["protocol"]
    out["split_name"] = split["split_name"]
    out["model"] = model
    out["feature_setting"] = feature_setting
    out["K"] = "NA" if k_value is None else int(k_value)
    return out.sort_values(["bearing_id", "time_index"]).reset_index(drop=True)


def cross_split_paths(split_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(split_dir.glob("*.json")):
        split = load_split(path)
        if split.get("protocol") == "cross_condition":
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No cross_condition splits found in {split_dir}")
    return paths


def all_eval_split_paths(split_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(split_dir.glob("*.json")):
        split = load_split(path)
        if split.get("protocol") in {"cross_condition", "mixed_condition"}:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No evaluation splits found in {split_dir}")
    return paths


def append_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def run_k_sensitivity(args) -> int:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    sequence_root = ensure_dir(args.sequence_root)
    features_df = pd.read_csv(args.features)
    split_paths = cross_split_paths(Path(args.split_dir))
    result_path = out_dir / "k_sensitivity_wavelet_validation.csv"
    if args.reset and result_path.exists():
        result_path.unlink()

    train_args = SimpleNamespace(
        epochs=args.k_epochs,
        patience=args.k_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        smooth_weight=args.smooth_weight,
        epsilon=args.epsilon,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in args.k_values:
        k_dir = ensure_dir(sequence_root / f"k{k}")
        for split_path in split_paths:
            split = load_split(split_path)
            seq_path = k_dir / f"{split['split_name']}_k{k}.npz"
            if args.rebuild_sequences or not seq_path.exists():
                seq_path = make_sequence_file(features_df, split_path, k, k_dir)
            for model_name in args.models:
                if result_path.exists():
                    existing = pd.read_csv(result_path)
                    mask = (
                        (existing["K"] == k)
                        & (existing["split_name"] == split["split_name"])
                        & (existing["model"] == model_name)
                    )
                    if mask.any():
                        print(f"Skipping existing K={k}, split={split['split_name']}, model={model_name}")
                        continue
                seed = args.seed + sum(ord(ch) for ch in f"{k}:{split['split_name']}:{model_name}:wavelet")
                print(f"K sensitivity: K={k}, split={split['split_name']}, model={model_name}, device={device}", flush=True)
                val_summary, _, _ = train_sequence_model(seq_path, model_name, train_args, device, seed)
                row = {
                    "feature_setting": "wavelet_only",
                    "K": k,
                    "protocol": split["protocol"],
                    "split_name": split["split_name"],
                    "model": model_name,
                    **val_summary,
                }
                append_row(result_path, row)

    results = pd.read_csv(result_path)
    avg = (
        results.groupby(["K", "model"], as_index=False)
        .agg(
            val_MAE=("mae", "mean"),
            val_RMSE=("rmse", "mean"),
            val_Spearman=("spearman", "mean"),
            val_LateMAE=("late_mae", "mean"),
            mae_std=("mae", "std"),
            n_train_sequences=("n_train_sequences", "mean"),
            n_val_sequences=("n_val_sequences", "mean"),
            num_splits=("split_name", "nunique"),
        )
        .sort_values(["K", "val_MAE"])
    )
    avg.to_csv(out_dir / "k_sensitivity_wavelet_validation_average.csv", index=False)
    k_summary = (
        avg.groupby("K", as_index=False)
        .agg(
            avg_val_MAE=("val_MAE", "mean"),
            avg_val_RMSE=("val_RMSE", "mean"),
            avg_val_Spearman=("val_Spearman", "mean"),
            avg_val_LateMAE=("val_LateMAE", "mean"),
        )
        .sort_values("avg_val_MAE")
    )
    k_summary.to_csv(out_dir / "k_selection_summary.csv", index=False)
    best_k = int(k_summary.iloc[0]["K"])
    (out_dir / "selected_k.txt").write_text(str(best_k), encoding="utf-8")
    plot_k_sensitivity(avg, k_summary, fig_dir)
    print(f"Selected K={best_k} by average validation MAE", flush=True)
    return best_k


def subset_by_bearings(df: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    return df[df["bearing_id"].isin(bearings)].copy()


def train_ridge_final(features_df: pd.DataFrame, split: dict, feature_cols: list[str], args) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = subset_by_bearings(features_df, split["train_bearings"])
    val_df = subset_by_bearings(features_df, split["val_bearings"])
    test_df = subset_by_bearings(features_df, split["test_bearings"])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_val = val_df["normalized_rul"].to_numpy(dtype=np.float32)
    best = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred_val = np.clip(model.predict(X_val), 0.0, 1.0)
        val_meta = val_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object)
        val_metrics = aggregate_bearing_metrics(per_bearing_metrics(val_meta, y_val, pred_val, args.epsilon))
        if best is None or val_metrics["mae"] < best["mae"]:
            best = {"model": model, "alpha": alpha, **val_metrics}
    pred_test = np.clip(best["model"].predict(X_test), 0.0, 1.0)
    test_meta = test_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object)
    pred_frame = prediction_frame(test_meta, test_df["normalized_rul"].to_numpy(dtype=np.float32), pred_test, split, "Ridge", "wavelet_only", None)
    per_bearing = per_bearing_metrics(test_meta, test_df["normalized_rul"].to_numpy(dtype=np.float32), pred_test, args.epsilon)
    per_bearing["model"] = "Ridge"
    per_bearing["feature_setting"] = "wavelet_only"
    per_bearing["K"] = "NA"
    per_bearing["protocol"] = split["protocol"]
    per_bearing["split_name"] = split["split_name"]
    per_bearing["best_alpha"] = best["alpha"]
    return pred_frame, per_bearing


def metrics_from_prediction_file(path: Path) -> pd.DataFrame:
    pred_frame = pd.read_csv(path)
    meta = pred_frame[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object)
    metrics = per_bearing_metrics(
        meta,
        pred_frame["y_true"].to_numpy(dtype=float),
        pred_frame["y_pred"].to_numpy(dtype=float),
        0.01,
    )
    first = pred_frame.iloc[0]
    metrics["model"] = first["model"]
    metrics["feature_setting"] = first["feature_setting"]
    metrics["K"] = first["K"]
    metrics["protocol"] = first["protocol"]
    metrics["split_name"] = first["split_name"]
    return metrics


def run_final(args, selected_k: int | None = None) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(args.fig_dir)
    pred_dir = ensure_dir(args.prediction_dir)
    sequence_dir = ensure_dir(args.final_sequence_dir)
    features_df = pd.read_csv(args.features)
    feature_cols = get_feature_columns(features_df)
    if len(feature_cols) != 72:
        raise RuntimeError(f"Expected 72 wavelet-only feature columns, found {len(feature_cols)}")
    if selected_k is None:
        selected_k = int((Path(args.out_dir) / "selected_k.txt").read_text(encoding="utf-8").strip())
    split_paths = all_eval_split_paths(Path(args.split_dir))
    train_args = SimpleNamespace(
        epochs=args.final_epochs,
        patience=args.final_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        smooth_weight=args.smooth_weight,
        epsilon=args.epsilon,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions = []
    all_per_bearing = []
    train_info = []
    for split_path in split_paths:
        split = load_split(split_path)
        print(f"Final comparison: split={split['split_name']}, K={selected_k}", flush=True)
        ridge_path = pred_dir / f"{split['split_name']}_Ridge_wavelet_KNA_predictions.csv"
        if args.resume_final and ridge_path.exists():
            print(f"Reusing existing {ridge_path.name}", flush=True)
            ridge_pred = pd.read_csv(ridge_path)
            ridge_metrics = metrics_from_prediction_file(ridge_path)
        else:
            ridge_pred, ridge_metrics = train_ridge_final(features_df, split, feature_cols, args)
            ridge_pred.to_csv(ridge_path, index=False)
        all_predictions.append(ridge_pred)
        all_per_bearing.append(ridge_metrics)

        seq_path = sequence_dir / f"{split['split_name']}_k{selected_k}.npz"
        if args.rebuild_sequences or not seq_path.exists():
            seq_path = make_sequence_file(features_df, split_path, selected_k, sequence_dir)
        data, _ = load_sequence(seq_path)
        for model_name in args.models:
            if model_name == "Ridge":
                continue
            pred_path = pred_dir / f"{split['split_name']}_{model_name}_wavelet_K{selected_k}_predictions.csv"
            if args.resume_final and pred_path.exists():
                print(f"Reusing existing {pred_path.name}", flush=True)
                pred_frame = pd.read_csv(pred_path)
                all_predictions.append(pred_frame)
                all_per_bearing.append(metrics_from_prediction_file(pred_path))
                continue
            seed = args.seed + sum(ord(ch) for ch in f"final:{selected_k}:{split['split_name']}:{model_name}")
            print(f"Training final {model_name} on {split['split_name']} using {device}", flush=True)
            val_summary, _, test_pred = train_sequence_model(seq_path, model_name, train_args, device, seed)
            test_pred = np.clip(test_pred, 0.0, 1.0)
            pred_frame = prediction_frame(data["meta_test"], data["y_test"], test_pred, split, model_name, "wavelet_only", selected_k)
            pred_frame.to_csv(pred_path, index=False)
            all_predictions.append(pred_frame)
            metrics = per_bearing_metrics(data["meta_test"], data["y_test"], test_pred, args.epsilon)
            metrics["model"] = model_name
            metrics["feature_setting"] = "wavelet_only"
            metrics["K"] = selected_k
            metrics["protocol"] = split["protocol"]
            metrics["split_name"] = split["split_name"]
            all_per_bearing.append(metrics)
            train_info.append({"split_name": split["split_name"], "model": model_name, "K": selected_k, **val_summary})

    predictions = pd.concat(all_predictions, ignore_index=True)
    per_bearing = pd.concat(all_per_bearing, ignore_index=True)
    predictions.to_csv(out_dir / "final_wavelet_predictions.csv", index=False)
    per_bearing.to_csv(out_dir / "final_wavelet_per_bearing_metrics.csv", index=False)
    pd.DataFrame(train_info).to_csv(out_dir / "final_wavelet_training_info.csv", index=False)
    split_summary, protocol_summary = summarize_final(per_bearing)
    split_summary.to_csv(out_dir / "final_wavelet_split_summary.csv", index=False)
    protocol_summary.to_csv(out_dir / "final_wavelet_protocol_summary.csv", index=False)
    save_final_tables_and_figures(protocol_summary, fig_dir, out_dir, selected_k)
    print(f"Saved final wavelet results to {out_dir}", flush=True)


def summarize_final(per_bearing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["mae", "rmse", "spearman", "late_mae", "monotonic_violation_rate"]
    split_summary = (
        per_bearing.groupby(["protocol", "split_name", "model", "feature_setting", "K"], as_index=False, dropna=False)[cols]
        .mean()
        .sort_values(["protocol", "split_name", "mae"])
    )
    protocol_summary = (
        split_summary.groupby(["protocol", "model", "feature_setting", "K"], as_index=False, dropna=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            late_mae_mean=("late_mae", "mean"),
            late_mae_std=("late_mae", "std"),
            monotonic_violation_rate_mean=("monotonic_violation_rate", "mean"),
            num_splits=("split_name", "nunique"),
        )
        .sort_values(["protocol", "mae_mean"])
    )
    return split_summary, protocol_summary


def plot_k_sensitivity(avg: pd.DataFrame, k_summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=220)
    for model in [m for m in MODEL_ORDER if m != "Ridge"]:
        sub = avg[avg["model"] == model].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["val_MAE"], marker="o", linewidth=2.2, label=model, color=MODEL_COLORS.get(model))
    best = k_summary.sort_values("avg_val_MAE").iloc[0]
    ax.axvline(best["K"], color="#1f2937", linestyle="--", linewidth=1.5)
    ax.set_title("Wavelet-only validation K sensitivity")
    ax.set_xlabel("Sequence length K")
    ax.set_ylabel("Validation MAE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    path = fig_dir / "wavelet_k_sensitivity_validation_mae.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def save_final_tables_and_figures(summary: pd.DataFrame, fig_dir: Path, out_dir: Path, selected_k: int) -> None:
    order = [m for m in MODEL_ORDER if m in set(summary["model"])]
    rows = []
    for protocol in ["cross_condition", "mixed_condition"]:
        sub = summary[summary["protocol"] == protocol].copy().sort_values("mae_mean")
        for _, row in sub.iterrows():
            rows.append(
                {
                    "Protocol": "Cross" if protocol == "cross_condition" else "Mixed",
                    "Model": row["model"],
                    "K": "NA" if row["model"] == "Ridge" else selected_k,
                    "MAE": f"{row['mae_mean']:.3f}",
                    "RMSE": f"{row['rmse_mean']:.3f}",
                    "Spearman": f"{row['spearman_mean']:.3f}",
                    "Late MAE": f"{row['late_mae_mean']:.3f}",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "final_wavelet_summary_for_ppt.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=220, sharey=True)
    for ax, protocol in zip(axes, ["cross_condition", "mixed_condition"], strict=False):
        sub = summary[summary["protocol"] == protocol].copy()
        sub["model"] = pd.Categorical(sub["model"], categories=order, ordered=True)
        sub = sub.sort_values("model")
        x = np.arange(len(sub))
        ax.bar(x, sub["mae_mean"], color=[MODEL_COLORS.get(str(m), "#64748b") for m in sub["model"]])
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].astype(str), rotation=25, ha="right")
        ax.set_title("Cross-condition" if protocol == "cross_condition" else "Mixed-condition")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Bearing-level MAE")
    fig.suptitle(f"Final Wavelet-only comparison (selected K={selected_k})")
    fig.tight_layout()
    fig.savefig(fig_dir / "final_wavelet_cross_mixed_mae.png", bbox_inches="tight")
    plt.close(fig)

    fig_h = max(4.8, 0.38 * len(table) + 1.3)
    fig, ax = plt.subplots(figsize=(12.8, fig_h), dpi=220)
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.2)
    tbl.scale(1.0, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if r == 0:
            cell.set_facecolor("#e8f0fb")
            cell.set_text_props(weight="bold", color="#0b1f3d")
        elif c == 0:
            cell.set_facecolor("#f8fafc")
            cell.set_text_props(weight="bold")
    ax.set_title(f"Final Wavelet-only RUL Results (selected K={selected_k})", fontsize=15, weight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_wavelet_summary_table.png", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Strict wavelet-only RUL experiment.")
    parser.add_argument("--mode", choices=["k", "final", "all"], default="all")
    parser.add_argument("--features", default="processed/features_wavelet_only.csv")
    parser.add_argument("--split_dir", default="processed/splits_final")
    parser.add_argument("--out_dir", type=Path, default=Path("results/tables/wavelet_strict"))
    parser.add_argument("--fig_dir", type=Path, default=Path("results/figures/wavelet_strict"))
    parser.add_argument("--prediction_dir", type=Path, default=Path("results/predictions_wavelet_strict"))
    parser.add_argument("--sequence_root", type=Path, default=Path("processed/sequences_wavelet_k_sensitivity"))
    parser.add_argument("--final_sequence_dir", type=Path, default=Path("processed/sequences_wavelet_final"))
    parser.add_argument("--k_values", nargs="+", type=int, default=[5, 10, 20, 40, 60, 80])
    parser.add_argument("--models", nargs="+", default=["LSTM", "TCN", "Transformer", "latent_ode"])
    parser.add_argument("--k_epochs", type=int, default=50)
    parser.add_argument("--k_patience", type=int, default=8)
    parser.add_argument("--final_epochs", type=int, default=120)
    parser.add_argument("--final_patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected_k", type=int)
    parser.add_argument("--rebuild_sequences", action="store_true")
    parser.add_argument("--resume_final", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_k = args.selected_k
    if args.mode in {"k", "all"}:
        selected_k = run_k_sensitivity(args)
    if args.mode in {"final", "all"}:
        run_final(args, selected_k)


if __name__ == "__main__":
    main()
