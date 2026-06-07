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
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from config import ACTIVE_MODEL_ORDER
from feature_analysis import score_features
from k_sensitivity import model_instance, predict_discrete, predict_ode, tensor_loader
from make_sequences import make_sequence_file
from utils import compute_mae_rmse_r2, ensure_dir, get_feature_columns, load_split, save_json, set_seed


MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "LSTM": "#54A24B",
    "TCN": "#F58518",
    "Transformer": "#72B7B2",
    "latent_ode": "#B279A2",
}

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


def ordered_models(models: list[str]) -> list[str]:
    return [m for m in ACTIVE_MODEL_ORDER if m in models] + [m for m in models if m not in ACTIVE_MODEL_ORDER]


def bearing_sort_key(bearing_id: str) -> tuple[int, int]:
    tail = str(bearing_id).split("Bearing")[-1]
    condition, bearing = tail.split("_")
    return int(condition), int(bearing)


def condition_bearings(features_df: pd.DataFrame, condition_id: str) -> list[str]:
    bearings = sorted(features_df.loc[features_df["condition_id"] == condition_id, "bearing_id"].unique())
    if len(bearings) != 5:
        raise ValueError(f"Expected 5 bearings for {condition_id}, found {len(bearings)}")
    return bearings


def make_mixed5fold_splits(features_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    ensure_dir(out_dir)
    conditions = ["C1", "C2", "C3"]
    by_condition = {condition: condition_bearings(features_df, condition) for condition in conditions}
    split_paths = []
    for fold in range(5):
        train_bearings = []
        val_bearings = []
        test_bearings = []
        for condition in conditions:
            bearings = by_condition[condition]
            test = bearings[fold]
            val = bearings[(fold + 1) % len(bearings)]
            train = [bearing for bearing in bearings if bearing not in {test, val}]
            test_bearings.append(test)
            val_bearings.append(val)
            train_bearings.extend(train)
        split = {
            "protocol": "mixed5fold_condition",
            "split_name": f"mixed5fold_fold{fold + 1}",
            "fold": fold + 1,
            "train_bearings": sorted(train_bearings),
            "val_bearings": sorted(val_bearings),
            "test_bearings": sorted(test_bearings),
            "train_conditions": conditions,
            "test_conditions": conditions,
        }
        path = out_dir / f"{split['split_name']}.json"
        save_json(path, split)
        split_paths.append(path)
        print(
            f"{split['split_name']}: train={len(train_bearings)}, "
            f"val={len(val_bearings)}, test={len(test_bearings)}"
        )
    return split_paths


def subset_by_bearings(df: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    return df[df["bearing_id"].isin(bearings)].copy()


def select_feature_frame(features_df: pd.DataFrame, setting: str, split: dict, top_k: int) -> tuple[pd.DataFrame, list[str]]:
    all_cols = get_feature_columns(features_df)
    if setting == "original":
        cols = [col for col in all_cols if "_wpt_" not in col and "_wav_" not in col]
    elif setting == "wavelet_only":
        cols = [col for col in all_cols if "_wpt_" in col or "_wav_" in col]
    elif setting == "all_expanded":
        cols = all_cols
    elif setting == "selected_top":
        train_df = subset_by_bearings(features_df, split["train_bearings"])
        scores = score_features(train_df, all_cols)
        cols = scores.head(top_k)["feature"].tolist()
    else:
        raise ValueError(f"Unknown feature setting: {setting}")
    return features_df[META_COLUMNS + cols].copy(), cols


def affine_calibration(y_val: np.ndarray, pred_val: np.ndarray) -> tuple[float, float]:
    y_val = np.asarray(y_val, dtype=float)
    pred_val = np.asarray(pred_val, dtype=float)
    if len(y_val) < 2 or np.std(pred_val) < 1e-8:
        return 1.0, float(np.mean(y_val - pred_val))
    design = np.column_stack([pred_val, np.ones_like(pred_val)])
    slope, intercept = np.linalg.lstsq(design, y_val, rcond=None)[0]
    return float(slope), float(intercept)


def apply_calibration(pred: np.ndarray, slope: float, intercept: float, clip: bool) -> np.ndarray:
    calibrated = slope * np.asarray(pred, dtype=float) + intercept
    if clip:
        calibrated = np.clip(calibrated, 0.0, 1.0)
    return calibrated


def prediction_frame(meta, y_true, pred, split: dict, model: str, variant: str) -> pd.DataFrame:
    meta = np.asarray(meta, dtype=object)
    out = pd.DataFrame(meta, columns=["bearing_id", "condition_id", "time_index"])
    out["time_index"] = out["time_index"].astype(int)
    out["normalized_rul"] = np.asarray(y_true, dtype=float)
    out["y_pred"] = np.asarray(pred, dtype=float)
    out["model"] = model
    out["prediction_variant"] = variant
    out["protocol"] = split["protocol"]
    out["split_name"] = split["split_name"]
    return out.sort_values(["bearing_id", "time_index"]).reset_index(drop=True)


def train_ridge_predictions(features_df: pd.DataFrame, split: dict, feature_cols: list[str], clip: bool) -> tuple[pd.DataFrame, list[dict]]:
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
        val_pred = model.predict(X_val)
        metrics = compute_mae_rmse_r2(y_val, val_pred)
        if best is None or metrics["mae"] < best["val_mae"]:
            best = {"model": model, "alpha": alpha, "val_mae": metrics["mae"], "val_pred": val_pred}

    raw_test = best["model"].predict(X_test)
    raw_test = np.clip(raw_test, 0.0, 1.0) if clip else raw_test
    slope, intercept = affine_calibration(y_val, best["val_pred"])
    cal_test = apply_calibration(best["model"].predict(X_test), slope, intercept, clip)

    meta = test_df[["bearing_id", "condition_id", "time_index"]].to_numpy(dtype=object)
    y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)
    frames = [
        prediction_frame(meta, y_test, raw_test, split, "Ridge", "raw"),
        prediction_frame(meta, y_test, cal_test, split, "Ridge", "val_calibrated"),
    ]
    info = [
        {"model": "Ridge", "variant": "raw", "best_epoch": 0, "best_val_mae": best["val_mae"], "alpha": best["alpha"]},
        {
            "model": "Ridge",
            "variant": "val_calibrated",
            "best_epoch": 0,
            "best_val_mae": best["val_mae"],
            "alpha": best["alpha"],
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        },
    ]
    return pd.concat(frames, ignore_index=True), info


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
        "meta_test": npz["meta_test"],
    }
    return data, split


def train_neural_predictions(
    seq_path: Path,
    model_name: str,
    train_args,
    device: torch.device,
    seed: int,
    clip: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    data, split = load_sequence(seq_path)
    if len(data["X_train"]) == 0 or len(data["X_val"]) == 0 or len(data["X_test"]) == 0:
        raise RuntimeError(f"Empty train/val/test in {seq_path}")
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
                smooth = torch.mean((z_traj[1:] - z_traj[:-1]) ** 2)
                loss = loss_fn(pred, yb) + train_args.smooth_weight * smooth
            else:
                xb, yb = batch
                loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        if model_name == "latent_ode":
            val_pred = predict_ode(model, data["X_val"], data["c_val"], data["tau_val"], train_args.batch_size, device)
        else:
            val_pred = predict_discrete(model, data["X_val"], train_args.batch_size, device)
        val_mae = compute_mae_rmse_r2(data["y_val"], val_pred)["mae"]
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
    if model_name == "latent_ode":
        val_pred = predict_ode(model, data["X_val"], data["c_val"], data["tau_val"], train_args.batch_size, device)
        test_pred = predict_ode(model, data["X_test"], data["c_test"], data["tau_test"], train_args.batch_size, device)
    else:
        val_pred = predict_discrete(model, data["X_val"], train_args.batch_size, device)
        test_pred = predict_discrete(model, data["X_test"], train_args.batch_size, device)

    raw_test = np.clip(test_pred, 0.0, 1.0) if clip else test_pred
    slope, intercept = affine_calibration(data["y_val"], val_pred)
    cal_test = apply_calibration(test_pred, slope, intercept, clip)
    frames = [
        prediction_frame(data["meta_test"], data["y_test"], raw_test, split, model_name, "raw"),
        prediction_frame(data["meta_test"], data["y_test"], cal_test, split, model_name, "val_calibrated"),
    ]
    info = [
        {"model": model_name, "variant": "raw", "best_epoch": best_epoch, "best_val_mae": best_val_mae},
        {
            "model": model_name,
            "variant": "val_calibrated",
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        },
    ]
    return pd.concat(frames, ignore_index=True), info


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def metric_row(group: pd.DataFrame, epsilon: float) -> dict:
    group = group.sort_values("time_index")
    y_true = group["normalized_rul"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true
    late = y_true <= 0.3
    return {
        "n_points": len(group),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "spearman": spearman_corr(y_true, y_pred),
        "late_mae": float(np.mean(np.abs(error[late]))) if np.any(late) else float("nan"),
        "monotonic_violation_rate": float(np.mean(np.diff(y_pred) > epsilon)) if len(y_pred) > 1 else float("nan"),
    }


def compute_metrics(predictions: pd.DataFrame, epsilon: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    group_cols = ["protocol", "split_name", "model", "prediction_variant", "condition_id", "bearing_id"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        rows.append({**dict(zip(group_cols, keys, strict=False)), **metric_row(group, epsilon)})
    per_bearing = pd.DataFrame(rows)
    metric_cols = ["mae", "rmse", "r2", "spearman", "late_mae", "monotonic_violation_rate"]
    split_summary = (
        per_bearing.groupby(["protocol", "split_name", "model", "prediction_variant"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["prediction_variant", "split_name", "mae"])
    )
    protocol_summary = (
        split_summary.groupby(["protocol", "model", "prediction_variant"], as_index=False)
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
        .sort_values(["prediction_variant", "mae_mean"])
    )
    return per_bearing, split_summary, protocol_summary


def save_bar(summary: pd.DataFrame, fig_dir: Path, metric: str, label: str) -> None:
    models = ordered_models(summary["model"].unique().tolist())
    variants = ["raw", "val_calibrated"]
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for offset, variant in [(-width / 2, "raw"), (width / 2, "val_calibrated")]:
        values = []
        errors = []
        for model in models:
            row = summary[(summary["model"] == model) & (summary["prediction_variant"] == variant)]
            values.append(float(row[f"{metric}_mean"].iloc[0]) if not row.empty else np.nan)
            errors.append(float(row[f"{metric}_std"].fillna(0.0).iloc[0]) if not row.empty else 0.0)
        ax.bar(x + offset, values, width, yerr=errors, capsize=3, label=variant.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel(label)
    if metric == "r2":
        ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_title(f"Mixed 5-fold {label}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    plt.tight_layout()
    path = fig_dir / f"mixed5fold_{metric}_bar.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def save_heatmap(split_summary: pd.DataFrame, fig_dir: Path) -> None:
    data = split_summary[split_summary["prediction_variant"] == "val_calibrated"].copy()
    models = ordered_models(data["model"].unique().tolist())
    folds = sorted(data["split_name"].unique())
    pivot = data.pivot_table(index="model", columns="split_name", values="mae", aggfunc="mean")
    pivot = pivot.loc[[m for m in models if m in pivot.index], folds]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels([fold.replace("mixed5fold_", "") for fold in folds], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="#102033")
    fig.colorbar(image, ax=ax, label="MAE")
    ax.set_title("Mixed 5-fold per-fold MAE (validation-calibrated)")
    plt.tight_layout()
    path = fig_dir / "mixed5fold_per_fold_mae_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def save_table_image(df: pd.DataFrame, path: Path, title: str) -> None:
    fig_height = max(3.0, 0.36 * len(df) + 1.4)
    fig_width = max(8.0, 1.18 * len(df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#C9D3E5")
        if row == 0:
            cell.set_text_props(weight="bold", color="#102033")
            cell.set_facecolor("#EAF0FB")
        else:
            cell.set_facecolor("#FFFFFF")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def save_summary_table_image(summary: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    data = summary[summary["prediction_variant"] == "val_calibrated"].copy()
    data["model"] = pd.Categorical(data["model"], categories=ordered_models(data["model"].unique().tolist()), ordered=True)
    data = data.sort_values("model")
    rows = []
    for _, row in data.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "MAE": f"{row['mae_mean']:.3f} +/- {row['mae_std']:.3f}",
                "RMSE": f"{row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f}",
                "R2": f"{row['r2_mean']:.3f} +/- {row['r2_std']:.3f}",
                "Spearman": f"{row['spearman_mean']:.3f} +/- {row['spearman_std']:.3f}",
                "Late MAE": f"{row['late_mae_mean']:.3f} +/- {row['late_mae_std']:.3f}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "mixed5fold_calibrated_summary_for_ppt.csv", index=False)
    save_table_image(out, fig_dir / "mixed5fold_calibrated_summary_table.png", "Mixed 5-fold summary (validation-calibrated)")


def save_best_per_bearing_table(per_bearing: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    calibrated = summary[summary["prediction_variant"] == "val_calibrated"].sort_values("mae_mean")
    best_model = str(calibrated.iloc[0]["model"])
    data = per_bearing[
        (per_bearing["prediction_variant"] == "val_calibrated") & (per_bearing["model"] == best_model)
    ].copy()
    data = data.sort_values("bearing_id", key=lambda s: s.map(bearing_sort_key))
    table = data[["bearing_id", "mae", "rmse", "r2", "spearman", "late_mae"]].rename(
        columns={
            "bearing_id": "Test bearing",
            "mae": "MAE",
            "rmse": "RMSE",
            "r2": "R2",
            "spearman": "Spearman",
            "late_mae": "Late MAE",
        }
    )
    for col in ["MAE", "RMSE", "R2", "Spearman", "Late MAE"]:
        table[col] = table[col].map(lambda value: f"{value:.3f}")
    table.to_csv(table_dir / f"mixed5fold_{best_model}_per_bearing_for_ppt.csv", index=False)
    save_table_image(table, fig_dir / f"mixed5fold_{best_model}_per_bearing_table.png", f"Per-bearing mixed 5-fold metrics ({best_model}, calibrated)")


def save_prediction_curve(predictions: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path) -> None:
    best_model = str(summary[summary["prediction_variant"] == "val_calibrated"].sort_values("mae_mean").iloc[0]["model"])
    data = predictions[
        (predictions["model"] == best_model)
        & (predictions["prediction_variant"] == "val_calibrated")
        & (predictions["bearing_id"] == "C3_Bearing3_5")
    ].copy()
    if data.empty:
        data = predictions[(predictions["model"] == best_model) & (predictions["prediction_variant"] == "val_calibrated")].copy()
        data = data[data["bearing_id"] == sorted(data["bearing_id"].unique(), key=bearing_sort_key)[-1]]
    data = data.sort_values("time_index")
    x = np.linspace(0, 100, len(data))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, data["normalized_rul"], color="#222222", linestyle="--", linewidth=1.5, label="constructed target")
    ax.plot(x, data["y_pred"], color=MODEL_COLORS.get(best_model, "#F58518"), linewidth=1.4, label=f"{best_model} calibrated")
    ax.set_xlabel("Life percentage (%)")
    ax.set_ylabel("Normalized RUL")
    ax.set_title(f"Mixed 5-fold representative trajectory: {data['bearing_id'].iloc[0]}")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    plt.tight_layout()
    path = fig_dir / "mixed5fold_representative_prediction_curve.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def make_figures(per_bearing: pd.DataFrame, split_summary: pd.DataFrame, summary: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path, fig_dir: Path) -> None:
    save_bar(summary, fig_dir, "mae", "MAE")
    save_bar(summary, fig_dir, "r2", "R2")
    save_heatmap(split_summary, fig_dir)
    save_summary_table_image(summary, fig_dir, out_dir)
    save_best_per_bearing_table(per_bearing, summary, fig_dir, out_dir)
    save_prediction_curve(predictions, summary, fig_dir)


def run(args) -> None:
    set_seed(args.seed)
    features_df = pd.read_csv(args.features)
    split_paths = make_mixed5fold_splits(features_df, Path(args.split_dir))
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
        print(f"Running {split['split_name']} with {len(feature_cols)} {args.feature_setting} features")
        ridge_pred, ridge_info = train_ridge_predictions(setting_df, split, feature_cols, clip=not args.no_clip)
        all_predictions.append(ridge_pred)
        train_info.extend([{**item, "split_name": split["split_name"]} for item in ridge_info])
        ridge_pred.to_csv(pred_dir / f"{split['split_name']}_Ridge.csv", index=False)

        seq_path = seq_dir / f"{split['split_name']}_k{args.k}.npz"
        if args.rebuild_sequences or not seq_path.exists():
            seq_path = make_sequence_file(setting_df, split_path, args.k, seq_dir)

        for model_name in args.models:
            if model_name == "Ridge":
                continue
            seed = args.seed + sum(ord(ch) for ch in f"{split['split_name']}:{model_name}:{args.feature_setting}")
            print(f"Training {model_name} on {split['split_name']} using {device}")
            pred_frame, info = train_neural_predictions(seq_path, model_name, train_args, device, seed, clip=not args.no_clip)
            all_predictions.append(pred_frame)
            train_info.extend([{**item, "split_name": split["split_name"], "seed": seed} for item in info])
            pred_frame.to_csv(pred_dir / f"{split['split_name']}_{model_name}.csv", index=False)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out_dir / "mixed5fold_predictions.csv", index=False)
    pd.DataFrame(train_info).to_csv(out_dir / "mixed5fold_training_info.csv", index=False)
    per_bearing, split_summary, protocol_summary = compute_metrics(predictions, args.epsilon)
    per_bearing.to_csv(out_dir / "mixed5fold_per_bearing_metrics.csv", index=False)
    split_summary.to_csv(out_dir / "mixed5fold_split_metrics.csv", index=False)
    protocol_summary.to_csv(out_dir / "mixed5fold_protocol_summary.csv", index=False)
    make_figures(per_bearing, split_summary, protocol_summary, predictions, out_dir, fig_dir)
    print(f"Saved mixed 5-fold outputs to {out_dir} and {fig_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run mixed-condition 5-fold trajectory-level RUL experiment.")
    parser.add_argument("--features", default="processed/features_wavelet.csv")
    parser.add_argument("--feature_setting", default="selected_top", choices=["original", "wavelet_only", "all_expanded", "selected_top"])
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--split_dir", default="processed/splits_mixed5fold")
    parser.add_argument("--sequence_dir", default="processed/sequences_mixed5fold")
    parser.add_argument("--prediction_dir", default="results/predictions_mixed5fold")
    parser.add_argument("--out_dir", default="results/tables/mixed5fold")
    parser.add_argument("--fig_dir", default="results/figures/mixed5fold")
    parser.add_argument("--models", nargs="+", default=["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"], choices=["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"])
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


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
