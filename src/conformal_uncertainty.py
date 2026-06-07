import argparse
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

from train_lstm import LSTMRegressor
from train_ode import LatentODERegressor
from train_tcn import TCNRegressor
from train_transformer import TransformerRegressor
from utils import compute_mae_rmse_r2, ensure_dir, get_feature_columns, load_split


MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "LSTM": "#54A24B",
    "TCN": "#F58518",
    "Transformer": "#72B7B2",
    "latent_ode": "#B279A2",
}


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    residuals = np.sort(np.asarray(residuals, dtype=float))
    n = len(residuals)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(residuals[rank - 1])


def interval_metrics(y_true: np.ndarray, y_pred: np.ndarray, qhat: float, clip: bool) -> dict:
    lower = y_pred - qhat
    upper = y_pred + qhat
    if clip:
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
    covered = (y_true >= lower) & (y_true <= upper)
    metrics = compute_mae_rmse_r2(y_true, y_pred)
    metrics.update(
        {
            "coverage": float(np.mean(covered)),
            "avg_interval_length": float(np.mean(upper - lower)),
            "qhat": float(qhat),
        }
    )
    return metrics


@torch.no_grad()
def predict_discrete(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


@torch.no_grad()
def predict_ode(model, X: np.ndarray, c: np.ndarray, tau: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
        cb = torch.tensor(c[start : start + batch_size], dtype=torch.float32, device=device)
        tb = torch.tensor(tau[start : start + batch_size], dtype=torch.float32, device=device)
        pred, _ = model(xb, cb, tb)
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


def load_sequence(path: Path) -> tuple[dict, dict]:
    npz = np.load(path, allow_pickle=True)
    split = json.loads(npz["split_json"].item()) if "split_json" in npz else {}
    data = {key: npz[key] for key in npz.files}
    return data, split


def load_deep_model(model_name: str, split_name: str, input_dim: int, device: torch.device):
    checkpoint_suffix = {
        "LSTM": "lstm",
        "TCN": "tcn",
        "Transformer": "transformer",
        "latent_ode": "latent_ode",
        "condition_aware_ode": "condition_aware_ode",
    }[model_name]
    checkpoint_path = Path("results/checkpoints") / f"{split_name}_{checkpoint_suffix}.pt"
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if model_name == "LSTM":
        model = LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=int(checkpoint.get("hidden_dim", 64)),
            num_layers=int(checkpoint.get("num_layers", 1)),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )
    elif model_name == "TCN":
        model = TCNRegressor(
            input_dim=input_dim,
            hidden_dim=int(checkpoint.get("hidden_dim", 64)),
            levels=int(checkpoint.get("levels", 3)),
            kernel_size=int(checkpoint.get("kernel_size", 3)),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )
    elif model_name == "Transformer":
        model = TransformerRegressor(
            input_dim=input_dim,
            d_model=int(checkpoint.get("d_model", 64)),
            nhead=int(checkpoint.get("nhead", 4)),
            num_layers=int(checkpoint.get("num_layers", 2)),
            dim_feedforward=int(checkpoint.get("dim_feedforward", 128)),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )
    else:
        model = LatentODERegressor(
            input_dim=input_dim,
            latent_dim=int(checkpoint.get("latent_dim", 16)),
            model_type=model_name,
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def deep_conformal_rows(seq_path: Path, models: list[str], alpha: float, batch_size: int, clip: bool) -> tuple[list[dict], list[pd.DataFrame]]:
    data, split = load_sequence(seq_path)
    split_name = split.get("split_name", seq_path.stem.replace("_k10", ""))
    protocol = split.get("protocol", "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    pred_frames = []
    input_dim = int(data["X_train"].shape[-1])
    for model_name in models:
        if model_name == "Ridge":
            continue
        model = load_deep_model(model_name, split_name, input_dim, device)
        if model is None:
            continue
        X_val = data["X_val"].astype(np.float32)
        y_val = data["y_val"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)
        if model_name in {"latent_ode", "condition_aware_ode"}:
            c_val = (data["c_norm_val"] if "c_norm_val" in data else data["c_val"]).astype(np.float32)
            c_test = (data["c_norm_test"] if "c_norm_test" in data else data["c_test"]).astype(np.float32)
            val_pred = predict_ode(model, X_val, c_val, data["tau_val"].astype(np.float32), batch_size, device)
            test_pred = predict_ode(model, X_test, c_test, data["tau_test"].astype(np.float32), batch_size, device)
        else:
            val_pred = predict_discrete(model, X_val, batch_size, device)
            test_pred = predict_discrete(model, X_test, batch_size, device)
        qhat = conformal_quantile(np.abs(y_val - val_pred), alpha)
        metrics = interval_metrics(y_test, test_pred, qhat, clip)
        metrics["calibration_gap"] = abs(metrics["coverage"] - (1.0 - alpha))
        rows.append(
            {
                "protocol": protocol,
                "split_name": split_name,
                "model": model_name,
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "calibration_size": len(y_val),
                "test_size": len(y_test),
                "calibration_source": "validation_residuals_existing_checkpoint",
                **metrics,
            }
        )
        frame = prediction_interval_frame(
            data["meta_test"], y_test, test_pred, qhat, split, model_name, clip
        )
        pred_frames.append(frame)
    return rows, pred_frames


def prediction_interval_frame(meta, y_true, y_pred, qhat, split: dict, model_name: str, clip: bool) -> pd.DataFrame:
    meta = np.asarray(meta, dtype=object)
    frame = pd.DataFrame(meta, columns=["bearing_id", "condition_id", "time_index"])
    frame["time_index"] = frame["time_index"].astype(int)
    frame["normalized_rul"] = np.asarray(y_true, dtype=float)
    frame["y_pred"] = np.asarray(y_pred, dtype=float)
    frame["lower"] = frame["y_pred"] - qhat
    frame["upper"] = frame["y_pred"] + qhat
    if clip:
        frame["lower"] = frame["lower"].clip(0.0, 1.0)
        frame["upper"] = frame["upper"].clip(0.0, 1.0)
    frame["model"] = model_name
    frame["protocol"] = split.get("protocol", "")
    frame["split_name"] = split.get("split_name", "")
    return frame.sort_values(["bearing_id", "time_index"]).reset_index(drop=True)


def ridge_conformal_rows(features_df: pd.DataFrame, split_dir: str, alpha: float, clip: bool, protocol_filter: str) -> tuple[list[dict], list[pd.DataFrame]]:
    feature_cols = get_feature_columns(features_df)
    rows = []
    pred_frames = []
    for split_path in sorted(Path(split_dir).glob("*.json")):
        split = load_split(split_path)
        if protocol_filter != "all" and split.get("protocol") != protocol_filter:
            continue
        train_df = features_df[features_df["bearing_id"].isin(split["train_bearings"])]
        val_df = features_df[features_df["bearing_id"].isin(split["val_bearings"])]
        test_df = features_df[features_df["bearing_id"].isin(split["test_bearings"])]
        if train_df.empty or val_df.empty or test_df.empty:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
        X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
        X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
        y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
        y_val = val_df["normalized_rul"].to_numpy(dtype=np.float32)
        y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)
        best = None
        for alpha_ridge in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha_ridge)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mae = compute_mae_rmse_r2(y_val, val_pred)["mae"]
            if best is None or val_mae < best["val_mae"]:
                best = {"model": model, "val_mae": val_mae, "alpha": alpha_ridge}
        val_pred = best["model"].predict(X_val)
        test_pred = best["model"].predict(X_test)
        qhat = conformal_quantile(np.abs(y_val - val_pred), alpha)
        metrics = interval_metrics(y_test, test_pred, qhat, clip)
        metrics["calibration_gap"] = abs(metrics["coverage"] - (1.0 - alpha))
        rows.append(
            {
                "protocol": split.get("protocol", ""),
                "split_name": split.get("split_name", split_path.stem),
                "model": "Ridge",
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "calibration_size": len(y_val),
                "test_size": len(y_test),
                "calibration_source": "validation_residuals_retrained_ridge",
                **metrics,
            }
        )
        frame = test_df[["bearing_id", "condition_id", "time_index", "normalized_rul"]].copy()
        frame["y_pred"] = test_pred
        frame["lower"] = test_pred - qhat
        frame["upper"] = test_pred + qhat
        if clip:
            frame["lower"] = frame["lower"].clip(0.0, 1.0)
            frame["upper"] = frame["upper"].clip(0.0, 1.0)
        frame["model"] = "Ridge"
        frame["protocol"] = split.get("protocol", "")
        frame["split_name"] = split.get("split_name", split_path.stem)
        pred_frames.append(frame.sort_values(["bearing_id", "time_index"]).reset_index(drop=True))
    return rows, pred_frames


def plot_representative_interval(predictions: pd.DataFrame, out_dir: Path, preferred_model: str) -> None:
    if predictions.empty:
        return
    frame = predictions[predictions["model"] == preferred_model].copy()
    if frame.empty:
        frame = predictions.copy()
        preferred_model = str(frame["model"].iloc[0])
    split_name = str(frame["split_name"].iloc[0])
    bearing_id = str(frame["bearing_id"].iloc[0])
    one = frame[(frame["split_name"] == split_name) & (frame["bearing_id"].astype(str) == bearing_id)].sort_values("time_index")
    if one.empty:
        return
    plt.figure(figsize=(7.2, 3.8))
    plt.fill_between(
        one["time_index"].to_numpy(),
        one["lower"].to_numpy(),
        one["upper"].to_numpy(),
        color=MODEL_COLORS.get(preferred_model, "#777777"),
        alpha=0.18,
        label="90% interval",
    )
    plt.plot(
        one["time_index"],
        one["normalized_rul"],
        color="#222222",
        linewidth=2.0,
        label="constructed normalized RUL target",
    )
    plt.plot(
        one["time_index"],
        one["y_pred"],
        color=MODEL_COLORS.get(preferred_model, "#777777"),
        linewidth=1.5,
        label=preferred_model,
    )
    plt.xlabel("Time index")
    plt.ylabel("Normalized RUL")
    plt.ylim(-0.03, 1.03)
    plt.grid(alpha=0.2)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    out_path = out_dir / f"{split_name}_{bearing_id}_{preferred_model}_conformal_interval.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-calibrated conformal intervals for RUL predictions.")
    parser.add_argument("--seq_dir", default="processed/sequences")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--split_dir", default="processed/splits")
    parser.add_argument("--out", default="results/tables/conformal_interval_results.csv")
    parser.add_argument("--protocol", default="cross_condition", choices=["all", "within_condition", "mixed_condition", "cross_condition", "literature_aligned_lobo"])
    parser.add_argument("--models", nargs="+", default=["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--plot_model", default="TCN")
    return parser.parse_args()


def main():
    args = parse_args()
    clip = not args.no_clip
    rows = []
    pred_frames = []
    if "Ridge" in args.models and Path(args.features).exists():
        ridge_rows, ridge_frames = ridge_conformal_rows(
            pd.read_csv(args.features), args.split_dir, args.alpha, clip, args.protocol
        )
        rows.extend(ridge_rows)
        pred_frames.extend(ridge_frames)
    for seq_path in sorted(Path(args.seq_dir).glob("*.npz")):
        _, split = load_sequence(seq_path)
        if args.protocol != "all" and split.get("protocol") != args.protocol:
            continue
        deep_rows, deep_frames = deep_conformal_rows(
            seq_path, args.models, args.alpha, args.batch_size, clip
        )
        rows.extend(deep_rows)
        pred_frames.extend(deep_frames)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    results = pd.DataFrame(rows)
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(results)} rows)")
    if not results.empty:
        avg = (
            results.groupby(["protocol", "model"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                coverage=("coverage", "mean"),
                avg_interval_length=("avg_interval_length", "mean"),
                calibration_gap=("calibration_gap", "mean"),
                qhat=("qhat", "mean"),
                num_splits=("split_name", "nunique"),
            )
            .sort_values(["protocol", "mae"])
        )
        avg_path = out_path.with_name("conformal_interval_average_results.csv")
        avg.to_csv(avg_path, index=False)
        print(f"Saved {avg_path} ({len(avg)} rows)")

    if pred_frames:
        prediction_df = pd.concat(pred_frames, ignore_index=True)
        pred_path = out_path.with_name("conformal_interval_predictions.csv")
        prediction_df.to_csv(pred_path, index=False)
        print(f"Saved {pred_path} ({len(prediction_df)} rows)")
        plot_representative_interval(prediction_df, ensure_dir("results/figures/uncertainty"), args.plot_model)


if __name__ == "__main__":
    main()
