import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED
from utils import append_result_csv, compute_mae_rmse_r2, ensure_dir, get_feature_columns, load_split, set_seed


RESULT_COLUMNS = [
    "protocol",
    "split_name",
    "model",
    "input_type",
    "best_params",
    "mae",
    "rmse",
    "r2",
    "num_train_bearings",
    "num_val_bearings",
    "num_test_bearings",
]


def subset_by_bearings(df: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    return df[df["bearing_id"].isin(bearings)].copy()


def make_prediction_frame(df: pd.DataFrame, y_pred, split: dict, model_name: str) -> pd.DataFrame:
    out = df[["bearing_id", "condition_id", "time_index", "normalized_rul"]].copy()
    out["y_pred"] = np.asarray(y_pred, dtype=float)
    out["model"] = model_name
    out["protocol"] = split["protocol"]
    out["split_name"] = split["split_name"]
    return out.sort_values(["bearing_id", "time_index"]).reset_index(drop=True)


def train_ridge(X_train, y_train, X_val, y_val):
    best = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        metrics = compute_mae_rmse_r2(y_val, val_pred)
        if best is None or metrics["mae"] < best["metrics"]["mae"]:
            best = {"model": model, "params": {"alpha": alpha}, "metrics": metrics}
    return best["model"], best["params"]


def run_split(features_df: pd.DataFrame, split_path: Path) -> None:
    split = load_split(split_path)
    feature_cols = get_feature_columns(features_df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found.")

    train_df = subset_by_bearings(features_df, split["train_bearings"])
    val_df = subset_by_bearings(features_df, split["val_bearings"])
    test_df = subset_by_bearings(features_df, split["test_bearings"])
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(f"Empty train/val/test subset for split {split['split_name']}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_val = val_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)

    models = [
        ("Ridge", train_ridge),
    ]
    for model_name, trainer in models:
        print(f"Training {model_name} on {split['split_name']}")
        model, best_params = trainer(X_train, y_train, X_val, y_val)
        y_pred = model.predict(X_test)
        metrics = compute_mae_rmse_r2(y_test, y_pred)

        pred_dir = ensure_dir("results/predictions")
        pred_path = pred_dir / f"{split['split_name']}_{model_name}.csv"
        make_prediction_frame(test_df, y_pred, split, model_name).to_csv(pred_path, index=False)

        result_row = {
            "protocol": split["protocol"],
            "split_name": split["split_name"],
            "model": model_name,
            "input_type": "current_window_features",
            "best_params": json.dumps(best_params, sort_keys=True),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "num_train_bearings": len(split["train_bearings"]),
            "num_val_bearings": len(split["val_bearings"]),
            "num_test_bearings": len(split["test_bearings"]),
        }
        append_result_csv("results/tables/all_results.csv", {k: result_row[k] for k in RESULT_COLUMNS})
        print(
            f"{split['split_name']} {model_name}: "
            f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
        )


def split_paths_from_args(split: str | None, split_dir: str | None) -> list[Path]:
    if split:
        return [Path(split)]
    if split_dir:
        paths = sorted(Path(split_dir).glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"No split JSON files found in {split_dir}")
        return paths
    raise ValueError("Provide either --split or --split_dir")


def parse_args():
    parser = argparse.ArgumentParser(description="Train feature-based ML baselines.")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--split")
    parser.add_argument("--split_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    features_df = pd.read_csv(args.features)
    for split_path in split_paths_from_args(args.split, args.split_dir):
        run_split(features_df, split_path)


if __name__ == "__main__":
    main()
