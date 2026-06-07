import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from conformal_uncertainty import load_deep_model, load_sequence, predict_discrete, predict_ode
from utils import compute_mae_rmse_r2, ensure_dir


def stable_seed(text: str, missing_ratio: float, base_seed: int) -> int:
    return base_seed + int(round(missing_ratio * 10000)) + sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 100000


def mean_impute_corruption(
    X: np.ndarray,
    train_feature_mean: np.ndarray,
    missing_ratio: float,
    seed: int,
) -> np.ndarray:
    if missing_ratio <= 0.0:
        return X.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    corrupted = X.astype(np.float32, copy=True)
    mask = rng.random(corrupted.shape) < missing_ratio
    impute_values = train_feature_mean.reshape(1, 1, -1)
    corrupted[mask] = np.broadcast_to(impute_values, corrupted.shape)[mask]
    return corrupted


def run_missing_robustness(args) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seq_path in sorted(Path(args.seq_dir).glob("*.npz")):
        data, split = load_sequence(seq_path)
        if args.protocol != "all" and split.get("protocol") != args.protocol:
            continue
        split_name = split.get("split_name", seq_path.stem.replace("_k10", ""))
        X_train = data["X_train"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.float32)
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        train_feature_mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        for model_name in args.models:
            model = load_deep_model(model_name, split_name, X_train.shape[-1], device)
            if model is None:
                print(f"Skipping {model_name} on {split_name}; checkpoint not found.")
                continue
            for missing_ratio in args.missing_ratios:
                seed = stable_seed(split_name + model_name, missing_ratio, args.seed)
                X_test_corrupt = mean_impute_corruption(X_test, train_feature_mean, missing_ratio, seed)
                if model_name in {"latent_ode", "condition_aware_ode"}:
                    c_test = (data["c_norm_test"] if "c_norm_test" in data else data["c_test"]).astype(np.float32)
                    y_pred = predict_ode(
                        model,
                        X_test_corrupt,
                        c_test,
                        data["tau_test"].astype(np.float32),
                        args.batch_size,
                        device,
                    )
                else:
                    y_pred = predict_discrete(model, X_test_corrupt, args.batch_size, device)
                metrics = compute_mae_rmse_r2(y_test, y_pred)
                rows.append(
                    {
                        "protocol": split.get("protocol", ""),
                        "split_name": split_name,
                        "model": model_name,
                        "missing_ratio": missing_ratio,
                        "imputation": "train_feature_mean",
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                    }
                )
                print(
                    f"Missing robustness: split={split_name}, model={model_name}, "
                    f"missing={missing_ratio:.2f}, MAE={metrics['mae']:.4f}"
                )
    results = pd.DataFrame(rows)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(results)} rows)")
    if not results.empty:
        avg = (
            results.groupby(["protocol", "model", "missing_ratio"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                mae_std=("mae", "std"),
                num_splits=("split_name", "nunique"),
            )
            .sort_values(["protocol", "missing_ratio", "mae"])
        )
        avg_path = out_path.with_name("missing_feature_robustness_average_results.csv")
        avg.to_csv(avg_path, index=False)
        print(f"Saved {avg_path} ({len(avg)} rows)")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models under random missing feature values.")
    parser.add_argument("--seq_dir", default="processed/sequences")
    parser.add_argument("--out", default="results/tables/missing_feature_robustness_results.csv")
    parser.add_argument("--protocol", default="cross_condition", choices=["all", "within_condition", "mixed_condition", "cross_condition", "literature_aligned_lobo"])
    parser.add_argument("--models", nargs="+", default=["LSTM", "TCN", "Transformer", "latent_ode"])
    parser.add_argument("--missing_ratios", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_missing_robustness(args)


if __name__ == "__main__":
    main()
