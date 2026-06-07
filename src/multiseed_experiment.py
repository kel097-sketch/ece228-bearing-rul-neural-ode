import argparse
from pathlib import Path

import pandas as pd
import torch

from k_sensitivity import load_sequence, model_instance, train_discrete_model, train_ode_model
from utils import ensure_dir, set_seed


def run_multiseed(args) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seq_path in sorted(Path(args.seq_dir).glob("*.npz")):
        data, split = load_sequence(seq_path)
        if args.protocol != "all" and split.get("protocol") != args.protocol:
            continue
        split_name = split.get("split_name", seq_path.stem.replace("_k10", ""))
        if len(data["X_train"]) == 0 or len(data["X_val"]) == 0 or len(data["X_test"]) == 0:
            continue
        for model_name in args.models:
            for seed in args.seeds:
                set_seed(seed)
                model = model_instance(model_name, data["X_train"].shape[-1]).to(device)
                print(f"Multi-seed: split={split_name}, model={model_name}, seed={seed}, device={device}")
                if model_name in {"latent_ode", "condition_aware_ode"}:
                    metrics = train_ode_model(model, data, args, device, seed)
                else:
                    metrics = train_discrete_model(model, data, args, device, seed)
                rows.append(
                    {
                        "protocol": split.get("protocol", ""),
                        "split_name": split_name,
                        "model": model_name,
                        "seed": seed,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "best_epoch": metrics["best_epoch"],
                        "best_val_mae": metrics["best_val_mae"],
                    }
                )
    results = pd.DataFrame(rows)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(results)} rows)")
    if not results.empty:
        avg = (
            results.groupby(["protocol", "split_name", "model"], as_index=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                r2_mean=("r2", "mean"),
                num_seeds=("seed", "nunique"),
            )
            .sort_values(["protocol", "split_name", "mae_mean"])
        )
        avg_path = out_path.with_name("multiseed_split_average_results.csv")
        avg.to_csv(avg_path, index=False)
        print(f"Saved {avg_path} ({len(avg)} rows)")
        model_avg = (
            results.groupby(["protocol", "model"], as_index=False)
            .agg(
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                r2_mean=("r2", "mean"),
                num_runs=("seed", "count"),
                num_splits=("split_name", "nunique"),
            )
            .sort_values(["protocol", "mae_mean"])
        )
        model_avg_path = out_path.with_name("multiseed_model_average_results.csv")
        model_avg.to_csv(model_avg_path, index=False)
        print(f"Saved {model_avg_path} ({len(model_avg)} rows)")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Repeat cross-condition training with multiple random seeds.")
    parser.add_argument("--seq_dir", default="processed/sequences")
    parser.add_argument("--out", default="results/tables/multiseed_results.csv")
    parser.add_argument("--protocol", default="cross_condition", choices=["all", "within_condition", "mixed_condition", "cross_condition", "literature_aligned_lobo"])
    parser.add_argument("--models", nargs="+", default=["LSTM", "TCN", "Transformer", "latent_ode"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    run_multiseed(args)


if __name__ == "__main__":
    main()
