import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from feature_analysis import feature_group, score_features
from k_sensitivity import load_sequence, model_instance, train_discrete_model, train_ode_model
from make_sequences import make_sequence_file
from utils import compute_mae_rmse_r2, ensure_dir, get_feature_columns, load_split, set_seed


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


def subset_by_bearings(df: pd.DataFrame, bearing_ids: list[str]) -> pd.DataFrame:
    return df[df["bearing_id"].isin(bearing_ids)].copy()


def selected_columns_for_setting(
    setting: str,
    original_df: pd.DataFrame,
    expanded_df: pd.DataFrame,
    split: dict,
    top_k: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    if setting == "original":
        feature_cols = [col for col in get_feature_columns(original_df) if feature_group(col) != "wavelet"]
        return original_df[META_COLUMNS + feature_cols].copy(), feature_cols, pd.DataFrame()

    expanded_feature_cols = get_feature_columns(expanded_df)
    if setting == "all_expanded":
        return expanded_df[META_COLUMNS + expanded_feature_cols].copy(), expanded_feature_cols, pd.DataFrame()

    if setting == "wavelet_only":
        wavelet_cols = [col for col in expanded_feature_cols if feature_group(col) == "wavelet"]
        return expanded_df[META_COLUMNS + wavelet_cols].copy(), wavelet_cols, pd.DataFrame()

    if setting == "selected_top":
        train_df = subset_by_bearings(expanded_df, split["train_bearings"])
        scores = score_features(train_df, expanded_feature_cols)
        selected = scores.head(top_k)["feature"].tolist()
        return expanded_df[META_COLUMNS + selected].copy(), selected, scores

    raise ValueError(f"Unknown feature setting: {setting}")


def train_ridge_for_split(features_df: pd.DataFrame, split: dict, feature_cols: list[str], seed: int) -> dict:
    set_seed(seed)
    train_df = subset_by_bearings(features_df, split["train_bearings"])
    val_df = subset_by_bearings(features_df, split["val_bearings"])
    test_df = subset_by_bearings(features_df, split["test_bearings"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    y_train = train_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_val = val_df["normalized_rul"].to_numpy(dtype=np.float32)
    y_test = test_df["normalized_rul"].to_numpy(dtype=np.float32)

    best = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        val_metrics = compute_mae_rmse_r2(y_val, model.predict(X_val))
        if best is None or val_metrics["mae"] < best["val_mae"]:
            best = {"model": model, "alpha": alpha, "val_mae": val_metrics["mae"]}

    pred = best["model"].predict(X_test)
    metrics = compute_mae_rmse_r2(y_test, pred)
    metrics["best_epoch"] = 0
    metrics["best_val_mae"] = best["val_mae"]
    metrics["alpha"] = best["alpha"]
    return metrics


def train_deep_model(seq_path: Path, model_name: str, args, seed: int) -> dict:
    data, _ = load_sequence(seq_path)
    if len(data["X_train"]) == 0 or len(data["X_val"]) == 0 or len(data["X_test"]) == 0:
        raise ValueError(f"Empty sequence arrays in {seq_path}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_instance(model_name, data["X_train"].shape[-1]).to(device)
    train_args = SimpleNamespace(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        smooth_weight=args.smooth_weight,
    )
    if model_name in {"latent_ode", "condition_aware_ode"}:
        return train_ode_model(model, data, train_args, device, seed)
    return train_discrete_model(model, data, train_args, device, seed)


def summarize_results(results: pd.DataFrame, out_path: Path) -> None:
    if results.empty:
        return
    avg = (
        results.groupby(["protocol", "feature_setting", "model"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            mae_std=("mae", "std"),
            num_features=("num_features", "mean"),
            num_splits=("split_name", "nunique"),
            best_epoch=("best_epoch", "mean"),
            best_val_mae=("best_val_mae", "mean"),
        )
        .sort_values(["protocol", "feature_setting", "mae"])
    )
    avg_path = out_path.with_name("selected_feature_retraining_average_results.csv")
    avg.to_csv(avg_path, index=False)
    print(f"Saved {avg_path} ({len(avg)} rows)")

    pivot = avg.pivot_table(index="model", columns="feature_setting", values="mae", aggfunc="mean")
    pivot_path = out_path.with_name("selected_feature_retraining_mae_pivot.csv")
    pivot.to_csv(pivot_path)
    print(f"Saved {pivot_path}")


def run_experiment(args) -> pd.DataFrame:
    original_df = pd.read_csv(args.original_features)
    expanded_df = pd.read_csv(args.expanded_features)
    split_paths = []
    for split_path in sorted(Path(args.split_dir).glob("*.json")):
        split = load_split(split_path)
        if args.protocol == "all" or split.get("protocol") == args.protocol:
            split_paths.append(split_path)
    if not split_paths:
        raise FileNotFoundError(f"No splits found for protocol={args.protocol}")

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    selected_rows = []
    result_rows = []

    sequence_root = ensure_dir(args.sequence_root)
    for split_path in split_paths:
        split = load_split(split_path)
        split_name = split.get("split_name", split_path.stem)
        for setting in args.feature_settings:
            features_df, feature_cols, scores = selected_columns_for_setting(
                setting, original_df, expanded_df, split, args.top_k
            )
            if not feature_cols:
                print(f"Skipping {setting} on {split_name}; no feature columns selected.")
                continue
            if setting == "selected_top" and not scores.empty:
                for rank, row in scores.head(args.top_k).reset_index(drop=True).iterrows():
                    selected_rows.append(
                        {
                            "protocol": split.get("protocol", ""),
                            "split_name": split_name,
                            "feature_setting": setting,
                            "rank": rank + 1,
                            "feature": row["feature"],
                            "group": row["group"],
                            "correlation": row["correlation"],
                            "monotonicity": row["monotonicity"],
                            "robustness": row["robustness"],
                            "total_score": row["total_score"],
                        }
                    )

            setting_dir = ensure_dir(sequence_root / f"{setting}_top{args.top_k}" / f"k{args.k}")
            seq_path = setting_dir / f"{split_name}_k{args.k}.npz"
            if args.rebuild_sequences or not seq_path.exists():
                seq_path = make_sequence_file(features_df, split_path, args.k, setting_dir)

            print(
                f"Feature-setting retraining: split={split_name}, setting={setting}, "
                f"features={len(feature_cols)}, models={','.join(args.models)}"
            )
            for model_name in args.models:
                seed = args.seed + sum(ord(ch) for ch in f"{split_name}:{setting}:{model_name}")
                if model_name == "Ridge":
                    metrics = train_ridge_for_split(features_df, split, feature_cols, seed)
                else:
                    metrics = train_deep_model(seq_path, model_name, args, seed)
                result_rows.append(
                    {
                        "protocol": split.get("protocol", ""),
                        "split_name": split_name,
                        "feature_setting": setting,
                        "model": model_name,
                        "num_features": len(feature_cols),
                        "top_k": args.top_k if setting == "selected_top" else "",
                        "k": args.k,
                        "seed": seed,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "best_epoch": metrics.get("best_epoch", 0),
                        "best_val_mae": metrics.get("best_val_mae", np.nan),
                        "extra": json.dumps(
                            {key: value for key, value in metrics.items() if key not in {"mae", "rmse", "r2", "best_epoch", "best_val_mae"}},
                            ensure_ascii=True,
                        ),
                    }
                )
                pd.DataFrame(result_rows).to_csv(out_path, index=False)
                print(
                    f"  {model_name}: MAE={metrics['mae']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
                )

    results = pd.DataFrame(result_rows)
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(results)} rows)")
    summarize_results(results, out_path)

    if selected_rows:
        selected_path = out_path.with_name("selected_feature_retraining_selected_features.csv")
        pd.DataFrame(selected_rows).to_csv(selected_path, index=False)
        print(f"Saved {selected_path} ({len(selected_rows)} rows)")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain all RUL models under selected feature settings.")
    parser.add_argument("--original_features", default="processed/features.csv")
    parser.add_argument("--expanded_features", default="processed/features_wavelet.csv")
    parser.add_argument("--split_dir", default="processed/splits")
    parser.add_argument("--sequence_root", type=Path, default=Path("processed/sequences_feature_settings"))
    parser.add_argument("--out", default="results/tables/selected_feature_retraining_results.csv")
    parser.add_argument("--protocol", default="cross_condition")
    parser.add_argument(
        "--feature_settings",
        nargs="+",
        default=["original", "wavelet_only", "all_expanded", "selected_top"],
        choices=["original", "wavelet_only", "all_expanded", "selected_top"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"],
        choices=["Ridge", "LSTM", "TCN", "Transformer", "latent_ode"],
    )
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild_sequences", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
