import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils import compute_mae_rmse_r2, ensure_dir, get_feature_columns, load_split


EPS = 1e-12
DEFAULT_WEIGHTS = {"correlation": 0.5, "monotonicity": 0.4, "robustness": 0.1}


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) < EPS or np.std(y) < EPS:
        return 0.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def _monotonicity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    diffs = np.diff(x)
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    return float(abs(pos - neg) / max(len(diffs), 1))


def _robustness(x: np.ndarray, window: int = 9) -> float:
    x = pd.Series(np.asarray(x, dtype=float))
    if len(x) < 3:
        return 0.0
    smooth = x.rolling(window=window, center=True, min_periods=1).median()
    denom = np.maximum(np.abs(smooth.to_numpy()), EPS)
    rel_noise = np.abs(x.to_numpy() - smooth.to_numpy()) / denom
    return float(np.exp(-np.nanmean(rel_noise)))


def score_features(features_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        corr_values = []
        mono_values = []
        robust_values = []
        for _, bearing_df in features_df.groupby("bearing_id", sort=True):
            bearing_df = bearing_df.sort_values("time_index")
            values = bearing_df[feature].to_numpy(dtype=float)
            rul = bearing_df["normalized_rul"].to_numpy(dtype=float)
            corr_values.append(_safe_abs_corr(values, rul))
            mono_values.append(_monotonicity(values))
            robust_values.append(_robustness(values))
        corr = float(np.nanmean(corr_values))
        mono = float(np.nanmean(mono_values))
        robust = float(np.nanmean(robust_values))
        total = (
            DEFAULT_WEIGHTS["correlation"] * corr
            + DEFAULT_WEIGHTS["monotonicity"] * mono
            + DEFAULT_WEIGHTS["robustness"] * robust
        )
        rows.append(
            {
                "feature": feature,
                "group": feature_group(feature),
                "correlation": corr,
                "monotonicity": mono,
                "robustness": robust,
                "total_score": float(total),
            }
        )
    return pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)


def feature_group(feature: str) -> str:
    if "_wpt_" in feature or "_wav_" in feature:
        return "wavelet"
    if "spectral" in feature or "band_" in feature or "dominant_frequency" in feature:
        return "frequency"
    return "time"


def columns_for_group(feature_cols: list[str], scores: pd.DataFrame, group_name: str, top_k: int) -> list[str]:
    if group_name == "all":
        return feature_cols
    if group_name == "original":
        return [col for col in feature_cols if feature_group(col) != "wavelet"]
    if group_name == "selected_top":
        selected = scores.head(top_k)["feature"].tolist()
        return [col for col in selected if col in feature_cols]
    return [col for col in feature_cols if feature_group(col) == group_name]


def subset_by_bearings(df: pd.DataFrame, bearings: list[str]) -> pd.DataFrame:
    return df[df["bearing_id"].isin(bearings)].copy()


def train_ridge_group(
    features_df: pd.DataFrame,
    split: dict,
    feature_cols: list[str],
) -> dict:
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
    metrics["alpha"] = best["alpha"]
    return metrics


def run_feature_group_sensitivity(
    features_df: pd.DataFrame,
    scores: pd.DataFrame,
    split_dir: str,
    out_dir: Path,
    protocol: str,
    top_k: int,
) -> pd.DataFrame:
    feature_cols = get_feature_columns(features_df)
    rows = []
    group_names = ["original", "time", "frequency", "wavelet", "all", "selected_top"]
    for split_path in sorted(Path(split_dir).glob("*.json")):
        split = load_split(split_path)
        if protocol != "all" and split.get("protocol") != protocol:
            continue
        train_scores = score_features(subset_by_bearings(features_df, split["train_bearings"]), feature_cols)
        for group_name in group_names:
            split_scores = train_scores if group_name == "selected_top" else scores
            cols = columns_for_group(feature_cols, split_scores, group_name, top_k)
            if not cols:
                continue
            print(f"Feature group Ridge: split={split['split_name']}, group={group_name}, features={len(cols)}")
            metrics = train_ridge_group(features_df, split, cols)
            rows.append(
                {
                    "protocol": split.get("protocol", ""),
                    "split_name": split.get("split_name", split_path.stem),
                    "feature_group": group_name,
                    "num_features": len(cols),
                    "alpha": metrics["alpha"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "selected_features": json.dumps(cols, ensure_ascii=True),
                }
            )
    results = pd.DataFrame(rows)
    out_path = out_dir / "feature_group_sensitivity_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(results)} rows)")
    if not results.empty:
        avg = (
            results.groupby(["protocol", "feature_group"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                num_features=("num_features", "mean"),
                num_splits=("split_name", "nunique"),
            )
            .sort_values(["protocol", "mae"])
        )
        avg_path = out_dir / "feature_group_sensitivity_average_results.csv"
        avg.to_csv(avg_path, index=False)
        print(f"Saved {avg_path} ({len(avg)} rows)")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Score degradation features and run feature group ablations.")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--split_dir", default="processed/splits")
    parser.add_argument("--out_dir", default="results/tables")
    parser.add_argument("--protocol", default="cross_condition", choices=["all", "within_condition", "mixed_condition", "cross_condition", "literature_aligned_lobo"])
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--skip_group_sensitivity", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    features_df = pd.read_csv(args.features)
    feature_cols = get_feature_columns(features_df)
    out_dir = ensure_dir(args.out_dir)
    scores = score_features(features_df, feature_cols)
    scores_path = out_dir / "feature_scores.csv"
    scores.to_csv(scores_path, index=False)
    print(f"Saved {scores_path} ({len(scores)} rows)")
    selected_path = out_dir / f"selected_features_top{args.top_k}.txt"
    selected_path.write_text("\n".join(scores.head(args.top_k)["feature"].tolist()), encoding="utf-8")
    print(f"Saved {selected_path}")
    if not args.skip_group_sensitivity:
        run_feature_group_sensitivity(features_df, scores, args.split_dir, out_dir, args.protocol, args.top_k)


if __name__ == "__main__":
    main()
