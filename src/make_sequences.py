import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import ensure_dir, get_feature_columns, load_split


def build_samples(df: pd.DataFrame, feature_cols: list[str], k: int):
    X, y, c, tau, meta = [], [], [], [], []
    time_vector = np.linspace(0.0, 1.0, k, dtype=np.float32)

    for bearing_id, bearing_df in df.groupby("bearing_id", sort=True):
        bearing_df = bearing_df.sort_values("time_index")
        features = bearing_df[feature_cols].to_numpy(dtype=np.float32)
        targets = bearing_df["normalized_rul"].to_numpy(dtype=np.float32)
        conditions = bearing_df[["speed_rpm", "load_kn"]].to_numpy(dtype=np.float32)
        condition_ids = bearing_df["condition_id"].astype(str).to_numpy()
        time_indices = bearing_df["time_index"].to_numpy(dtype=np.int64)

        if len(bearing_df) < k:
            continue
        for end in range(k - 1, len(bearing_df)):
            start = end - k + 1
            X.append(features[start : end + 1])
            y.append(targets[end])
            c.append(conditions[end])
            tau.append(time_vector)
            meta.append((str(bearing_id), str(condition_ids[end]), int(time_indices[end])))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)
    meta = np.asarray(meta, dtype=object)
    return X, y, c, tau, meta


def fit_condition_scaler(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = train_df[["speed_rpm", "load_kn"]].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def normalize_conditions(c: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if c.size == 0:
        return c.astype(np.float32)
    return ((c - mean) / std).astype(np.float32)


def split_df(features_df: pd.DataFrame, bearing_ids: list[str]) -> pd.DataFrame:
    return features_df[features_df["bearing_id"].isin(bearing_ids)].copy()


def make_sequence_file(features_df: pd.DataFrame, split_path: Path, k: int, out_dir: Path) -> Path:
    split = load_split(split_path)
    feature_cols = get_feature_columns(features_df)
    train_df = split_df(features_df, split["train_bearings"])
    val_df = split_df(features_df, split["val_bearings"])
    test_df = split_df(features_df, split["test_bearings"])

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    scaled_df = features_df.copy()
    scaled_df[feature_cols] = scaler.transform(features_df[feature_cols].to_numpy(dtype=np.float32))
    train_scaled = split_df(scaled_df, split["train_bearings"])
    val_scaled = split_df(scaled_df, split["val_bearings"])
    test_scaled = split_df(scaled_df, split["test_bearings"])

    condition_mean, condition_std = fit_condition_scaler(train_df)

    X_train, y_train, c_train, tau_train, meta_train = build_samples(train_scaled, feature_cols, k)
    X_val, y_val, c_val, tau_val, meta_val = build_samples(val_scaled, feature_cols, k)
    X_test, y_test, c_test, tau_test, meta_test = build_samples(test_scaled, feature_cols, k)

    c_norm_train = normalize_conditions(c_train, condition_mean, condition_std)
    c_norm_val = normalize_conditions(c_val, condition_mean, condition_std)
    c_norm_test = normalize_conditions(c_test, condition_mean, condition_std)

    out_path = out_dir / f"{split['split_name']}_k{k}.npz"
    np.savez_compressed(
        out_path,
        X_train=X_train,
        y_train=y_train,
        c_train=c_train,
        c_norm_train=c_norm_train,
        tau_train=tau_train,
        meta_train=meta_train,
        X_val=X_val,
        y_val=y_val,
        c_val=c_val,
        c_norm_val=c_norm_val,
        tau_val=tau_val,
        meta_val=meta_val,
        X_test=X_test,
        y_test=y_test,
        c_test=c_test,
        c_norm_test=c_norm_test,
        tau_test=tau_test,
        meta_test=meta_test,
        feature_names=np.asarray(feature_cols, dtype=object),
        split_json=json.dumps(split),
        condition_mean=condition_mean,
        condition_std=condition_std,
    )
    print(
        f"Saved {out_path}: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}, features={len(feature_cols)}"
    )
    return out_path


def make_sequences(features: str, split_dir: str, k: int, out_dir: str) -> list[Path]:
    features_df = pd.read_csv(features)
    output_dir = ensure_dir(out_dir)
    split_paths = sorted(Path(split_dir).glob("*.json"))
    if not split_paths:
        raise FileNotFoundError(f"No split JSON files found in {split_dir}")
    outputs = []
    for split_path in tqdm(split_paths, desc="Building sequences"):
        outputs.append(make_sequence_file(features_df, split_path, k, output_dir))
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed-length bearing sequences.")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--split_dir", default="processed/splits")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out_dir", default="processed/sequences")
    return parser.parse_args()


def main():
    args = parse_args()
    make_sequences(args.features, args.split_dir, args.k, args.out_dir)


if __name__ == "__main__":
    main()
