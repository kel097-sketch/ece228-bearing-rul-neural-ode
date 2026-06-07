import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import ensure_dir


def resolve_csv_path(file_path: str, metadata_path: Path) -> Path:
    path = Path(file_path)
    if path.exists():
        return path
    root_relative = Path.cwd() / path
    if root_relative.exists():
        return root_relative
    metadata_relative = metadata_path.parent / path
    if metadata_relative.exists():
        return metadata_relative
    return path


def read_two_channel_csv(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, usecols=(0, 1))
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("expected a two-column vibration file")
        return values[:, :2]
    except Exception:
        df = pd.read_csv(path)
        values = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError(f"failed to parse two vibration channels from {path}")
        return values[:, :2]


def uniform_downsample(x: np.ndarray, points: int) -> np.ndarray:
    if len(x) == points:
        return x.astype(np.float32, copy=False)
    indices = np.linspace(0, len(x) - 1, points).round().astype(np.int64)
    return x[indices].astype(np.float32, copy=False)


def extract_raw_downsample(metadata: str, out: str, points_per_channel: int) -> pd.DataFrame:
    metadata_path = Path(metadata)
    meta_df = pd.read_csv(metadata_path)
    rows = []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Extracting raw downsample vectors"):
        row_dict = row.to_dict()
        csv_path = resolve_csv_path(str(row_dict["file_path"]), metadata_path)
        values = read_two_channel_csv(csv_path)
        horizontal = uniform_downsample(values[:, 0], points_per_channel)
        vertical = uniform_downsample(values[:, 1], points_per_channel)
        for i, value in enumerate(horizontal):
            row_dict[f"h_raw_{i:04d}"] = float(value)
        for i, value in enumerate(vertical):
            row_dict[f"v_raw_{i:04d}"] = float(value)
        rows.append(row_dict)

    out_path = Path(out)
    ensure_dir(out_path.parent)
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_path, index=False)
    print(f"Saved raw downsample features to {out_path}")
    print(f"Rows: {len(raw_df)}, raw dimensions: {points_per_channel * 2}")
    return raw_df


def parse_args():
    parser = argparse.ArgumentParser(description="Extract downsampled raw horizontal/vertical waveform vectors.")
    parser.add_argument("--metadata", default="processed/metadata.csv")
    parser.add_argument("--out", default="processed/features_raw_downsample_256.csv")
    parser.add_argument("--points_per_channel", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    extract_raw_downsample(args.metadata, args.out, args.points_per_channel)


if __name__ == "__main__":
    main()
