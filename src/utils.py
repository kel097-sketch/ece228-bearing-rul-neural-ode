import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from config import METADATA_COLUMNS, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def numeric_sort_key(filename) -> tuple:
    name = Path(filename).stem
    parts = re.split(r"(\d+(?:\.\d+)?)", name)
    key = []
    for part in parts:
        if not part:
            continue
        try:
            key.append((0, float(part)))
        except ValueError:
            key.append((1, part.lower()))
    return tuple(key)


def safe_read_vibration_csv(path):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Vibration CSV does not exist: {csv_path}")

    read_errors = []
    read_attempts = [
        {"header": None},
        {"header": None, "sep": r"[\s,;]+", "engine": "python"},
        {"header": 0},
        {"header": 0, "sep": r"[\s,;]+", "engine": "python"},
    ]

    for kwargs in read_attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            if df.shape[1] < 2:
                raise ValueError(f"expected at least two columns, found {df.shape[1]}")

            two_cols = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
            two_cols = two_cols.dropna(axis=0, how="any")
            if two_cols.empty:
                raise ValueError("no numeric rows remained after parsing first two columns")

            values = two_cols.to_numpy(dtype=np.float32, copy=True)
            return values[:, 0], values[:, 1]
        except Exception as exc:
            read_errors.append(f"{kwargs}: {exc}")

    details = " | ".join(read_errors)
    raise RuntimeError(f"Failed to read vibration CSV {csv_path}: {details}")


def compute_mae_rmse_r2(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def append_result_csv(path, row_dict: dict) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(METADATA_COLUMNS)
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def load_split(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload: dict) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def project_path(path) -> Path:
    return Path(path).expanduser()
