import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from utils import ensure_dir, save_json, set_seed


def sorted_bearings(df: pd.DataFrame, condition_id: str) -> list[str]:
    return sorted(df.loc[df["condition_id"] == condition_id, "bearing_id"].unique().tolist())


def split_bearings(bearings: list[str], seed: int = RANDOM_SEED) -> tuple[list[str], list[str], list[str]]:
    bearings = list(bearings)
    rng = np.random.default_rng(seed)
    rng.shuffle(bearings)
    n = len(bearings)
    if n == 0:
        return [], [], []
    if n == 1:
        return bearings, [], []
    if n == 2:
        return [bearings[0]], [], [bearings[1]]

    n_test = max(1, int(round(0.2 * n)))
    n_val = max(1, int(round(0.2 * n)))
    if n - n_val - n_test < 1:
        n_val = 1
        n_test = 1
    train = sorted(bearings[: n - n_val - n_test])
    val = sorted(bearings[n - n_val - n_test : n - n_test])
    test = sorted(bearings[n - n_test :])
    return train, val, test


def holdout_validation_for_conditions(df: pd.DataFrame, train_conditions: list[str], seed: int):
    train_bearings = []
    val_bearings = []
    for i, condition_id in enumerate(train_conditions):
        bearings = sorted_bearings(df, condition_id)
        rng = np.random.default_rng(seed + i)
        shuffled = list(bearings)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            val = []
            train = shuffled
        else:
            val = [shuffled[0]]
            train = shuffled[1:]
        train_bearings.extend(train)
        val_bearings.extend(val)
    return sorted(train_bearings), sorted(val_bearings)


def make_literature_aligned_lobo_splits(
    df: pd.DataFrame, condition_id: str, condition_index: int
) -> list[dict]:
    """Within-condition leave-one-bearing-out splits for paper-style comparison."""
    bearings = sorted_bearings(df, condition_id)
    splits = []
    for test_index, test_bearing in enumerate(bearings):
        candidates = [bearing for bearing in bearings if bearing != test_bearing]
        rng = np.random.default_rng(RANDOM_SEED + 100 * condition_index + test_index)
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            val_bearings = []
            train_bearings = sorted(shuffled)
        else:
            val_bearings = sorted([shuffled[0]])
            train_bearings = sorted(shuffled[1:])
        splits.append(
            {
                "protocol": "literature_aligned_lobo",
                "split_name": f"literature_{condition_id}_test_{test_bearing}",
                "train_bearings": train_bearings,
                "val_bearings": val_bearings,
                "test_bearings": [test_bearing],
                "train_conditions": [condition_id],
                "test_conditions": [condition_id],
            }
        )
    return splits


def write_split(out_dir: Path, split: dict) -> None:
    save_json(out_dir / f"{split['split_name']}.json", split)
    print(
        f"{split['split_name']}: "
        f"train={len(split['train_bearings'])}, "
        f"val={len(split['val_bearings'])}, "
        f"test={len(split['test_bearings'])}, "
        f"train_conditions={split['train_conditions']}, "
        f"test_conditions={split['test_conditions']}"
    )


def make_splits(features: str, out_dir: str) -> list[dict]:
    set_seed(RANDOM_SEED)
    features_df = pd.read_csv(features)
    out_path = ensure_dir(out_dir)
    splits = []
    conditions = sorted(features_df["condition_id"].unique().tolist())

    for i, condition_id in enumerate(conditions):
        train, val, test = split_bearings(sorted_bearings(features_df, condition_id), RANDOM_SEED + i)
        split = {
            "protocol": "within_condition",
            "split_name": f"within_{condition_id}",
            "train_bearings": train,
            "val_bearings": val,
            "test_bearings": test,
            "train_conditions": [condition_id],
            "test_conditions": [condition_id],
        }
        write_split(out_path, split)
        splits.append(split)

    for i, condition_id in enumerate(conditions):
        for split in make_literature_aligned_lobo_splits(features_df, condition_id, i):
            write_split(out_path, split)
            splits.append(split)

    mixed_train, mixed_val, mixed_test = [], [], []
    for i, condition_id in enumerate(conditions):
        train, val, test = split_bearings(sorted_bearings(features_df, condition_id), RANDOM_SEED + i)
        mixed_train.extend(train)
        mixed_val.extend(val)
        mixed_test.extend(test)
    split = {
        "protocol": "mixed_condition",
        "split_name": "mixed_condition",
        "train_bearings": sorted(mixed_train),
        "val_bearings": sorted(mixed_val),
        "test_bearings": sorted(mixed_test),
        "train_conditions": conditions,
        "test_conditions": conditions,
    }
    write_split(out_path, split)
    splits.append(split)

    for test_condition in conditions:
        train_conditions = [c for c in conditions if c != test_condition]
        train_bearings, val_bearings = holdout_validation_for_conditions(
            features_df, train_conditions, RANDOM_SEED
        )
        test_bearings = []
        for condition_id in [test_condition]:
            test_bearings.extend(sorted_bearings(features_df, condition_id))
        split_name = f"cross_train_{'_'.join(train_conditions)}_test_{test_condition}"
        split = {
            "protocol": "cross_condition",
            "split_name": split_name,
            "train_bearings": sorted(train_bearings),
            "val_bearings": sorted(val_bearings),
            "test_bearings": sorted(test_bearings),
            "train_conditions": train_conditions,
            "test_conditions": [test_condition],
        }
        write_split(out_path, split)
        splits.append(split)

    return splits


def parse_args():
    parser = argparse.ArgumentParser(description="Create bearing-level train/val/test splits.")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--out_dir", default="processed/splits")
    return parser.parse_args()


def main():
    args = parse_args()
    make_splits(args.features, args.out_dir)


if __name__ == "__main__":
    main()
