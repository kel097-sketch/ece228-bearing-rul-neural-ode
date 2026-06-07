import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dir, save_json


def condition_bearings(features_df: pd.DataFrame, condition_id: str) -> list[str]:
    bearings = sorted(features_df.loc[features_df["condition_id"] == condition_id, "bearing_id"].unique())
    if len(bearings) != 5:
        raise ValueError(f"Expected 5 bearings for {condition_id}, found {len(bearings)}: {bearings}")
    return bearings


def write_split(out_dir: Path, split: dict) -> None:
    save_json(out_dir / f"{split['split_name']}.json", split)
    print(
        f"{split['split_name']}: train={len(split['train_bearings'])}, "
        f"val={len(split['val_bearings'])}, test={len(split['test_bearings'])}, "
        f"train_conditions={split['train_conditions']}, test_conditions={split['test_conditions']}"
    )


def make_final_splits(features: str, out_dir: str) -> list[dict]:
    features_df = pd.read_csv(features, usecols=["bearing_id", "condition_id"])
    out_path = ensure_dir(out_dir)

    condition_ids = ["C1", "C2", "C3"]
    bearings_by_condition = {condition: condition_bearings(features_df, condition) for condition in condition_ids}
    splits: list[dict] = []

    for test_condition in condition_ids:
        train_conditions = [condition for condition in condition_ids if condition != test_condition]
        train_bearings: list[str] = []
        val_bearings: list[str] = []
        for condition in train_conditions:
            bearings = bearings_by_condition[condition]
            train_bearings.extend(bearings[:4])
            val_bearings.append(bearings[4])
        split = {
            "protocol": "cross_condition",
            "split_name": f"cross_train_{'_'.join(train_conditions)}_test_{test_condition}",
            "train_bearings": sorted(train_bearings),
            "val_bearings": sorted(val_bearings),
            "test_bearings": sorted(bearings_by_condition[test_condition]),
            "train_conditions": train_conditions,
            "test_conditions": [test_condition],
        }
        write_split(out_path, split)
        splits.append(split)

    mixed_train: list[str] = []
    mixed_val: list[str] = []
    mixed_test: list[str] = []
    for condition in condition_ids:
        bearings = bearings_by_condition[condition]
        mixed_train.extend(bearings[:3])
        mixed_val.append(bearings[3])
        mixed_test.append(bearings[4])
    split = {
        "protocol": "mixed_condition",
        "split_name": "mixed_condition",
        "train_bearings": sorted(mixed_train),
        "val_bearings": sorted(mixed_val),
        "test_bearings": sorted(mixed_test),
        "train_conditions": condition_ids,
        "test_conditions": condition_ids,
    }
    write_split(out_path, split)
    splits.append(split)
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description="Create final leakage-free cross-condition and mixed-condition splits.")
    parser.add_argument("--features", default="processed/features.csv")
    parser.add_argument("--out_dir", default="processed/splits_final")
    return parser.parse_args()


def main():
    args = parse_args()
    make_final_splits(args.features, args.out_dir)


if __name__ == "__main__":
    main()
