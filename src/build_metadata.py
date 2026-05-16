import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from config import CONDITION_MAP, METADATA_COLUMNS, RAW_DATA_DIR
from utils import ensure_dir, numeric_sort_key


def infer_condition(path: Path):
    text = str(path).replace("\\", "/")
    for condition_name, meta in CONDITION_MAP.items():
        if condition_name in text:
            return condition_name, meta
    return None, None


def readable_bearing_id(condition_id: str, folder_name: str, used_ids: set[str]) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in folder_name)
    base = f"{condition_id}_{safe_name}"
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def path_for_csv(csv_path: Path, root: Path) -> str:
    try:
        return str(csv_path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(csv_path.resolve())


def build_metadata(raw_dir: str, out: str) -> pd.DataFrame:
    root = Path.cwd()
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    bearing_dirs = sorted({p.parent for p in raw_path.rglob("*.csv")}, key=lambda p: str(p))
    rows = []
    used_ids = set()
    skipped_dirs = []

    for bearing_dir in bearing_dirs:
        condition_name, condition_meta = infer_condition(bearing_dir)
        if condition_meta is None:
            skipped_dirs.append(str(bearing_dir))
            continue

        csv_files = sorted(bearing_dir.glob("*.csv"), key=numeric_sort_key)
        if not csv_files:
            continue

        bearing_id = readable_bearing_id(condition_meta["condition_id"], bearing_dir.name, used_ids)
        failure_time = len(csv_files) - 1

        for time_index, csv_file in enumerate(csv_files):
            rul = failure_time - time_index
            normalized_rul = float(rul / failure_time) if failure_time > 0 else 0.0
            rows.append(
                {
                    "bearing_id": bearing_id,
                    "condition_id": condition_meta["condition_id"],
                    "speed_rpm": condition_meta["speed_rpm"],
                    "load_kn": condition_meta["load_kn"],
                    "file_path": path_for_csv(csv_file, root),
                    "file_index": time_index,
                    "time_index": time_index,
                    "failure_time": failure_time,
                    "rul": rul,
                    "normalized_rul": normalized_rul,
                }
            )

    if not rows:
        raise RuntimeError(f"No CSV files with recognized conditions were found under {raw_path}")

    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    out_path = Path(out)
    ensure_dir(out_path.parent)
    metadata.to_csv(out_path, index=False)

    files_per_bearing = metadata.groupby("bearing_id").size().sort_index()
    condition_count = metadata["condition_id"].nunique()
    bearing_count = metadata["bearing_id"].nunique()
    csv_count = len(metadata)
    condition_files = Counter(metadata["condition_id"])

    print(f"Saved metadata to {out_path}")
    print(f"Number of conditions found: {condition_count}")
    print(f"Number of bearings: {bearing_count}")
    print(f"Number of CSV files: {csv_count}")
    print("Files per condition:")
    for condition_id, count in sorted(condition_files.items()):
        print(f"  {condition_id}: {count}")
    print("Files per bearing:")
    for bearing_id, count in files_per_bearing.items():
        print(f"  {bearing_id}: {count}")
        if count < 10:
            print(f"  WARNING: {bearing_id} has very few files ({count})")
    if skipped_dirs:
        print("WARNING: skipped CSV folders with unknown condition:")
        for item in skipped_dirs[:20]:
            print(f"  {item}")
        if len(skipped_dirs) > 20:
            print(f"  ... {len(skipped_dirs) - 20} more")

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Build bearing-level metadata for XJTU-SY CSV files.")
    parser.add_argument("--raw_dir", default=RAW_DATA_DIR)
    parser.add_argument("--out", default="processed/metadata.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    build_metadata(args.raw_dir, args.out)


if __name__ == "__main__":
    main()
