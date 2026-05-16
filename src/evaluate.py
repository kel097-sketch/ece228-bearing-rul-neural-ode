import argparse
from pathlib import Path

import pandas as pd

from config import ACTIVE_MODEL_ORDER
from utils import ensure_dir


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["mae", "rmse", "r2", "num_train_bearings", "num_val_bearings", "num_test_bearings"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.drop_duplicates(["protocol", "split_name", "model"], keep="last").copy()
    df = df[df["model"].isin(ACTIVE_MODEL_ORDER)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Using active model set; ignored {dropped} duplicate or inactive rows.")
    return df


def save_protocol_table(df: pd.DataFrame, protocol: str, path: Path) -> pd.DataFrame:
    table = df[df["protocol"] == protocol].sort_values("mae", ascending=True)
    table.to_csv(path, index=False)
    print(f"Saved {path} ({len(table)} rows)")
    return table


def save_average_table(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df.empty:
        avg = pd.DataFrame(columns=["model", "mae", "rmse", "r2", "num_splits"])
    else:
        avg = (
            df.groupby("model", as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                num_splits=("split_name", "nunique"),
            )
            .sort_values("mae", ascending=True)
        )
    avg.to_csv(path, index=False)
    print(f"Saved {path} ({len(avg)} rows)")
    return avg


def best_model(table: pd.DataFrame) -> str:
    if table.empty:
        return "not available"
    row = table.sort_values("mae", ascending=True).iloc[0]
    return f"{row['model']} (MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f})"


def compare_models(table: pd.DataFrame, model_a: str, model_b: str, metric: str = "mae") -> str:
    if table.empty:
        return f"{model_a} vs {model_b}: not available"
    grouped = table.groupby("model", as_index=False)[metric].mean()
    lookup = dict(zip(grouped["model"], grouped[metric]))
    if model_a not in lookup or model_b not in lookup:
        return f"{model_a} vs {model_b}: not available"
    a_value = lookup[model_a]
    b_value = lookup[model_b]
    if a_value < b_value:
        winner = model_a
        delta = b_value - a_value
    else:
        winner = model_b
        delta = a_value - b_value
    return (
        f"{model_a} vs {model_b}: {winner} lower average {metric.upper()} "
        f"by {delta:.4f} ({model_a}={a_value:.4f}, {model_b}={b_value:.4f})"
    )


def evaluate(results: str) -> None:
    results_path = Path(results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    out_dir = ensure_dir("results/tables")
    all_results = clean_results(pd.read_csv(results_path))

    within = save_protocol_table(
        all_results, "within_condition", out_dir / "within_condition_results.csv"
    )
    mixed = save_protocol_table(
        all_results, "mixed_condition", out_dir / "mixed_condition_results.csv"
    )
    cross = save_protocol_table(
        all_results, "cross_condition", out_dir / "cross_condition_results.csv"
    )
    literature = save_protocol_table(
        all_results, "literature_aligned_lobo", out_dir / "literature_aligned_results.csv"
    )

    cross_avg = save_average_table(cross, out_dir / "cross_condition_average_results.csv")
    literature_avg = save_average_table(
        literature, out_dir / "literature_aligned_average_results.csv"
    )

    print("\nConcise summary")
    print(f"Best mixed-condition model: {best_model(mixed)}")
    print(f"Best average cross-condition model: {best_model(cross_avg)}")
    print(f"Best literature-aligned LOO model: {best_model(literature_avg)}")
    comparison_table = cross if not cross.empty else mixed
    print(compare_models(comparison_table, "Ridge", "LSTM"))
    print(compare_models(comparison_table, "LSTM", "TCN"))
    print(compare_models(comparison_table, "TCN", "Transformer"))
    print(compare_models(comparison_table, "LSTM", "latent_ode"))
    print(compare_models(comparison_table, "latent_ode", "condition_aware_ode"))

    if not cross_avg.empty:
        print("\nMost important final table")
        display = cross_avg[["model", "mae", "rmse"]].rename(
            columns={
                "model": "Model",
                "mae": "Average Cross-condition MAE",
                "rmse": "Average Cross-condition RMSE",
            }
        )
        print(display.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize experiment results by protocol.")
    parser.add_argument("--results", default="results/tables/all_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args.results)


if __name__ == "__main__":
    main()
