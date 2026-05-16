import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n>>> {printable}", flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the full bearing RUL experiment pipeline.")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--skip_ml", action="store_true")
    parser.add_argument("--skip_deep", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--run_sparse", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_features:
        run([py, "src/build_metadata.py", "--raw_dir", "data", "--out", "processed/metadata.csv"])
        run([py, "src/extract_features.py", "--metadata", "processed/metadata.csv", "--out", "processed/features.csv"])

    run([py, "src/make_splits.py", "--features", "processed/features.csv", "--out_dir", "processed/splits"])

    if not args.skip_ml:
        run([py, "src/train_ml.py", "--features", "processed/features.csv", "--split_dir", "processed/splits"])

    if not args.skip_deep:
        run(
            [
                py,
                "src/make_sequences.py",
                "--features",
                "processed/features.csv",
                "--split_dir",
                "processed/splits",
                "--k",
                "10",
                "--out_dir",
                "processed/sequences",
            ]
        )
        run([py, "src/train_lstm.py", "--seq_dir", "processed/sequences"])
        run([py, "src/train_tcn.py", "--seq_dir", "processed/sequences"])
        run([py, "src/train_transformer.py", "--seq_dir", "processed/sequences"])
        run([py, "src/train_ode.py", "--seq_dir", "processed/sequences", "--model", "latent_ode"])
        run([py, "src/train_ode.py", "--seq_dir", "processed/sequences", "--model", "condition_aware_ode"])

    run([py, "src/evaluate.py", "--results", "results/tables/all_results.csv"])

    if args.run_sparse:
        run([py, "src/sparse_observation.py", "--seq_dir", "processed/sequences"])

    if not args.skip_plots:
        run([py, "src/plot_results.py", "--all"])


if __name__ == "__main__":
    main()
