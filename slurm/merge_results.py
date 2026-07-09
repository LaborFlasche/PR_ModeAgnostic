#!/usr/bin/env python3
"""Merges all per-task CSVs from an input directory into a single output CSV.

Usage: python slurm/merge_results.py [--input-dir DIR] [--output-csv FILE]
Defaults match the original single-config layout (Benchmarking/slurm_results/
-> Benchmarking/results.csv); pass both when merging a tree-config run, whose
array tasks write to a separate directory (see submit.sh).
"""
import argparse
import glob
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="Benchmarking/slurm_results")
    parser.add_argument("--output-csv", default="Benchmarking/results.csv")
    args = parser.parse_args()

    input_glob = f"{args.input_dir}/results_*.csv"
    files = sorted(glob.glob(input_glob))
    if not files:
        print(f"No files matched {input_glob}", file=sys.stderr)
        sys.exit(1)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Identity = every column that is not a per-run output. Allowlisting
    # hyperparam columns previously collapsed runs that differed in any other
    # swept param (min_samples_split, alpha, NN params, ...) and ignored
    # `order`; deriving the key from the schema keeps every run_meta and sweep
    # column in the identity automatically.
    output_cols = {
        "library", "computation_type", "n_eval", "runtime_s", "n_model_evals",
        "additivity_gap", "relative_additivity_gap", "shapley_values",
        "shapley_n_eval", "shapley_n_features", "pairwise_metrics",
    }
    identity_cols = [c for c in df.columns if c not in output_cols]
    df = df.drop_duplicates(subset=identity_cols, keep="last")

    df.to_csv(args.output_csv, index=False)
    print(f"Merged {len(files)} files -> {len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
