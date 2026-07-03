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

    # Model hyperparameters (e.g. max_depth swept in configs/config-tree.yaml) are part
    # of the identity key too, so different values don't collapse into one row. Only
    # included if present, since non-tree configs' models don't share these param names.
    hyperparam_cols = [c for c in ("n_estimators", "max_depth", "learning_rate") if c in df.columns]
    df = df.drop_duplicates(
        subset=["dataset", "model", "n_features", "n_samples", "seed", "n_background",
                "backend", "approximator", "budget"] + hyperparam_cols,
        keep="last",
    )

    df.to_csv(args.output_csv, index=False)
    print(f"Merged {len(files)} files -> {len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
