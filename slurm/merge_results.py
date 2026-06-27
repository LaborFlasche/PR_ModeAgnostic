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

    dedup_cols = ["dataset", "model", "n_features", "n_samples", "backend", "approximator", "budget"]
    if "n_background" in df.columns:
        dedup_cols.append("n_background")
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    df.to_csv(args.output_csv, index=False)
    print(f"Merged {len(files)} files -> {len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
