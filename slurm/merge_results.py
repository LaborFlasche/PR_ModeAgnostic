#!/usr/bin/env python3
"""Merges all per-task CSVs from Benchmarking/slurm_results/ into Benchmarking/results.csv."""
import glob
import sys
import pandas as pd

INPUT_GLOB = "Benchmarking/slurm_results/results_*.csv"
OUTPUT_CSV = "Benchmarking/results.csv"

files = sorted(glob.glob(INPUT_GLOB))
if not files:
    print(f"No files matched {INPUT_GLOB}", file=sys.stderr)
    sys.exit(1)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = df.drop_duplicates(
    subset=["dataset", "model", "n_features", "n_samples", "backend", "approximator", "budget"],
    keep="last",
)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Merged {len(files)} files → {len(df)} rows → {OUTPUT_CSV}")
