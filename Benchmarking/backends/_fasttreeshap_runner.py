"""Standalone script executed by the dedicated fasttreeshap interpreter (numpy<2),
not the main project's venv — see scripts/setup_fasttreeshap_env.sh. Deliberately
self-contained (no import of the Benchmarking package) so the two environments never
need to agree on anything beyond these few stdlib/pandas/numpy calls.

Usage:
    python _fasttreeshap_runner.py --model model.pkl --x x.csv --output out.csv \
        [--algorithm v2]
"""
import argparse
import pickle
import sys

import numpy as np
import pandas as pd


def reduce_multiclass(values):
    if isinstance(values, list):
        idx = 1 if len(values) == 2 else 0
        return np.asarray(values[idx])
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] == 2 else arr[:, :, 0]
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--x", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithm", default="v2")
    args = parser.parse_args()

    import fasttreeshap

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    x = pd.read_csv(args.x, index_col=0)

    explainer = fasttreeshap.TreeExplainer(model, algorithm=args.algorithm, n_jobs=-1, shortcut=False)
    values = reduce_multiclass(explainer(x).values)

    pd.DataFrame(values, index=x.index, columns=x.columns).to_csv(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[BUG] fasttreeshap runner failed: {e.__class__.__name__}: {e}", file=sys.stderr)
        sys.exit(1)
