"""Standalone script executed by the dedicated fasttreeshap interpreter (numpy<2),
not the main project's venv — see scripts/setup_fasttreeshap_env.sh. Deliberately
self-contained (no import of the Benchmarking package) so the two environments never
need to agree on anything beyond these few stdlib/pandas/numpy calls.

Usage:
    python _fasttreeshap_runner.py --model model.pkl --x x.csv --output out.csv \
        [--algorithm v2] [--interactions]
"""
import argparse
import pickle
import sys

import numpy as np
import pandas as pd


def reduce_multiclass(values, order=1):
    """Mirrors Benchmarking/backends/base_backend.py's reduce_multiclass — kept as a
    local copy since this script must stay import-free of the Benchmarking package
    (see module docstring)."""
    if isinstance(values, list):
        idx = 1 if len(values) == 2 else 0
        return np.asarray(values[idx])
    arr = np.asarray(values)
    if arr.ndim > order + 1:
        return arr[..., 1] if arr.shape[-1] == 2 else arr[..., 0]
    return arr


def flatten_interactions(values, x):
    """Mirrors base_backend.py's flatten_interactions (see reduce_multiclass docstring)."""
    n, d, _ = values.shape
    cols = [f"{a}__{b}" for a in x.columns for b in x.columns]
    return pd.DataFrame(values.reshape(n, d * d), index=x.index, columns=cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--x", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithm", default="v2")
    parser.add_argument("--interactions", action="store_true")
    args = parser.parse_args()

    import fasttreeshap

    with open(args.model, "rb") as f:
        model = pickle.load(f)
    x = pd.read_csv(args.x, index_col=0)

    # algorithm="v2" (the default, for speed) doesn't support interactions —
    # fasttreeshap itself falls back to v1 with a printed warning if asked, so fall
    # back explicitly instead to keep the choice visible and avoid the extra internal
    # v2-then-v1 detection round trip.
    algorithm = "v1" if (args.interactions and args.algorithm == "v2") else args.algorithm
    explainer = fasttreeshap.TreeExplainer(model, algorithm=algorithm, n_jobs=-1, shortcut=False)

    if args.interactions:
        values = reduce_multiclass(explainer.shap_interaction_values(x), order=2)
        result = flatten_interactions(values, x)
    else:
        values = reduce_multiclass(explainer(x).values, order=1)
        result = pd.DataFrame(values, index=x.index, columns=x.columns)

    result.to_csv(args.output)

    # Base value of the path-dependent game, class-selected like the values above
    # (binary -> class 1, multiclass -> class 0). Written to a sidecar the parent
    # backend reads back so the runner can check additivity against the game
    # fasttreeshap actually explains, not the marginal background baseline.
    ev = np.ravel(np.asarray(explainer.expected_value, dtype=float))
    base = float(ev[1] if ev.size == 2 else ev[0])
    with open(args.output + ".baseline", "w") as f:
        f.write(repr(base))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[BUG] fasttreeshap runner failed: {e.__class__.__name__}: {e}", file=sys.stderr)
        sys.exit(1)
