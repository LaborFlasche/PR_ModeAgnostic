#!/usr/bin/env python3
"""
Rebuild every row's pairwise_metrics from the stored shapley_values vectors.

Result CSVs written before the spec-qualified pairwise keys existed (see
benchmarking.runner.spec_key) keyed entries by backend class name only, so
approximation specs of the same library (kernel vs permutation, different
budgets) overwrote each other's entries. The stored Shapley matrices are
complete, so nothing needs re-running: this script regroups rows into their
benchmark cells and recomputes the full pairwise dict per row with the same
keys a fresh run would emit.

Order-2 interaction cells pass through untouched — they only ever ran
distinctly-named true-value backends, so their dicts were never corrupted.

Self-check: the old collided entries were valid comparisons, just ambiguously
keyed, so every stored entry must match one recomputed entry of the same
backend (reported as max |stored - recomputed| mean_abs_diff, should be ~0;
NaN-valued stored entries are skipped).

Usage:
    python scripts/recompute_pairwise_metrics.py results.csv -o results_fixed.csv
    python scripts/recompute_pairwise_metrics.py results.csv --in-place
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarking.metrics import (  # noqa: E402
    mean_abs_diff,
    relative_mae,
    sign_agreement,
    mean_sample_rho,
)
from benchmarking.runner import spec_key  # noqa: E402

# Per-backend output columns (same split as slurm/merge_results.py); everything
# else identifies the cell, except the trio naming the spec within it.
OUTPUT_COLS = {
    "library", "computation_type", "n_eval", "runtime_s", "n_model_evals",
    "additivity_gap", "relative_additivity_gap", "shapley_values",
    "shapley_n_eval", "shapley_n_features", "pairwise_metrics",
}
ROW_ID_COLS = {"backend", "approximator", "budget"}


def contrib_frame(row: pd.Series) -> pd.DataFrame:
    """Rebuild the (n_eval, n_features) contribution DataFrame the runner fed
    to the metric functions. Column labels are positional: within one cell all
    backends explained the same X_eval, so positions line up."""
    values = np.array(json.loads(row["shapley_values"]), dtype=float)
    n, d = int(row["shapley_n_eval"]), int(row["shapley_n_features"])
    return pd.DataFrame(values.reshape(n, d))


def _pairwise_entry(candidate: pd.DataFrame, reference: pd.DataFrame, is_self: bool) -> dict:
    if is_self:
        return {
            "mean_abs_diff": 0.0,
            "relative_mae": 0.0,
            "sign_agreement": float(sign_agreement(candidate, candidate)),
            "mean_sample_rho": 1.0,
        }
    return {
        "mean_abs_diff": mean_abs_diff(candidate, reference),
        "relative_mae": relative_mae(candidate, reference),
        "sign_agreement": sign_agreement(candidate, reference),
        "mean_sample_rho": mean_sample_rho(candidate, reference),
    }


def _self_check(stored_json: str, recomputed: dict) -> float:
    """Max deviation between each stored entry and its closest recomputed
    entry for the same backend name (old keys were the bare name; new keys
    are prefixed with it)."""
    worst = 0.0
    for old_key, old_entry in json.loads(stored_json).items():
        old_mad = old_entry.get("mean_abs_diff")
        if old_mad is None or (isinstance(old_mad, float) and np.isnan(old_mad)):
            continue
        candidates = [v["mean_abs_diff"] for k, v in recomputed.items()
                      if k == old_key or k.startswith(old_key + "|")]
        candidates = [c for c in candidates if not np.isnan(c)]
        if candidates:
            worst = max(worst, min(abs(old_mad - c) for c in candidates))
    return worst


def recompute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a copy of df with pairwise_metrics rebuilt for all order-1 cells,
    plus a report dict (cells/rows processed, self-check max deviation)."""
    df = df.copy()
    cell_cols = [c for c in df.columns if c not in OUTPUT_COLS | ROW_ID_COLS]
    report = {"cells": 0, "rows": 0, "skipped_order2": 0, "max_deviation": 0.0}

    for _, cell in df.groupby(cell_cols, dropna=False):
        if "order" in cell.columns and (cell["order"] == 2).any():
            report["skipped_order2"] += len(cell)
            continue
        contribs = {}  # spec key -> contribution frame, insertion order = row order
        for idx, row in cell.iterrows():
            contribs[idx] = (
                spec_key(row["backend"], row["approximator"], row["budget"]),
                contrib_frame(row),
            )
        for idx, row in cell.iterrows():
            _, own = contribs[idx]
            pairwise = {
                key: _pairwise_entry(own, ref, is_self=ref_idx == idx)
                for ref_idx, (key, ref) in contribs.items()
            }
            report["max_deviation"] = max(
                report["max_deviation"],
                _self_check(row["pairwise_metrics"], pairwise),
            )
            df.at[idx, "pairwise_metrics"] = json.dumps(pairwise)
            report["rows"] += 1
        report["cells"] += 1

    return df, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("input_csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-o", "--output-csv", help="write the fixed CSV here")
    group.add_argument("--in-place", action="store_true",
                       help="overwrite the input CSV")
    args = parser.parse_args()

    output_csv = args.input_csv if args.in_place else args.output_csv
    if not args.in_place and Path(output_csv).resolve() == Path(args.input_csv).resolve():
        parser.error("output equals input; pass --in-place to overwrite")

    df = pd.read_csv(args.input_csv)
    fixed, report = recompute(df)
    fixed.to_csv(output_csv, index=False)
    print(f"cells recomputed:  {report['cells']} ({report['rows']} rows)")
    if report["skipped_order2"]:
        print(f"order-2 rows kept as-is: {report['skipped_order2']}")
    print(f"self-check: max |stored - recomputed| mean_abs_diff = "
          f"{report['max_deviation']:.3g} (should be ~0)")
    print(f"-> {output_csv}")


if __name__ == "__main__":
    main()
