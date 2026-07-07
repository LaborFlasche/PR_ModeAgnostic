#!/usr/bin/env python3
"""Merge the fasttreeshap repair sweep back into the main tree results CSV.

The original results_config-tree.csv has all-NaN fasttreeshap rows (the
dedicated venv was missing on the cluster, see BUGS_TO_FIX.md Bug 5). The
repair sweep (configs/config-tree-fasttreeshap.yaml) reran only the
fasttreeshap backends plus the oracles. This script:

1. replaces each cell's NaN fasttreeshap rows with the repaired ones,
2. recomputes the full pairwise_metrics dict for EVERY row of a repaired cell
   from the stored shapley_values vectors — so the other backends' rows gain
   real fasttreeshap entries and the fasttreeshap rows gain entries vs all
   backends, exactly as if everything had run in one task,
3. validates per cell that both runs trained the identical model, by requiring
   the oracle rows' (shap_true_value / shap_interaction) stored values to
   match; mismatching cells are skipped with a warning,
4. self-checks the offline metric recomputation by comparing a recomputed
   already-existing pairwise entry against its stored run-time value.

Usage:
    python scripts/merge_fasttreeshap_repair.py \
        --tree-csv Benchmarking/results_config-tree.csv \
        --repair-csv Benchmarking/results_config-tree-fasttreeshap.csv \
        --output-csv Benchmarking/results_config-tree-merged.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Benchmarking.metrics import (  # noqa: E402
    mean_abs_diff,
    relative_mae,
    sign_agreement,
    mean_sample_rho,
)

# One benchmark cell = one runner.run() invocation. Everything that
# distinguishes cells in both CSVs; hyperparams NaN-filled so lightgbm
# (no min_samples_split) and random_forest (no learning_rate) key cleanly.
CELL_KEY = ["dataset", "model", "order", "n_background", "n_features",
            "n_samples", "learning_rate", "max_depth", "n_estimators", "seed"]

ORACLE_BY_ORDER = {1: "shap_true_value", 2: "shap_interaction"}


def contrib_frame(row: pd.Series) -> pd.DataFrame:
    """Rebuild the (n_eval, n_features) contribution DataFrame the runner fed
    to the metric functions. Column labels are positional on purpose: metrics
    subtract DataFrames, which aligns by column name, and the two runs may
    carry different feature-name orders only if the model differed — which the
    oracle check already rules out."""
    values = np.array(json.loads(row["shapley_values"]), dtype=float)
    n, d = int(row["shapley_n_eval"]), int(row["shapley_n_features"])
    return pd.DataFrame(values.reshape(n, d))


def pairwise_dict(candidate: pd.DataFrame, others: dict[str, pd.DataFrame],
                  self_name: str) -> str:
    """Recompute runner._row's pairwise_metrics JSON for one candidate."""
    out = {}
    for name, ref in others.items():
        if name == self_name:
            out[name] = {
                "mean_abs_diff": 0.0,
                "relative_mae": 0.0,
                "sign_agreement": float(sign_agreement(candidate, candidate)),
                "mean_sample_rho": 1.0,
            }
        else:
            out[name] = {
                "mean_abs_diff": mean_abs_diff(candidate, ref),
                "relative_mae": relative_mae(candidate, ref),
                "sign_agreement": sign_agreement(candidate, ref),
                "mean_sample_rho": mean_sample_rho(candidate, ref),
            }
    return json.dumps(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree-csv", default="Benchmarking/results_config-tree.csv")
    parser.add_argument("--repair-csv", default="Benchmarking/results_config-tree-fasttreeshap.csv")
    parser.add_argument("--output-csv", default="Benchmarking/results_config-tree-merged.csv")
    parser.add_argument("--oracle-rtol", type=float, default=1e-5,
                        help="Relative tolerance for the identical-model oracle check")
    args = parser.parse_args()

    # round_trip parser: the default fast parser is off by 1 ULP on some
    # floats, which would introduce spurious last-digit changes in rows this
    # script never touches.
    tree = pd.read_csv(args.tree_csv, float_precision="round_trip")
    repair = pd.read_csv(args.repair_csv, float_precision="round_trip")
    if list(repair.columns) != list(tree.columns):
        sys.exit("error: column mismatch between the two CSVs — did the runner change?")

    key_of = lambda df: df[CELL_KEY].astype(float, errors="ignore").apply(
        lambda r: tuple(-1.0 if pd.isna(v) else v for v in r), axis=1)
    tree["_cell"] = key_of(tree)
    repair["_cell"] = key_of(repair)

    stats = {"repaired": 0, "skipped_missing": 0, "skipped_oracle": 0, "rows_replaced": 0}
    selfcheck_diffs = []

    for cell, rep_rows in repair.groupby("_cell"):
        orig_idx = tree.index[tree["_cell"] == cell]
        if len(orig_idx) == 0:
            print(f"  [WARN] repair cell not in tree CSV, skipped: {dict(zip(CELL_KEY, cell))}")
            stats["skipped_missing"] += 1
            continue
        cell_rows = tree.loc[orig_idx]
        order = int(cell_rows["order"].iloc[0])
        oracle = ORACLE_BY_ORDER[order]

        # --- identical-model check: oracle values must match across runs ---
        v_tree = contrib_frame(cell_rows[cell_rows.backend == oracle].iloc[0]).values
        v_rep = contrib_frame(rep_rows[rep_rows.backend == oracle].iloc[0]).values
        if not np.allclose(v_tree, v_rep, rtol=args.oracle_rtol, atol=1e-9, equal_nan=True):
            max_diff = float(np.nanmax(np.abs(v_tree - v_rep)))
            print(f"  [WARN] oracle mismatch (max diff {max_diff:.3g}) — different trained "
                  f"model, cell skipped: {dict(zip(CELL_KEY, cell))}")
            stats["skipped_oracle"] += 1
            continue

        # --- replace fasttreeshap rows in place (row order preserved) ---
        ft_idx = cell_rows.index[cell_rows.library == "fasttreeshap"]
        ft_new = rep_rows[rep_rows.library == "fasttreeshap"]
        if len(ft_idx) != len(ft_new):
            print(f"  [WARN] fasttreeshap row-count mismatch ({len(ft_idx)} vs {len(ft_new)}), "
                  f"cell skipped: {dict(zip(CELL_KEY, cell))}")
            stats["skipped_missing"] += 1
            continue
        replace_cols = [c for c in tree.columns if c != "_cell"]
        tree.loc[ft_idx, replace_cols] = ft_new[replace_cols].values
        stats["rows_replaced"] += len(ft_idx)

        # --- recompute the full pairwise dict for every row of the cell ---
        cell_rows = tree.loc[orig_idx]  # re-read: fasttreeshap rows are now real
        contribs = {r["backend"]: contrib_frame(r) for _, r in cell_rows.iterrows()}

        # self-check: a pre-existing non-fasttreeshap entry recomputed offline
        # must match its stored run-time value (validates metric parity).
        probe = cell_rows[(cell_rows.library != "fasttreeshap") & (cell_rows.backend != oracle)]
        if len(probe) > 0:
            row = probe.iloc[0]
            stored = json.loads(row["pairwise_metrics"])[oracle]["mean_abs_diff"]
            recomputed = mean_abs_diff(contribs[row.backend], contribs[oracle])
            if np.isfinite(stored) and np.isfinite(recomputed):
                selfcheck_diffs.append(abs(stored - recomputed))

        for idx in orig_idx:
            backend = tree.at[idx, "backend"]
            tree.at[idx, "pairwise_metrics"] = pairwise_dict(
                contribs[backend], contribs, backend)
        stats["repaired"] += 1

    tree.drop(columns="_cell").to_csv(args.output_csv, index=False)
    print(f"\ncells repaired:   {stats['repaired']}")
    print(f"rows replaced:    {stats['rows_replaced']}")
    print(f"cells skipped:    {stats['skipped_missing']} (missing/count mismatch), "
          f"{stats['skipped_oracle']} (oracle mismatch)")
    if selfcheck_diffs:
        print(f"metric self-check: max |stored - recomputed| = {max(selfcheck_diffs):.3g} "
              f"over {len(selfcheck_diffs)} probes (should be ~0)")
    print(f"-> {args.output_csv}")


if __name__ == "__main__":
    main()
