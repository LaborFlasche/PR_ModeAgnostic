#!/usr/bin/env python3
"""Pre-download every dataset referenced by the benchmark configs into the cache.

Compute nodes may have no outbound internet, so all datasets must be cached
before submitting (SLURM.md step 3). Run this once on a node *with* internet;
the scikit-learn cache lives under a shared NFS home, so a single run populates
it for every compute node.

Caching is per raw dataset, not per (n_features, n_samples, seed): the OpenML /
sklearn fetch that populates the cache downloads the *full* dataset and happens
before any subsampling, so one load per key with a fixed seed is enough — every
seed's subsample is then derivable offline from that one cached copy. Re-running
is cheap: an already-cached dataset is read from disk (no network), it just
re-runs the in-memory preprocessing.

Usage (from the repo root):
    uv run python scripts/cache_datasets.py                 # all configs/*/*.yaml
    uv run python scripts/cache_datasets.py configs/RQ4-tree/config-tree.yaml ...

Then verify offline-readiness and gate the submit on it:
    uv run python scripts/check_dataset_cache.py && uv run python slurm/submit_all.py
"""
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import yaml


def _dataset_keys(config_paths: list[str]) -> list[str]:
    keys: list[str] = []
    for path in config_paths:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for key in (raw.get("datasets") or {}):
            if key not in keys:
                keys.append(key)
    return keys


def main() -> int:
    # Configs live in per-RQ subfolders (configs/RQ*/...); this glob must match
    # check_dataset_cache.py's exactly so cache and verify cover the same files.
    config_paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(REPO_ROOT, "configs", "*", "*.yaml"))
    )
    if not config_paths:
        print("No configs found.", file=sys.stderr)
        return 1

    keys = _dataset_keys(config_paths)
    print(f"Datasets referenced by {len(config_paths)} config(s): {', '.join(keys)}")
    from sklearn.datasets import get_data_home
    print(f"scikit-learn cache: {get_data_home()}\n")

    from Models.dataset_and_models import Dataset

    for key in keys:
        # One load per key populates the cache; nf/ns/seed don't affect what is
        # fetched, so we pass a fixed seed and no subsetting.
        print(f"loading {key} ...", flush=True)
        Dataset[key.upper()].load_dataset(seed=0)

    print(f"\ndone — {len(keys)} dataset(s) cached")
    print("Verify offline-readiness with: uv run python scripts/check_dataset_cache.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
