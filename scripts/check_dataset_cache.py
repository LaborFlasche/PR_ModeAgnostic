#!/usr/bin/env python3
"""Verify that every dataset used by the benchmark configs is fully cached.

Compute nodes may have no outbound internet, so all datasets must be cached
(SLURM.md step 3) before submitting. This script checks that *without ever
downloading anything*: it disables network access for this process, then runs
the same ``Dataset.load_dataset()`` code path the SLURM workers use. A loader
that succeeds proves the cache is complete and readable (a corrupt/partial
cache fails too); one that hits the network block is reported as NOT CACHED.

Usage (from the repo root):
    uv run python scripts/check_dataset_cache.py                 # all configs/*/*.yaml
    uv run python scripts/check_dataset_cache.py configs/RQ4-tree/config-tree.yaml ...

Exits 0 if everything is cached, 1 otherwise — safe to gate a submit on:
    uv run python scripts/check_dataset_cache.py && uv run python slurm/submit_all.py
"""
import glob
import os
import socket
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import yaml


class _NetworkBlocked(Exception):
    pass


def _block_network() -> None:
    """Make any DNS/socket use raise, so cache misses fail instead of downloading."""

    def _blocked(*_args, **_kwargs):
        raise _NetworkBlocked("network access blocked by check_dataset_cache.py")

    socket.getaddrinfo = _blocked
    socket.create_connection = _blocked


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
    config_paths = sys.argv[1:] or sorted(glob.glob(os.path.join(REPO_ROOT, "configs", "*", "*.yaml")))
    if not config_paths:
        print("No configs found.", file=sys.stderr)
        return 1

    keys = _dataset_keys(config_paths)
    print(f"Datasets referenced by {len(config_paths)} config(s): {', '.join(keys)}")
    from sklearn.datasets import get_data_home
    print(f"scikit-learn cache: {get_data_home()}\n")

    _block_network()
    # Import after the network block so any import-time fetch would be caught too.
    from Models.dataset_and_models import Dataset

    missing: list[str] = []
    for key in keys:
        # Full load through the same enum the SLURM workers use; caching is per
        # dataset, so n_features/n_samples subsetting doesn't matter here.
        try:
            ds = Dataset[key.upper()].load_dataset(seed=0)
            print(f"  OK          {key:20s} ({ds['X'].shape[0]} rows x {ds['X'].shape[1]} features)")
        except _NetworkBlocked:
            print(f"  NOT CACHED  {key:20s} (loader tried to download)")
            missing.append(key)
        except Exception as e:
            # Cache present but unreadable (corrupt/partial), or a loader bug.
            print(f"  ERROR       {key:20s} ({e.__class__.__name__}: {e})")
            missing.append(key)

    if missing:
        print(f"\n{len(missing)} dataset(s) not usable offline: {', '.join(missing)}")
        print("Pre-download them on a node with internet access (see SLURM.md step 3).")
        return 1
    print("\nAll datasets cached — safe to submit on nodes without internet access.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
