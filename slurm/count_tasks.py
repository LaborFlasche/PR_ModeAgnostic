#!/usr/bin/env python3
"""Prints the number of (dataset, model) task combinations for the given config.

Usage: python slurm/count_tasks.py [config_path]"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from sklearn.model_selection import ParameterGrid
from Models.config_parser import load_config, load_dataset_config, as_list

if len(sys.argv) != 2:
    print("Usage: python slurm/count_tasks.py <config.yaml>", file=sys.stderr)
    sys.exit(1)

CONFIG = sys.argv[1]
model_runs = [p for pg in load_config(CONFIG).values() for p in ParameterGrid(pg)]
dataset_runs = [p for pg in load_dataset_config(CONFIG).values() for p in ParameterGrid(pg)]
# seed and n_background are swept as extra grid dimensions (scalar or list); must
# match the product built by build_all_runs in slurm/run_benchmark.py.
with open(CONFIG) as f:
    bench = yaml.safe_load(f)["benchmark"]
n_seeds = len(as_list(bench["seed"]))
n_backgrounds = len(as_list(bench["n_background"]))
print(len(model_runs) * len(dataset_runs) * n_seeds * n_backgrounds)
