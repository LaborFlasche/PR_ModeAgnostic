#!/usr/bin/env python3
"""Prints the number of SLURM array tasks for the given config.

Usage: python slurm/count_tasks.py [config_path]"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_grid import build_all_runs, build_all_runs_nn

if len(sys.argv) != 2:
    print("Usage: python slurm/count_tasks.py <config.yaml>", file=sys.stderr)
    sys.exit(1)

CONFIG = sys.argv[1]
# Matches the *neural-networks* dispatch in submit.sh / submit_all.py: NN
# configs are indexed by run_benchmark_nn.py via build_all_runs_nn, everything
# else by run_benchmark.py via build_all_runs.
build_fn = build_all_runs_nn if "neural-networks" in CONFIG else build_all_runs
print(len(build_fn(CONFIG)))
