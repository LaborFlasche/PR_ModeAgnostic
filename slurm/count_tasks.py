#!/usr/bin/env python3
"""Prints the number of benchmark cells (SLURM array tasks) for a config.

Usage: python slurm/count_tasks.py <config.yaml>"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_grid import build_all_runs  # noqa: E402

if len(sys.argv) != 2:
    print("Usage: python slurm/count_tasks.py <config.yaml>", file=sys.stderr)
    sys.exit(1)

print(len(build_all_runs(sys.argv[1])))
