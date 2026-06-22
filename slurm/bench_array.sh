#!/bin/bash
#SBATCH --job-name=shap_bench
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_%A_%a.out
#SBATCH --error=slurm/logs/bench_%A_%a.err
# --array is set dynamically by submit.sh — do not set it here

# Run from repo root (submit.sh does: sbatch --chdir=<repo_root> ...)
~/.local/bin/uv run python slurm/run_benchmark.py --task-id "$SLURM_ARRAY_TASK_ID"
