#!/bin/bash
#SBATCH --job-name=captum_nn_bench
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/bench_nn_%A_%a.out
#SBATCH --error=slurm/logs/bench_nn_%A_%a.err
# --array is set dynamically by submit.sh — do not set it here

# Run from repo root (submit.sh does: sbatch --chdir=<repo_root> ...)
~/.local/bin/uv run python slurm/run_benchmark_nn.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --config configs/config-nn.yaml \
    --output-dir Benchmarking/results_nn
