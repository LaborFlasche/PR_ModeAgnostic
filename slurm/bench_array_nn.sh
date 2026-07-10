#!/bin/bash
#SBATCH --job-name=nn_bench
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_nn_%A_%a.out
#SBATCH --error=slurm/logs/bench_nn_%A_%a.err
# --array is set dynamically by submit.sh — do not set it here
# $1 = config path, $2 = output dir — both passed through by submit.sh so each
# config writes to its own results directory.

CONFIG="${1:-configs/RQ3-neural-networks/config-neural-networks-RQ3-gpu.yaml}"
OUTPUT_DIR="${2:-Benchmarking/slurm_results}"
~/.local/bin/uv run python slurm/run_benchmark_nn.py --task-id "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
