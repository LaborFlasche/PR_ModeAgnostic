#!/bin/bash
#SBATCH --job-name=nn_bench_cpu
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_nn_cpu_%A_%a.out
#SBATCH --error=slurm/logs/bench_nn_cpu_%A_%a.err
# CPU counterpart of bench_array_nn.sh for NN configs with device: cpu (name
# contains "neural-networks" and "cpu", see slurm/select_array_script.sh) —
# runs on the CPU partition without allocating a GPU it would never use.
# --array is set dynamically by submit.sh — do not set it here
# $1 = config path, $2 = output dir — both passed through by submit.sh.

CONFIG="${1:-configs/config-neural-networks-RQ3-cpu.yaml}"
OUTPUT_DIR="${2:-Benchmarking/slurm_results}"
~/.local/bin/uv run python slurm/run_benchmark_nn.py --task-id "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
