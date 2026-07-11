#!/bin/bash
#SBATCH --job-name=nn_bench_cpu
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_nn_%A_%a.out
#SBATCH --error=slurm/logs/bench_nn_%A_%a.err
# CPU counterpart of bench_array_nn.sh, for the device=cpu NN config.
# --array is set dynamically by submit.sh — do not set it here.
# $1 = config path, $2 = output dir — passed through by submit.sh.

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "bench_array_nn_cpu.sh: missing config argument" >&2
    exit 1
fi
OUTPUT_DIR="${2:-Benchmarking/slurm_results}"
~/.local/bin/uv run python slurm/run_benchmark_nn.py --task-id "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
