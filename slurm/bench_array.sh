#!/bin/bash
#SBATCH --job-name=shap_bench
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_%A_%a.out
#SBATCH --error=slurm/logs/bench_%A_%a.err
# --array is set dynamically by submit.sh — do not set it here
# $1 = config path, $2 = output dir — both passed through by submit.sh so each
# config (model-agnostic vs tree) writes to its own results directory.

# Run from repo root (submit.sh does: sbatch --chdir=<repo_root> ... slurm/bench_array.sh <config>)
CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "bench_array.sh: missing config argument" >&2
    exit 1
fi
OUTPUT_DIR="${2:-benchmarking/slurm_results}"
~/.local/bin/uv run python slurm/run_benchmark.py --task-id "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
