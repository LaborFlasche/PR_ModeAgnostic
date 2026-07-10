#!/bin/bash
#SBATCH --job-name=shap_bench_gpu
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_%A_%a.out
#SBATCH --error=slurm/logs/bench_%A_%a.err
# GPU counterpart of bench_array.sh — for non-NN configs whose name contains
# "gpu" (woodelf's cupy path, see select_array_script.sh). The CIP cluster
# defines no GPU GRES (sinfo: GRES=(null) everywhere), so --gres/--gpus are
# rejected — the NvidiaAll partition alone provides the node's GPU.
# --array is set dynamically by submit.sh — do not set it here.
# $1 = config path, $2 = output dir — passed through by submit.sh.

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "bench_array_gpu.sh: missing config argument" >&2
    exit 1
fi
OUTPUT_DIR="${2:-Benchmarking/slurm_results}"
~/.local/bin/uv run python slurm/run_benchmark.py --task-id "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
