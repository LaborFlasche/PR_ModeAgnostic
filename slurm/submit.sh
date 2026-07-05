#!/bin/bash
# Run this from the repo root:  bash slurm/submit.sh <config_path>
# The config path is required, e.g. configs/config-accuracy.yaml. Run it again
# with a different config (e.g. configs/config-tree.yaml) for another sweep —
# each config gets its own output directory and merged CSV so the runs don't
# collide and can be submitted in parallel.
# Run this from the repo root:  bash slurm/submit.sh [config_path]
# Defaults to configs/config.yaml (model-agnostic). Run again with
# configs/config-tree.yaml for the tree-specific sweep, or
# configs/config-tree-gpu.yaml for the woodelf cpu-vs-gpu sweep (runs on a GPU
# node, see slurm/bench_array_gpu.sh) — each config gets its own output
# directory and merged CSV so the runs don't collide.
# It submits the array job and then a merge job that waits for it.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "Usage: bash slurm/submit.sh <config_path>   (e.g. config-accuracy.yaml or configs/config-accuracy.yaml)" >&2
    exit 1
fi
# Allow passing just the filename: fall back to configs/ if not found as given.
[ -f "$CONFIG" ] || CONFIG="configs/$CONFIG"
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
OUTPUT_DIR="Benchmarking/slurm_results/$CONFIG_NAME"
MERGED_CSV="Benchmarking/results_$CONFIG_NAME.csv"

# Pick the array script once: NN configs use run_benchmark_nn.py, GPU configs
# (name contains "gpu", e.g. configs/config-tree-gpu.yaml) need a GPU node
# (--gres=gpu:1, NvidiaAll partition — see slurm/bench_array_gpu.sh), everything
# else runs the CPU array script. Must stay a single if/elif chain: two separate
# assignments previously let the CPU branch silently overwrite the GPU choice.
if [[ "$CONFIG" == *neural-networks* ]]; then
    ARRAY_SCRIPT="slurm/bench_array_nn.sh"
elif [[ "$CONFIG_NAME" == *gpu* ]]; then
    ARRAY_SCRIPT="slurm/bench_array_gpu.sh"
else
    ARRAY_SCRIPT="slurm/bench_array.sh"
fi

# Count (dataset × model) combinations from the config
N=$(~/.local/bin/uv run python slurm/count_tasks.py "$CONFIG")
echo "Submitting $N array tasks for config=$CONFIG..."

mkdir -p slurm/logs "$OUTPUT_DIR"

# Start each sweep from a clean per-task output dir. These files are transient — they
# are merged into $MERGED_CSV at the end. Clearing them prevents stale files from a
# previous sweep (possibly a different task count or column schema) leaking into the merge.
rm -f "$OUTPUT_DIR"/results_*.csv

ARRAY_JOB=$(sbatch \
    --array=0-$((N - 1)) \
    --chdir="$REPO_ROOT" \
    "$ARRAY_SCRIPT" "$CONFIG" "$OUTPUT_DIR" \
    | awk '{print $4}')
echo "Array job ID: $ARRAY_JOB"

MERGE_JOB=$(sbatch \
    --dependency=afterok:"$ARRAY_JOB" \
    --chdir="$REPO_ROOT" \
    slurm/merge.sh --input-dir "$OUTPUT_DIR" --output-csv "$MERGED_CSV" \
    | awk '{print $4}')
echo "Merge job ID: $MERGE_JOB (runs after all array tasks succeed)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       slurm/logs/"
echo "Results in:    $OUTPUT_DIR/  (merged → $MERGED_CSV)"
