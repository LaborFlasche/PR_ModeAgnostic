#!/bin/bash
# Run this from the repo root:  bash slurm/submit.sh <config_path>
# The config path is required, e.g. configs/RQ1-accuracy/config-accuracy.yaml.
# Run it again with a different config (e.g. configs/RQ4-tree/config-tree.yaml)
# for another sweep — each config gets its own output directory and merged CSV
# so the runs don't collide and can be submitted in parallel.
# It submits the array job and then a merge job that waits for it.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source slurm/select_array_script.sh

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "Usage: bash slurm/submit.sh <config_path>   (e.g. config-accuracy.yaml or configs/RQ1-accuracy/config-accuracy.yaml)" >&2
    exit 1
fi
# Allow passing just the filename: search configs/ (and its RQ*/ subfolders)
# if not found as given.
if [ ! -f "$CONFIG" ]; then
    FOUND="$(find configs -maxdepth 2 -type f -name "$(basename "$CONFIG")" | head -1)"
    [ -n "$FOUND" ] && CONFIG="$FOUND"
fi
CONFIG_NAME="$(basename "$CONFIG" .yaml)"
OUTPUT_DIR="benchmarking/slurm_results/$CONFIG_NAME"
MERGED_CSV="benchmarking/results_$CONFIG_NAME.csv"

ARRAY_SCRIPT="$(select_array_script "$CONFIG")"

# Count (dataset × model) combinations from the config
N=$(~/.local/bin/uv run python slurm/count_tasks.py "$CONFIG")
echo "Submitting $N array tasks for config=$CONFIG (via $ARRAY_SCRIPT)..."

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
