#!/bin/bash
# Run this from the repo root:  bash slurm/submit.sh
# It submits the array job and then a merge job that waits for it.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Count (dataset × model) combinations from the config
N=$(~/.local/bin/uv run python slurm/count_tasks.py)
echo "Submitting $N array tasks..."

mkdir -p slurm/logs

# Start each sweep from a clean per-task output dir. These files are transient — they
# are merged into Benchmarking/results.csv at the end. Clearing them prevents stale files
# from a previous sweep (possibly a different task count or column schema) leaking into the merge.
mkdir -p Benchmarking/slurm_results
rm -f Benchmarking/slurm_results/results_*.csv

ARRAY_JOB=$(sbatch \
    --array=0-$((N - 1)) \
    --chdir="$REPO_ROOT" \
    slurm/bench_array.sh \
    | awk '{print $4}')
echo "Array job ID: $ARRAY_JOB"

MERGE_JOB=$(sbatch \
    --dependency=afterok:"$ARRAY_JOB" \
    --chdir="$REPO_ROOT" \
    slurm/merge.sh \
    | awk '{print $4}')
echo "Merge job ID: $MERGE_JOB (runs after all array tasks succeed)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       slurm/logs/"
echo "Results in:    Benchmarking/slurm_results/  (merged → Benchmarking/results.csv)"
