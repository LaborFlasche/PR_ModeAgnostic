#!/bin/bash
#SBATCH --job-name=shap_bench_gpu
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_%A_%a.out
#SBATCH --error=slurm/logs/bench_%A_%a.err
# GPU counterpart of bench_array.sh — used for configs/RQ5-gpu/config-tree-gpu.yaml
# (and any other config whose name contains "gpu", see submit.sh). The CIP
# cluster defines no GPU GRES (sinfo: GRES=(null) everywhere), so --gres/--gpus
# are rejected — the NvidiaAll partition alone provides the node's GPU.
# Everything else mirrors bench_array.sh.
# --array is set dynamically by submit.sh — do not set it here
# $1 = config path, $2 = output dir — both passed through by submit.sh.

# Run from repo root (submit.sh does: sbatch --chdir=<repo_root> ...)
CONFIG="${1:-configs/RQ5-gpu/config-tree-gpu.yaml}"
OUTPUT_DIR="${2:-Benchmarking/slurm_results}"
~/.local/bin/uv run python slurm/run_benchmark.py --task-id "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
