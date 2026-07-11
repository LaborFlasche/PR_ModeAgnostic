#!/bin/bash
#SBATCH --job-name=nn_bench
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/bench_nn_%A_%a.out
#SBATCH --error=slurm/logs/bench_nn_%A_%a.err
# NN array worker for device=cuda configs. The CIP cluster defines no GPU GRES
# (sinfo: GRES=(null) everywhere), so --gres/--gpus are rejected — the
# NvidiaAll partition alone provides the node's GPU. CPU NN configs use
# bench_array_nn_cpu.sh instead (see select_array_script.sh).
# --array is set dynamically by submit.sh — do not set it here
# $1 = config path, $2 = output dir — both passed through by submit.sh so each
# config writes to its own results directory.

CONFIG="${1:-configs/RQ3-neural-networks/config-neural-networks-gpu.yaml}"
OUTPUT_DIR="${2:-benchmarking/slurm_results}"
~/.local/bin/uv run python slurm/run_benchmark_nn.py --task-id "$SLURM_ARRAY_TASK_ID" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
