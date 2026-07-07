#!/bin/bash
# Generic single-task SLURM wrapper used by submit_all.py.
# Args (positional, after sbatch flags): worker_script  task_id  config  output_dir
#
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/task_%j.out
#SBATCH --error=slurm/logs/task_%j.err
# Note: NN jobs (run_benchmark_nn.py) use device=cuda from the config and are
# routed to --partition=NvidiaAll by submit_all.py. Do NOT add --gres/--gpus:
# the CIP cluster defines no GPU GRES (sinfo: GRES=(null) everywhere) and
# rejects those flags; the NvidiaAll nodes expose their GPU directly.

~/.local/bin/uv run python "$1" --task-id "$2" --config "$3" --output-dir "$4"
