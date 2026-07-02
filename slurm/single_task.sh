#!/bin/bash
# Generic single-task SLURM wrapper used by submit_all.py.
# Args (positional, after sbatch flags): worker_script  task_id  config  output_dir
#
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/task_%j.out
#SBATCH --error=slurm/logs/task_%j.err
# Note: NN jobs (run_benchmark_nn.py) use device=cuda from the config —
# submit_all.py overrides the partition/gres above with --partition=NvidiaAll
# --gres=gpu:1 for entries whose worker_script is slurm/run_benchmark_nn.py.

~/.local/bin/uv run python "$1" --task-id "$2" --config "$3" --output-dir "$4"
