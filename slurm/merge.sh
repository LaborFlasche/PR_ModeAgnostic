#!/bin/bash
#SBATCH --job-name=shap_merge
#SBATCH --partition=Krater
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm/logs/merge_%j.out
#SBATCH --error=slurm/logs/merge_%j.err
# Any args ($1 $2 ... e.g. --input-dir DIR --output-csv FILE) are forwarded by
# submit.sh so each config's run merges into its own output file.

~/.local/bin/uv run python slurm/merge_results.py "$@"
