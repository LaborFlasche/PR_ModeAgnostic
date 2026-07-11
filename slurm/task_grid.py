"""The benchmark task grid, shared by workers and submitters.

One benchmark cell = (seed, dataset, dataset_params, model, model_params,
n_background). Submitters (submit_all.py, submit.sh via count_tasks.py) must
submit exactly len(build_all_runs(config)) array tasks, and the workers
(run_benchmark.py, run_benchmark_nn.py) index into the same list via
--task-id — any drift between the two silently drops or fails cells.
"""
import yaml
from sklearn.model_selection import ParameterGrid

from Models.config_parser import load_config, load_dataset_config, as_list


def load_bench(config_path: str) -> dict:
    """The `benchmark:` section of a config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)["benchmark"]


def build_all_runs(config_path: str) -> list[tuple]:
    """Every independent benchmark cell for a config, one per SLURM array task.
    ``seed`` and ``n_background`` may each be a scalar or a list and are swept
    as extra grid dimensions (see as_list)."""
    bench = load_bench(config_path)
    model_runs = [(k, p) for k, pg in load_config(config_path).items()
                  for p in ParameterGrid(pg)]
    dataset_runs = [(k, p) for k, pg in load_dataset_config(config_path).items()
                    for p in ParameterGrid(pg)]
    return [
        (seed, dataset, dataset_params, model, model_params, n_background)
        for seed in as_list(bench["seed"])
        for dataset, dataset_params in dataset_runs
        for model, model_params in model_runs
        for n_background in as_list(bench["n_background"])
    ]
