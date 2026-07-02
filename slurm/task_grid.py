"""Shared task-grid construction for the SLURM workers and submitters.

The submitters (submit.sh via count_tasks.py, submit_all.py) must produce
exactly as many task-ids as the worker scripts index into, or some cells
silently never run. Both sides therefore build the grid through this module:

    build_all_runs     -> run_benchmark.py     (sweeps seed x n_background)
    build_all_runs_nn  -> run_benchmark_nn.py  (sweeps n_background only; the
                          NN worker reads a single seed value from the config)

Import requires the repo root on sys.path (workers run from the repo root).
"""
import yaml
from sklearn.model_selection import ParameterGrid

from Models.config_parser import load_config, load_dataset_config, as_list


def _grid_axes(config_path: str) -> tuple[dict, list[tuple], list[tuple]]:
    """The three grid axes every worker shares: benchmark dict, (dataset, params)
    cells and (model, params) cells."""
    with open(config_path) as f:
        bench = yaml.safe_load(f)["benchmark"]
    dataset_runs = [
        (k, p) for k, pg in load_dataset_config(config_path).items() for p in ParameterGrid(pg)
    ]
    model_runs = [
        (k, p) for k, pg in load_config(config_path).items() for p in ParameterGrid(pg)
    ]
    return bench, dataset_runs, model_runs


def build_all_runs(config_path: str) -> list[tuple]:
    """Every independent benchmark cell for a config, as (seed, dataset,
    dataset_params, model, model_params, n_background) tuples — one per SLURM
    array task. ``seed`` and ``n_background`` may each be a scalar or a list and
    are swept as extra grid dimensions (see as_list)."""
    bench, dataset_runs, model_runs = _grid_axes(config_path)
    return [
        (seed, dk, dp, mk, mp, n_bg)
        for seed in as_list(bench["seed"])
        for dk, dp in dataset_runs
        for mk, mp in model_runs
        for n_bg in as_list(bench["n_background"])
    ]


def build_all_runs_nn(config_path: str) -> list[tuple]:
    """NN counterpart of build_all_runs: (dataset, dataset_params, model,
    model_params, n_background) tuples without the seed dimension —
    run_benchmark_nn.py reads a single seed value directly from the config."""
    bench, dataset_runs, model_runs = _grid_axes(config_path)
    return [
        (dk, dp, mk, mp, n_bg)
        for dk, dp in dataset_runs
        for mk, mp in model_runs
        for n_bg in as_list(bench["n_background"])
    ]
