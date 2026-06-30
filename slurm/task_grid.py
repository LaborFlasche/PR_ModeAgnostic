"""Canonical task grids for SLURM array indexing.

Single source of truth for "how many tasks does this config have, and what
does task-id N mean" — used by run_benchmark.py / run_benchmark_nn.py (to
resolve --task-id) and by submit_all.py / count_tasks.py (to know how many
tasks to submit). Keeping one implementation means the submitted task count
can never drift from what --task-id actually indexes into.
"""
from sklearn.model_selection import ParameterGrid
import yaml

from Models.config_parser import load_config, load_dataset_config, as_list


def build_all_runs(config_path: str) -> list[tuple]:
    """Every independent benchmark cell for a config, as (seed, dataset,
    dataset_params, model, model_params, n_background) tuples — one per SLURM
    array task. ``seed`` and ``n_background`` may each be a scalar or a list
    and are swept as extra grid dimensions (see as_list)."""
    model_config = load_config(config_path)
    dataset_config = load_dataset_config(config_path)
    with open(config_path) as f:
        bench = yaml.safe_load(f)["benchmark"]
    seeds = as_list(bench["seed"])
    model_runs = [(k, p) for k, pg in model_config.items() for p in ParameterGrid(pg)]
    dataset_runs = [(k, p) for k, pg in dataset_config.items() for p in ParameterGrid(pg)]
    n_backgrounds = as_list(bench["n_background"])
    return [
        (seed, dk, dp, mk, mp, n_bg)
        for seed in seeds
        for dk, dp in dataset_runs
        for mk, mp in model_runs
        for n_bg in n_backgrounds
    ]


def build_all_runs_nn(config_path: str) -> list[tuple]:
    """Every independent NN benchmark cell, as (dataset, dataset_params, model,
    model_params, n_background) tuples. No seed dimension: run_benchmark_nn.py
    reads a single seed value straight from the config rather than sweeping it."""
    model_config = load_config(config_path)
    dataset_config = load_dataset_config(config_path)
    with open(config_path) as f:
        bench = yaml.safe_load(f)["benchmark"]
    model_runs = [(k, p) for k, pg in model_config.items() for p in ParameterGrid(pg)]
    dataset_runs = [(k, p) for k, pg in dataset_config.items() for p in ParameterGrid(pg)]
    n_backgrounds = as_list(bench["n_background"])
    return [
        (dk, dp, mk, mp, n_bg)
        for dk, dp in dataset_runs
        for mk, mp in model_runs
        for n_bg in n_backgrounds
    ]
