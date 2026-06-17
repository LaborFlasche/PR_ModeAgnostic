#!/usr/bin/env python3
"""
SLURM array worker — runs exactly one (dataset, model) benchmark cell.

Usage:
    python slurm/run_benchmark.py --task-id $SLURM_ARRAY_TASK_ID

Run from the repo root so that Models/, Benchmarking/, configs/ are importable.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Not all budget.*", category=UserWarning, module="shapiq")
warnings.filterwarnings("ignore", message="The sample size is larger.*", category=UserWarning, module="shapiq")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)

import yaml
from sklearn.model_selection import ParameterGrid

from Models.config_parser import load_config, load_dataset_config
from Models.dataset_and_models import Dataset, Model
from Benchmarking import BenchmarkRunner
from Benchmarking.backends import (
    ShapTrueValueBackend,
    ShapApproxBackend,
    ShapIQApproxBackend,
    LightShapApproxBackend,
    DalexApproxBackend,
)

APPROX_MAP = {
    "shap": ShapApproxBackend,
    "shapiq": ShapIQApproxBackend,
    "lightshap": LightShapApproxBackend,
    "dalex": DalexApproxBackend,
}


def build_all_runs(config_path: str) -> list[tuple]:
    model_config = load_config(config_path)
    dataset_config = load_dataset_config(config_path)
    model_runs = [(k, p) for k, pg in model_config.items() for p in ParameterGrid(pg)]
    dataset_runs = [(k, p) for k, pg in dataset_config.items() for p in ParameterGrid(pg)]
    return [
        (dk, dp, mk, mp)
        for dk, dp in dataset_runs
        for mk, mp in model_runs
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True,
                        help="SLURM_ARRAY_TASK_ID — index into all (dataset, model) combinations")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default="Benchmarking/slurm_results")
    args = parser.parse_args()

    all_runs = build_all_runs(args.config)
    if args.task_id >= len(all_runs):
        print(f"task-id {args.task_id} out of range (max {len(all_runs) - 1})", file=sys.stderr)
        sys.exit(1)

    dk, dp, mk, mp = all_runs[args.task_id]
    print(f"[task {args.task_id}] dataset={dk} {dp} | model={mk} {mp}")

    with open(args.config) as f:
        bench = yaml.safe_load(f)["benchmark"]

    # Single source of truth for randomness: the same seed drives data subsampling,
    # model training, and every stochastic approximator, so the whole cell is
    # reproducible end to end. Required in the config (no hardcoded fallback).
    seed = bench["seed"]
    # Shared value function every library explains (no hardcoded fallback).
    imputer = bench["imputer"]

    approx_specs = [
        (APPROX_MAP[lib], {"approximator": appr, "budget": bgt})
        for lib in bench["libraries"]
        for appr in bench["approximators"]
        for bgt in bench["budgets"]
        if appr in getattr(APPROX_MAP[lib], "SUPPORTED_APPROXIMATORS", bench["approximators"])
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, f"results_{args.task_id:04d}.csv")
    # Each SLURM task owns exactly one file. The runner appends (so a header is only
    # written when the file is absent); a stale file from a previous run — possibly with
    # a different column schema — would get new rows appended under its old header and
    # corrupt the file. Remove it so every task always writes a fresh, self-consistent CSV.
    if os.path.exists(output_csv):
        os.remove(output_csv)

    runner = BenchmarkRunner(
        true_value_backends=[ShapTrueValueBackend],
        approximation_specs=approx_specs,
        output_csv=output_csv,
        n_background=bench["n_background"],
        n_eval=bench["n_eval"],
        seed=seed,
        imputer=imputer,
    )

    dataset_enum = Dataset[dk.upper()]
    ds = dataset_enum.load_dataset(**dp, seed=seed)
    trainer = Model[mk.upper()].get_model_with_params(dataset_enum, mp, seed=seed)
    trainer.fit(ds["X"], ds["y"], task=ds["task"])

    runner.run(
        model=trainer.get_model(),
        X=ds["X"],
        run_meta={"dataset": dk, "model": mk, **dp},
    )
    print(f"[task {args.task_id}] done -> {output_csv}")


if __name__ == "__main__":
    main()
