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

# xgboost/lightgbm must be imported before shapiq anywhere in this process, or a
# later xgboost/lightgbm .fit() segfaults — establishes safe load order before
# `from Benchmarking.backends import (...)` pulls in shapiq.
import xgboost  # noqa: F401
import lightgbm  # noqa: F401

warnings.filterwarnings("ignore", message="Not all budget.*", category=UserWarning, module="shapiq")
warnings.filterwarnings("ignore", message="The sample size is larger.*", category=UserWarning, module="shapiq")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)

import yaml
from sklearn.model_selection import ParameterGrid

from Models.config_parser import load_config, load_dataset_config, as_list
from Models.dataset_and_models import Dataset, Model
from Benchmarking import BenchmarkRunner
from Benchmarking.backends import (
    ShapTrueValueBackend,
    ShapApproxBackend,
    ShapIQApproxBackend,
    LightShapApproxBackend,
    DalexApproxBackend,
    ShapTreePathDependentBackend,
    ShapInteractionBackend,
    ShapIQTreePathDependentBackend,
    ShapIQInteractionBackend,
    WoodelfTreePathDependentBackend,
    WoodelfTreeInterventionalBackend,
    WoodelfInteractionBackend,
    FastTreeShapBackend,
)
# GPU backends (WoodelfGPU*, GPUTreeShap*) exist in Benchmarking.backends for
# future use but aren't wired in here for now.

APPROX_MAP = {
    "shap": ShapApproxBackend,
    "shapiq": ShapIQApproxBackend,
    "lightshap": LightShapApproxBackend,
    "dalex": DalexApproxBackend,
}

# Tree-specific true-value backends, only applied to tree models (Model.is_tree).
# Keyed by (library, mode). No ("shapiq_tree", "interventional") entry: it
# crashes in this dependency stack — see tree_shapiq_backend.py.
TREE_TRUE_VALUE_MAP = {
    ("shap_tree", "path_dependent"): ShapTreePathDependentBackend,
    ("shapiq_tree", "path_dependent"): ShapIQTreePathDependentBackend,
    ("woodelf", "path_dependent"): WoodelfTreePathDependentBackend,
    ("woodelf", "interventional"): WoodelfTreeInterventionalBackend,
    ("fasttreeshap", "path_dependent"): FastTreeShapBackend,
}

# Pairwise-interaction (order-2) backends, path-dependent only. "shap_tree" is
# absent: ShapInteractionBackend is hardcoded as the always-on oracle below.
INTERACTION_TRUE_VALUE_MAP = {
    "shapiq_tree": ShapIQInteractionBackend,
    "woodelf": WoodelfInteractionBackend,
}


def build_all_runs(config_path: str) -> list[tuple]:
    """Every independent benchmark cell for a config, as (seed, dataset,
    dataset_params, model, model_params, n_background) tuples — one per SLURM array
    task. ``seed`` and ``n_background`` may each be a scalar or a list and are swept
    as extra grid dimensions (see as_list)."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True,
                        help="SLURM_ARRAY_TASK_ID — index into all (dataset, model) combinations")
    parser.add_argument("--config", required=True,
                        help="Path to the config file used to run the benchmark")
    parser.add_argument("--output-dir", default="Benchmarking/slurm_results")
    args = parser.parse_args()

    all_runs = build_all_runs(args.config)
    if args.task_id >= len(all_runs):
        print(f"task-id {args.task_id} out of range (max {len(all_runs) - 1})", file=sys.stderr)
        sys.exit(1)

    seed, dk, dp, mk, mp, n_background = all_runs[args.task_id]
    print(f"[task {args.task_id}] dataset={dk} {dp} | model={mk} {mp} "
          f"| seed={seed} | n_background={n_background}")

    with open(args.config) as f:
        bench = yaml.safe_load(f)["benchmark"]

    imputer = bench["imputer"]

    # Support scalar or list for n_background (list = sweep, e.g. config-accuracy).
    n_backgrounds = bench["n_background"]
    if isinstance(n_backgrounds, int):
        n_backgrounds = [n_backgrounds]

    approx_specs = [
        (APPROX_MAP[lib], {"approximator": appr, "budget": bgt})
        for lib in bench["libraries"]
        for appr in bench["approximators"]
        for bgt in bench["budgets"]
        if appr in getattr(APPROX_MAP[lib], "SUPPORTED_APPROXIMATORS", bench["approximators"])
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, f"results_{args.task_id:04d}.csv")
    if os.path.exists(output_csv):
        os.remove(output_csv)

    model_enum = Model[mk.upper()]

    # ShapTrueValueBackend must stay first: it's picked as the oracle.
    true_value_backends = [ShapTrueValueBackend]
    if model_enum.is_tree:
        for lib in bench.get("tree_libraries", []):
            for mode in bench.get("tree_modes", []):
                cls = TREE_TRUE_VALUE_MAP.get((lib, mode))
                if cls is not None:
                    true_value_backends.append(cls)

    dataset_enum = Dataset[dk.upper()]
    ds = dataset_enum.load_dataset(**dp, seed=seed)
    trainer = model_enum.get_model_with_params(dataset_enum, mp, seed=seed)
    trainer.fit(ds["X"], ds["y"], task=ds["task"])

    for nbg in n_backgrounds:
        runner = BenchmarkRunner(
            true_value_backends=true_value_backends,
            approximation_specs=approx_specs,
            output_csv=output_csv,
            n_background=nbg,
            n_eval=bench["n_eval"],
            seed=seed,
            imputer=imputer,
        )
        runner.run(
            model=trainer.get_model(),
            X=ds["X"],
            run_meta={"dataset": dk, "model": mk, "order": 1, "n_background": nbg, **dp},
        )

    # Pairwise interactions: a separate runner.run() call (different oracle,
    # different output shape) writing to the same output_csv.
    interaction_libs = bench.get("interaction_libraries", [])
    if model_enum.is_tree and interaction_libs:
        max_features = bench.get("interaction_max_features", 16)
        if ds["X"].shape[1] > max_features:
            print(f"[task {args.task_id}] skipping interactions: "
                  f"n_features={ds['X'].shape[1]} > interaction_max_features={max_features}")
        else:
            # ShapInteractionBackend must stay first: it's the order-2 oracle.
            interaction_backends = [ShapInteractionBackend]
            for lib in interaction_libs:
                cls = INTERACTION_TRUE_VALUE_MAP.get(lib)
                if cls is not None:
                    interaction_backends.append(cls)

            for nbg in n_backgrounds:
                interaction_runner = BenchmarkRunner(
                    true_value_backends=interaction_backends,
                    approximation_specs=[],
                    output_csv=output_csv,
                    n_background=nbg,
                    n_eval=bench["n_eval"],
                    seed=seed,
                    imputer=imputer,
                )
                interaction_runner.run(
                    model=trainer.get_model(),
                    X=ds["X"],
                    run_meta={"dataset": dk, "model": mk, "order": 2, "n_background": nbg, **dp},
                )

    print(f"[task {args.task_id}] done -> {output_csv}")


if __name__ == "__main__":
    main()
