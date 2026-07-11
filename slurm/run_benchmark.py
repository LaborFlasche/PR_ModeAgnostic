#!/usr/bin/env python3
"""
SLURM array worker — runs exactly one (dataset, model) benchmark cell.

Usage:
    python slurm/run_benchmark.py --task-id $SLURM_ARRAY_TASK_ID

Run from the repo root so that models/, benchmarking/, configs/ are importable.
"""
import argparse
import os
import sys
import warnings

# xgboost/lightgbm must be imported before shapiq anywhere in this process, or a
# later xgboost/lightgbm .fit() segfaults — establishes safe load order before
# `from backends import (...)` pulls in shapiq. Keep these imports
# at the very top; do not let an import sorter move them below the benchmarking
# imports.
import xgboost  # noqa: F401,E402  isort:skip
import lightgbm  # noqa: F401,E402  isort:skip

import yaml
from sklearn.model_selection import ParameterGrid

from benchmarking.config import load_config, load_dataset_config, as_list
from datasets.load_datasets import Dataset
from models.model import Model, actual_max_depth
from benchmarking import BenchmarkRunner
from backends import (
    ShapTrueValueBackend,
    ShapApproxBackend,
    ShapIQTrueValueBackend,
    ShapIQApproxBackend,
    LightShapExactBackend,
    LightShapApproxBackend,
    DalexTrueBackend,
    DalexApproxBackend,
    ShapTreePathDependentBackend,
    ShapInteractionBackend,
    ShapIQTreePathDependentBackend,
    ShapIQTreeInterventionalBackend,
    ShapIQInteractionBackend,
    WoodelfTreePathDependentBackend,
    WoodelfTreeInterventionalBackend,
    WoodelfGPUPathDependentBackend,
    WoodelfGPUInterventionalBackend,
    WoodelfInteractionBackend,
    FastTreeShapBackend,
    FastTreeShapInteractionBackend,
)

warnings.filterwarnings("ignore", message="Not all budget.*",
                        category=UserWarning, module="shapiq")
warnings.filterwarnings("ignore", message="The sample size is larger.*",
                        category=UserWarning, module="shapiq")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated.*", category=UserWarning)


# GPU backends (WoodelfGPU*, GPUTreeShap*) exist in backends for
# future use but aren't wired in here yet (XGBoost-only, unverified on real
# GPU hardware) — only woodelf's GPU=True path is wired below.

# Approximator backends, selectable via the config's `approx_backends` list and
# run once per (approx_backend × approximator × budget) combination.
APPROX_MAP = {
    "shap": ShapApproxBackend,
    "shapiq": ShapIQApproxBackend,
    "lightshap": LightShapApproxBackend,
    "dalex": DalexApproxBackend,
}

# Model-agnostic true-value backends, selectable via the config's `true_backends`
# list (accuracy-comparison configs like config-accuracy.yaml pit these against
# each other directly, optionally alongside the approximation sweep).
TRUE_VALUE_BACKEND_MAP = {
    "shap_true_value": ShapTrueValueBackend,
    "shapiq_true_value": ShapIQTrueValueBackend,
    "lightshap_exact": LightShapExactBackend,
    "dalex_true_value": DalexTrueBackend,
}

# Tree-specific true-value backends, only applied to tree models (Model.is_tree).
# Keyed by (library, mode). shapiq_tree interventional was previously excluded
# (reported hangs/segfaults against shapiq 1.5.0) but re-verified clean against
# shapiq 1.5.2 across the full config-tree.yaml depth/feature-count grid — see
# ShapIQTreeInterventionalBackend's docstring. backend_timeout_s (BenchmarkRunner)
# is the safety net if it still misbehaves on an untested topology.
# "gpu_path_dependent"/"gpu_interventional" run woodelf's GPU=True (cupy-backed)
# path — unverified on real GPU hardware, skips to all-NaN without a CUDA device.
TREE_TRUE_VALUE_MAP = {
    ("shap_tree", "path_dependent"): ShapTreePathDependentBackend,
    ("shapiq_tree", "path_dependent"): ShapIQTreePathDependentBackend,
    ("shapiq_tree", "interventional"): ShapIQTreeInterventionalBackend,
    ("woodelf", "path_dependent"): WoodelfTreePathDependentBackend,
    ("woodelf", "interventional"): WoodelfTreeInterventionalBackend,
    ("woodelf", "gpu_path_dependent"): WoodelfGPUPathDependentBackend,
    ("woodelf", "gpu_interventional"): WoodelfGPUInterventionalBackend,
    ("fasttreeshap", "path_dependent"): FastTreeShapBackend,
}

# Pairwise-interaction (order-2) backends, path-dependent only. "shap_tree" is
# absent: ShapInteractionBackend is hardcoded as the always-on oracle below.
# fasttreeshap does support interactions (shap_interaction_values), just not via
# its faster "v2" algorithm — see FastTreeShapInteractionBackend's docstring.
INTERACTION_TRUE_VALUE_MAP = {
    "shapiq_tree": ShapIQInteractionBackend,
    "woodelf": WoodelfInteractionBackend,
    "fasttreeshap": FastTreeShapInteractionBackend,
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
    parser.add_argument("--output-dir", default="benchmarking/slurm_results")
    args = parser.parse_args()

    all_runs = build_all_runs(args.config)
    if args.task_id >= len(all_runs):
        print(
            f"task-id {args.task_id} out of range (max {len(all_runs) - 1})", file=sys.stderr)
        sys.exit(1)

    seed, dk, dp, mk, mp, n_background = all_runs[args.task_id]
    print(f"[task {args.task_id}] dataset={dk} {dp} | model={mk} {mp} "
          f"| seed={seed} | n_background={n_background}")

    with open(args.config) as f:
        bench = yaml.safe_load(f)["benchmark"]

    # `libraries`/`backends` were renamed to `approx_backends`/`true_backends`;
    # fail loudly instead of silently running zero backends off a stale config.
    stale = {old: new for old, new in
             (("libraries", "approx_backends"), ("backends", "true_backends"))
             if old in bench}
    if stale:
        raise ValueError(f"config uses renamed benchmark keys, update them: {stale}")

    imputer = bench.get("imputer")

    approximators = bench.get("approximators", [])
    approx_specs = [
        (APPROX_MAP[lib], {"approximator": appr, "budget": bgt})
        for lib in bench.get("approx_backends", [])
        for appr in approximators
        for bgt in bench.get("budgets", [])
        if appr in getattr(APPROX_MAP[lib], "SUPPORTED_APPROXIMATORS", approximators)
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(
        args.output_dir, f"results_{args.task_id:04d}.csv")
    if os.path.exists(output_csv):
        os.remove(output_csv)

    model_enum = Model[mk.upper()]

    # `true_backends` selects model-agnostic true-value backends directly — only
    # config-accuracy.yaml sets this, to pit true-value backends against each
    # other. Every other config omits it, so no model-agnostic true-value
    # backend runs there; those configs rely solely on their approximation
    # sweep (approx_specs above) and/or the tree-specific true-value backends
    # appended below.
    backend_names = bench.get("true_backends", [])
    unknown = [name for name in backend_names if name not in TRUE_VALUE_BACKEND_MAP]
    if unknown:
        raise ValueError(
            f"Unknown true-value backend(s) in config 'true_backends': {unknown} "
            f"(known: {list(TRUE_VALUE_BACKEND_MAP)})"
        )
    true_value_backends = [TRUE_VALUE_BACKEND_MAP[name] for name in backend_names]
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

    # For tree models the CSV's max_depth column reports the depth the fitted
    # model actually reached; the configured cap moves to max_depth_config
    # (kept because two caps can grow the exact same tree — see merge_results.py).
    base_meta = {"dataset": dk, "model": mk, "n_background": n_background, **dp, **mp}
    if model_enum.is_tree:
        base_meta["max_depth_config"] = mp.get("max_depth")
        base_meta["max_depth"] = actual_max_depth(trainer.get_model())

    runner = BenchmarkRunner(
        true_value_backends=true_value_backends,
        approximation_specs=approx_specs,
        output_csv=output_csv,
        n_background=n_background,
        n_eval=bench["n_eval"],
        seed=seed,
        imputer=imputer,
        backend_timeout_s=bench.get("backend_timeout_s"),
    )

    runner.run(
        model=trainer.get_model(),
        X=ds["X"],
        run_meta={**base_meta, "order": 1},
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

            interaction_runner = BenchmarkRunner(
                true_value_backends=interaction_backends,
                approximation_specs=[],
                output_csv=output_csv,
                n_background=n_background,
                n_eval=bench["n_eval"],
                seed=seed,
                imputer=imputer,
                backend_timeout_s=bench.get("backend_timeout_s"),
            )
            interaction_runner.run(
                model=trainer.get_model(),
                X=ds["X"],
                run_meta={**base_meta, "order": 2},
            )

    print(f"[task {args.task_id}] done -> {output_csv}")


if __name__ == "__main__":
    main()
