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

# xgboost and lightgbm must be imported — and, if used, fitted — before shapiq is
# imported anywhere in this process. Confirmed firsthand: whichever of
# {xgboost, lightgbm} vs shapiq claims its native runtime first works fine
# afterward; if shapiq is imported first, a later xgboost/lightgbm .fit() segfaults
# outright (reproduced independent of any shapiq explainer ever actually running —
# merely importing shapiq is enough). This import has no direct use below; it only
# establishes the safe load order before `from Benchmarking.backends import (...)`
# pulls in shapiq via shapiq_backend.py / tree_shapiq_backend.py.
import xgboost  # noqa: F401
import lightgbm  # noqa: F401

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
    ShapTreePathDependentBackend,
    ShapInteractionBackend,
    ShapIQTreePathDependentBackend,
    ShapIQInteractionBackend,
    WoodelfTreePathDependentBackend,
    WoodelfTreeInterventionalBackend,
    WoodelfGPUPathDependentBackend,
    WoodelfGPUInterventionalBackend,
    WoodelfInteractionBackend,
    FastTreeShapBackend,
    GPUTreeShapBackend,
    GPUTreeShapInteractionBackend,
)

APPROX_MAP = {
    "shap": ShapApproxBackend,
    "shapiq": ShapIQApproxBackend,
    "lightshap": LightShapApproxBackend,
    "dalex": DalexApproxBackend,
}

# Tree-specific true-value backends, only applied to tree models (Model.is_tree).
# Keyed by (library, mode); only libraries supporting a given mode have an entry.
# Deliberately no ("shapiq_tree", "interventional") entry: shapiq's interventional
# TreeExplainer crashes in this dependency stack (confirmed: hangs on actual
# XGBoost/LightGBM models, segfaults on plain sklearn models depending on tree
# topology) — see ShapIQTreeInterventionalBackend's docstring in
# Benchmarking/backends/tree_shapiq_backend.py for the full diagnosis.
# woodelf_gpu/gputreeshap always self-skip (NaN) without a CUDA device — see their
# class docstrings — and are unverified on real GPU hardware.
TREE_TRUE_VALUE_MAP = {
    ("shap_tree", "path_dependent"): ShapTreePathDependentBackend,
    ("shapiq_tree", "path_dependent"): ShapIQTreePathDependentBackend,
    ("woodelf", "path_dependent"): WoodelfTreePathDependentBackend,
    ("woodelf", "interventional"): WoodelfTreeInterventionalBackend,
    ("woodelf_gpu", "path_dependent"): WoodelfGPUPathDependentBackend,
    ("woodelf_gpu", "interventional"): WoodelfGPUInterventionalBackend,
    ("fasttreeshap", "path_dependent"): FastTreeShapBackend,
    ("gputreeshap", "path_dependent"): GPUTreeShapBackend,
}

# Pairwise-interaction (order-2) backends, only path-dependent (shap's interaction
# support requires it; the others follow suit for consistency — see each class's
# docstring). Keyed by library name only. "shap_tree" is intentionally absent:
# ShapInteractionBackend is hardcoded as the always-on order-2 oracle below, the
# same way ShapTrueValueBackend is hardcoded for order-1 — adding a "shap_tree"
# entry here too would instantiate it twice.
INTERACTION_TRUE_VALUE_MAP = {
    "shapiq_tree": ShapIQInteractionBackend,
    "woodelf": WoodelfInteractionBackend,
    "gputreeshap": GPUTreeShapInteractionBackend,
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

    seed = bench["seed"]
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
    if os.path.exists(output_csv):
        os.remove(output_csv)

    model_enum = Model[mk.upper()]

    # ShapTrueValueBackend must stay first: BenchmarkRunner._oracle_name() picks the
    # first true_value backend whose library == "shap" as the oracle, and
    # ShapTreePathDependentBackend below is also library "shap".
    true_value_backends = [ShapTrueValueBackend]
    if model_enum.is_tree:
        for lib in bench.get("tree_libraries", []):
            for mode in bench.get("tree_modes", []):
                cls = TREE_TRUE_VALUE_MAP.get((lib, mode))
                if cls is not None:
                    true_value_backends.append(cls)

    runner = BenchmarkRunner(
        true_value_backends=true_value_backends,
        approximation_specs=approx_specs,
        output_csv=output_csv,
        n_background=bench["n_background"],
        n_eval=bench["n_eval"],
        seed=seed,
        imputer=imputer,
    )

    dataset_enum = Dataset[dk.upper()]
    ds = dataset_enum.load_dataset(**dp, seed=seed)
    trainer = model_enum.get_model_with_params(dataset_enum, mp, seed=seed)
    trainer.fit(ds["X"], ds["y"], task=ds["task"])

    runner.run(
        model=trainer.get_model(),
        X=ds["X"],
        run_meta={"dataset": dk, "model": mk, "order": 1, **dp},
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
                n_background=bench["n_background"],
                n_eval=bench["n_eval"],
                seed=seed,
                imputer=imputer,
            )
            interaction_runner.run(
                model=trainer.get_model(),
                X=ds["X"],
                run_meta={"dataset": dk, "model": mk, "order": 2, **dp},
            )

    print(f"[task {args.task_id}] done -> {output_csv}")


if __name__ == "__main__":
    main()
