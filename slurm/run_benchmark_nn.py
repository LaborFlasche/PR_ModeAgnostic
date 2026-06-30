#!/usr/bin/env python3
"""
SLURM array worker for neural network benchmarks (RQ3).

Runs exactly one (dataset, model) cell with NN-specific gradient-based backends
(captum, shap_nn) and model-agnostic backends (lightshap, dalex), plus
true-value backends that auto-select their approximator (shapiq).

Usage:
    python slurm/run_benchmark_nn.py --task-id $SLURM_ARRAY_TASK_ID

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

from Models.dataset_and_models import Dataset, Model
from task_grid import build_all_runs_nn as build_all_runs
from Benchmarking import BenchmarkRunner
from Benchmarking.backends import (
    ShapIQTrueValueBackend,
    CaptumApproxBackend,
    ShapNNApproxBackend,
    LightShapApproxBackend,
    DalexApproxBackend,
)

APPROX_MAP = {
    "captum": CaptumApproxBackend,
    "shap_nn": ShapNNApproxBackend,
    "lightshap": LightShapApproxBackend,
    "dalex": DalexApproxBackend,
}

NN_TRUE_VALUE_MAP = {
    "shapiq": ShapIQTrueValueBackend,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True,
                        help="SLURM_ARRAY_TASK_ID — index into all (dataset, model) combinations")
    parser.add_argument("--config", default="configs/config-neural-networks-RQ3.yaml")
    parser.add_argument("--output-dir", default="Benchmarking/slurm_results")
    args = parser.parse_args()

    all_runs = build_all_runs(args.config)
    if args.task_id >= len(all_runs):
        print(f"task-id {args.task_id} out of range (max {len(all_runs) - 1})", file=sys.stderr)
        sys.exit(1)

    dk, dp, mk, mp, n_background = all_runs[args.task_id]
    print(f"[task {args.task_id}] dataset={dk} {dp} | model={mk} {mp} | n_background={n_background}")

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

    true_value_backends = []
    for lib in bench.get("nn_true_value_libraries", []):
        cls = NN_TRUE_VALUE_MAP.get(lib)
        if cls is not None:
            true_value_backends.append(cls)

    dataset_enum = Dataset[dk.upper()]
    ds = dataset_enum.load_dataset(**dp, seed=seed)
    device = bench.get("device", "cpu")
    trainer = model_enum.get_model_with_params(dataset_enum, {**mp, "device": device}, seed=seed)
    trainer.fit(ds["X"], ds["y"], task=ds["task"])

    runner = BenchmarkRunner(
        true_value_backends=true_value_backends,
        approximation_specs=approx_specs,
        output_csv=output_csv,
        n_background=n_background,
        n_eval=bench["n_eval"],
        seed=seed,
        imputer=imputer,
    )
    runner.run(
        model=trainer.get_model(),
        X=ds["X"],
        run_meta={"dataset": dk, "model": mk, "order": 1, "n_background": n_background, **dp},
    )

    print(f"[task {args.task_id}] done -> {output_csv}")


if __name__ == "__main__":
    main()
