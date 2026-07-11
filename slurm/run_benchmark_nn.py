#!/usr/bin/env python3
"""
SLURM array worker for neural network benchmarks (RQ3).

Runs exactly one (dataset, model) cell with NN-specific gradient-based backends
(captum, shap_nn) and model-agnostic backends (lightshap, dalex), plus
true-value backends that auto-select their approximator (shapiq).

Usage:
    python slurm/run_benchmark_nn.py --task-id $SLURM_ARRAY_TASK_ID

Run from the repo root so that models/, benchmarking/, configs/ are importable.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Not all budget.*", category=UserWarning, module="shapiq")
warnings.filterwarnings("ignore", message="The sample size is larger.*", category=UserWarning, module="shapiq")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)

import yaml
from sklearn.model_selection import ParameterGrid

from models.config_parser import load_config, load_dataset_config, as_list
from models.dataset_and_models import Dataset, Model
from benchmarking import BenchmarkRunner
from benchmarking.backends import (
    ShapIQTrueValueBackend,
    ShapIQNNApproxBackend,
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
    "shapiq_proxy": ShapIQNNApproxBackend,
}

NN_TRUE_VALUE_MAP = {
    "shapiq": ShapIQTrueValueBackend,
}


def build_approx_specs(bench: dict) -> list[tuple]:
    """(backend class, config) per approx_backend × approximator × budget, filtered by
    each backend's SUPPORTED_APPROXIMATORS. An optional top-level `proxy_model`
    key ("xgboost"/"lightgbm"/"tree"/"linear") is forwarded to ProxySHAP specs
    only — see ShapIQNNApproxBackend for why non-default proxies matter locally."""
    if "libraries" in bench:
        raise ValueError(
            "config key 'libraries' was renamed to 'approx_backends', update the config")
    proxy_model = bench.get("proxy_model")
    return [
        (
            APPROX_MAP[lib],
            {"approximator": appr, "budget": bgt}
            | ({"proxy_model": proxy_model} if proxy_model and appr == "proxy" else {}),
        )
        for lib in bench["approx_backends"]
        for appr in bench["approximators"]
        for bgt in bench["budgets"]
        if appr in getattr(APPROX_MAP[lib], "SUPPORTED_APPROXIMATORS", bench["approximators"])
    ]


def build_run_meta(*, dataset: str, dataset_params: dict, model: str,
                   n_background: int, device: str,
                   model_params: dict | None = None) -> dict:
    """Identity metadata for every CSV row of one cell. Records the device —
    cpu and cuda sweeps of the same config are otherwise indistinguishable in
    merged results — and flattens dataset and model params in (parity with
    run_benchmark.py's run_meta). Non-scalar model params (e.g. mlp's
    hidden_sizes list) are JSON-encoded so the CSV cells stay hashable for
    merge_results.py's drop_duplicates."""
    mp_meta = {k: json.dumps(v) if isinstance(v, (list, dict)) else v
               for k, v in (model_params or {}).items()}
    return {"dataset": dataset, "model": model, "order": 1,
            "n_background": n_background, "device": device,
            **dataset_params, **mp_meta}


def build_all_runs(config_path: str) -> list[tuple]:
    """Every independent benchmark cell as (seed, dataset, dataset_params, model,
    model_params, n_background) tuples — same shape and sweep order as
    slurm/run_benchmark.py's build_all_runs, so count_tasks.py counts both alike.
    ``seed`` and ``n_background`` may each be a scalar or a list."""
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
    parser.add_argument("--config", default="configs/RQ3-neural-networks/config-neural-networks-gpu.yaml")
    parser.add_argument("--output-dir", default="benchmarking/slurm_results")
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

    approx_specs = build_approx_specs(bench)

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
        backend_timeout_s=bench.get("backend_timeout_s"),
    )
    runner.run(
        model=trainer.get_model(),
        X=ds["X"],
        run_meta=build_run_meta(dataset=dk, dataset_params=dp, model=mk,
                                n_background=n_background, device=device,
                                model_params=mp),
    )

    print(f"[task {args.task_id}] done -> {output_csv}")


if __name__ == "__main__":
    main()
