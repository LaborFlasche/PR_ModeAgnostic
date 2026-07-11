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

from slurm.task_grid import build_all_runs, load_bench
from datasets.load_datasets import Dataset
from models.model import Model
from benchmarking import BenchmarkRunner
from backends import (
    ShapIQTrueValueBackend,
    ShapIQApproxBackend,
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
    "shapiq": ShapIQApproxBackend,
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
    """Identity metadata for every CSV row of one cell. Records the device
    (cpu/cuda sweeps of the same config are otherwise indistinguishable in
    merged results) and flattens dataset/model params, matching
    run_benchmark.py's run_meta. Non-scalar model params (e.g. mlp's
    hidden_sizes list) are JSON-encoded so CSV cells stay hashable for
    merge_results.py's drop_duplicates."""
    mp_meta = {k: json.dumps(v) if isinstance(v, (list, dict)) else v
               for k, v in (model_params or {}).items()}
    return {"dataset": dataset, "model": model, "order": 1,
            "n_background": n_background, "device": device,
            **dataset_params, **mp_meta}


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

    seed, dataset, dataset_params, model, model_params, n_background = all_runs[args.task_id]
    print(f"[task {args.task_id}] dataset={dataset} {dataset_params} "
          f"| model={model} {model_params} | seed={seed} | n_background={n_background}")

    bench = load_bench(args.config)

    imputer = bench["imputer"]

    approx_specs = build_approx_specs(bench)

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, f"results_{args.task_id:04d}.csv")
    if os.path.exists(output_csv):
        os.remove(output_csv)

    true_value_backends = [
        NN_TRUE_VALUE_MAP[lib]
        for lib in bench.get("nn_true_value_libraries", [])
        if lib in NN_TRUE_VALUE_MAP
    ]

    dataset_enum = Dataset[dataset.upper()]
    ds = dataset_enum.load_dataset(**dataset_params, seed=seed)
    device = bench.get("device", "cpu")
    trainer = Model[model.upper()].get_model_with_params(
        dataset_enum, {**model_params, "device": device}, seed=seed)
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
        run_meta=build_run_meta(dataset=dataset, dataset_params=dataset_params,
                                model=model, n_background=n_background,
                                device=device, model_params=model_params),
    )

    print(f"[task {args.task_id}] done -> {output_csv}")


if __name__ == "__main__":
    main()
