#!/usr/bin/env python3
"""
SLURM array worker for the neural-network benchmark — runs exactly one
(dataset, model) cell using Captum backends.

Usage:
    python slurm/run_benchmark_nn.py --task-id $SLURM_ARRAY_TASK_ID

Run from the repo root so that Models/, Benchmarking/, configs/ are importable.

Design differences from run_benchmark.py
-----------------------------------------
- Model:   only pytorch_neural_network (torch.nn.Module).
- Oracle:  none — no exact oracle for arbitrary NNs; metrics are recorded
           without a reference.
- Counter: TorchCountingModel instead of CountingModel so that (a) gradient
           methods can backpropagate through the wrapper and (b) forward passes
           are still counted for the n_model_evals column.
- Target:  the output neuron to explain — 0 for regression, 1 for binary
           classification (P(class 1)), 0 for multi-class.
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Setting forward.*", category=UserWarning)

import yaml
import pandas as pd
from sklearn.model_selection import ParameterGrid

from Models.config_parser import load_config, load_dataset_config
from Models.dataset_and_models import Dataset, Model
from Benchmarking.backends import CaptumBackend
from Benchmarking.eval_counter import TorchCountingModel


def build_all_runs(config_path: str) -> list[tuple]:
    model_config = load_config(config_path)
    dataset_config = load_dataset_config(config_path)
    # Only pytorch_neural_network entries are relevant here
    nn_config = {k: v for k, v in model_config.items() if k == "pytorch_neural_network"}
    model_runs = [(k, p) for k, pg in nn_config.items() for p in ParameterGrid(pg)]
    dataset_runs = [(k, p) for k, pg in dataset_config.items() for p in ParameterGrid(pg)]
    return [
        (dk, dp, mk, mp)
        for dk, dp in dataset_runs
        for mk, mp in model_runs
    ]


def _target_for_task(task: str, y) -> int:
    """Class index to explain.

    Regression → 0 (single output neuron).
    Binary classification → 1 (P(positive class)).
    Multi-class classification → 0 (first class; picked consistently).
    """
    if task != "classification":
        return 0
    n_classes = len(set(y.values if hasattr(y, "values") else y))
    return 1 if n_classes == 2 else 0


def _row(run_meta, name, library, computation_type, approximator, budget,
         contrib, runtime, n_model_evals) -> dict:
    return {
        **run_meta,
        "backend": name,
        "library": library,
        "computation_type": computation_type,
        "approximator": approximator if approximator is not None else float("nan"),
        "budget": budget if budget is not None else float("nan"),
        "n_eval": len(contrib),
        "runtime_s": round(runtime, 4),
        "n_model_evals": n_model_evals if n_model_evals is not None else float("nan"),
    }


def _append_csv(rows: list[dict], path: str) -> None:
    df = pd.DataFrame(rows)
    write_header = not Path(path).exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--config", default="configs/config-nn.yaml")
    parser.add_argument("--output-dir", default="Benchmarking/results_nn")
    args = parser.parse_args()

    all_runs = build_all_runs(args.config)
    if args.task_id >= len(all_runs):
        print(f"task-id {args.task_id} out of range (max {len(all_runs) - 1})", file=sys.stderr)
        sys.exit(1)

    dk, dp, mk, mp = all_runs[args.task_id]
    print(f"[task {args.task_id}] dataset={dk} {dp} | model={mk} {mp}")

    with open(args.config) as f:
        bench = yaml.safe_load(f)["benchmark"]

    n_background = bench["n_background"]
    n_eval = bench.get("n_eval")
    captum_methods = bench["captum_methods"]
    budgets = bench["budgets"]

    # Load dataset and train model
    dataset_enum = Dataset[dk.upper()]
    ds = dataset_enum.load_dataset(**dp)
    trainer = Model[mk.upper()].get_model_with_params(dataset_enum, mp)
    trainer.fit(ds["X"], ds["y"], task=ds["task"])
    model = trainer.get_model()

    X = ds["X"]
    if len(X) <= n_background:
        print(f"[task {args.task_id}] skip: only {len(X)} rows, need >{n_background}", file=sys.stderr)
        sys.exit(0)

    background = X.iloc[:n_background]
    X_eval = X.iloc[n_background:] if n_eval is None else X.iloc[n_background:n_background + n_eval]

    target = _target_for_task(ds["task"], ds["y"])
    run_meta = {"dataset": dk, "model": mk, "hidden_dims": str(mp.get("hidden_dims")), **dp}

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, f"results_nn_{args.task_id:04d}.csv")

    rows = []

    # --- approximation specs ---
    # Deterministic methods (deep_lift_shap) run once at each budget listed,
    # but the budget value has no effect — recorded as-is for reference.
    for method in captum_methods:
        method_budgets = budgets if method != "deep_lift_shap" else [None]
        for budget in method_budgets:
            config = {"method": method, "approximator": method, "target": target}
            if budget is not None:
                config["budget"] = budget

            label = f"{method}" + (f" budget={budget}" if budget else " (deterministic)")
            print(f"[task {args.task_id}] running {label} ...")

            counter = TorchCountingModel(model)
            t0 = time.perf_counter()
            try:
                contrib = CaptumBackend(counter, background, config).run_explainer(X_eval)
            except Exception as exc:
                print(f"[task {args.task_id}] ERROR {label}: {exc}", file=sys.stderr)
                continue
            runtime = time.perf_counter() - t0

            rows.append(_row(
                run_meta,
                name=f"captum_{method}",
                library="captum",
                computation_type="approximation",
                approximator=method,
                budget=budget,
                contrib=contrib,
                runtime=runtime,
                n_model_evals=counter.n_rows,
            ))

    _append_csv(rows, output_csv)
    print(f"[task {args.task_id}] done → {output_csv}")


if __name__ == "__main__":
    main()
