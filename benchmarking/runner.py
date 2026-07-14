import json
import time
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd

from backends.base_backend import BaseBackend, marginal_predict, nan_result, nan_interaction_result
from .eval_counter import CountingModel
from .metrics import (
    mean_abs_diff,
    relative_mae,
    sign_agreement,
    mean_sample_rho,
    additivity_gap,
    relative_additivity_gap,
)
from .timeout import BackendTimeout, time_limit


# Per-run output columns emitted by _row, as opposed to columns that identify
# a cell/row (run_meta, backend, approximator, budget, seed, ...). Shared with
# slurm/merge_results.py (identifies dedup key) and
# scripts/recompute_pairwise_metrics.py (identifies cell-grouping key).
RUN_OUTPUT_COLUMNS = {
    "library", "computation_type", "n_eval", "runtime_s", "n_model_evals",
    "additivity_gap", "relative_additivity_gap", "shapley_values",
    "shapley_n_eval", "shapley_n_features", "pairwise_metrics",
}


def spec_key(name: str, approximator=None, budget=None) -> str:
    """pairwise_metrics key for one run spec — single source of truth, shared
    with scripts/recompute_pairwise_metrics.py. True-value backends keep the
    bare backend name (merge_fasttreeshap_repair.py looks entries up by name).
    Approximation specs can share a class (same library, different
    approximator/budget), so their key must carry both. Budgets are normalized
    through int() because a CSV round-trip turns 256 into 256.0."""
    if approximator is None or (isinstance(approximator, float) and np.isnan(approximator)):
        return name
    if isinstance(budget, float) and budget == int(budget):
        budget = int(budget)
    return f"{name}|{approximator}|{budget}"


class BenchmarkRunner:
    """Runs all backends per (model, data) cell and appends one CSV row per
    backend, each carrying the accuracy metrics against every other backend
    in its pairwise_metrics column.
    """

    def __init__(
        self,
        true_value_backends: list[Type[BaseBackend]],
        approximation_specs: list[tuple[Type[BaseBackend], dict]],
        output_csv: str,
        n_background: int = 100,
        n_eval: int | None = None,
        seed: int | None = None,
        imputer: str | None = None,
        backend_timeout_s: float | None = None,
    ):
        self.true_value_backends = true_value_backends
        self.approximation_specs = approximation_specs
        self.output_csv = output_csv
        self.n_background = n_background
        self.n_eval = n_eval
        self.seed = seed
        self.imputer = imputer
        self.backend_timeout_s = backend_timeout_s

    def run(self, model, X: pd.DataFrame, run_meta: dict) -> None:
        background, X_eval = self._split_data(X)

        # Reference predictions for the additivity check; mean f(background) is
        # the fallback base value for backends that don't report their own.
        f = marginal_predict(model, X.columns)
        eval_preds = np.asarray(f(X_eval), dtype=float)
        baseline = float(np.mean(np.asarray(f(background), dtype=float)))

        results = self._run_true_value_backends(model, background, X_eval)
        results += self._run_approximation_specs(model, background, X_eval)

        rows = [self._row(run_meta, r, results, eval_preds, baseline) for r in results]
        self._append_to_csv(rows)

    def _split_data(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(X) <= self.n_background:
            raise ValueError(
                f"X has {len(X)} rows but n_background={self.n_background}; "
                "no evaluation rows remain."
            )
        if X.shape[1] < 4:
            raise ValueError(
                f"n_features={X.shape[1]} < 4: the approximators require at least 4 "
                "features (lightshap permutation sampling is unsupported below 4, and "
                "below that approximation is trivially exact anyway). Raise the "
                "n_features floor in the config."
            )
        background = X.iloc[:self.n_background]
        end = None if self.n_eval is None else self.n_background + self.n_eval
        return background, X.iloc[self.n_background:end]

    def _run_true_value_backends(self, model, background, X_eval) -> list[dict]:
        config = {"n_background": self.n_background}
        if self.seed is not None:
            config["seed"] = self.seed
        return [
            {
                **self._run_backend(cls, model, background, config, X_eval, label=cls.name),
                "config": {},
                "n_model_evals": float("nan"),
            }
            for cls in self.true_value_backends
        ]

    def _run_approximation_specs(self, model, background, X_eval) -> list[dict]:
        results = []
        for cls, spec in self.approximation_specs:
            config = {**spec}
            if self.seed is not None:
                config["seed"] = self.seed
            if self.imputer is not None:
                config["imputer"] = self.imputer
            if self.n_eval is not None:
                config["n_eval"] = self.n_eval
            counter = CountingModel(model)
            result = self._run_backend(cls, counter, background, config, X_eval,
                                       label=f"{cls.name} ({spec})")
            results.append({**result, "config": spec, "n_model_evals": counter.n_rows})
        return results

    def _run_backend(self, cls, model, background, config, X_eval, label: str) -> dict:
        """Run one backend with timeout and crash isolation: a hung or crashing
        backend yields an all-NaN result instead of killing the whole
        (model, dataset) cell. A timed-out row records runtime_s ~= the timeout
        (right-censored) — filter NaN-value rows out of runtime aggregations."""
        backend = None
        t0 = time.perf_counter()
        try:
            with time_limit(self.backend_timeout_s):
                backend = cls(model, background, config)
                contrib = backend.run_explainer(X_eval)
        except BackendTimeout:
            print(f"  [SKIP] {label}: exceeded {self.backend_timeout_s}s timeout")
            contrib = self._nan_contrib(cls, X_eval)
        except Exception as e:
            print(f"  [BUG] {label} crashed: {e.__class__.__name__}: {e}")
            contrib = self._nan_contrib(cls, X_eval)
        return {
            "cls": cls,
            "contrib": contrib,
            "runtime": time.perf_counter() - t0,
            "baseline": backend.baseline_ if backend is not None else None,
        }

    @staticmethod
    def _nan_contrib(cls: Type[BaseBackend], X_eval: pd.DataFrame) -> pd.DataFrame:
        return nan_result(X_eval) if cls.order == 1 else nan_interaction_result(X_eval)

    def _row(self, run_meta, candidate, all_results, eval_preds, baseline) -> dict:
        c_contrib = candidate["contrib"]
        cls = candidate["cls"]
        config = candidate["config"]
        # The backend's own game's base value when reported (path-dependent tree
        # backends), else the marginal game's (mean f over the background).
        base = candidate["baseline"] if candidate["baseline"] is not None else baseline
        gap = additivity_gap(c_contrib, eval_preds, base)

        pairwise = {
            spec_key(ref["cls"].name, ref["config"].get("approximator"),
                     ref["config"].get("budget")): self._pair_metrics(candidate, ref)
            for ref in all_results
        }

        return {
            **run_meta,
            "backend": cls.name,
            "library": cls.library,
            "computation_type": cls.computation_type,
            "approximator": config.get("approximator", float("nan")),
            "budget": config.get("budget", float("nan")),
            "seed": self.seed if self.seed is not None else float("nan"),
            "imputer": self.imputer if self.imputer is not None else float("nan"),
            "n_eval": len(c_contrib),
            "runtime_s": round(candidate["runtime"], 4),
            "n_model_evals": candidate["n_model_evals"],
            "base_value": base,
            "additivity_gap": gap,
            "relative_additivity_gap": relative_additivity_gap(c_contrib, eval_preds, base, gap=gap),
            "shapley_values": json.dumps(c_contrib.values.flatten().tolist()),
            "shapley_n_eval": c_contrib.shape[0],
            "shapley_n_features": c_contrib.shape[1],
            "pairwise_metrics": json.dumps(pairwise),
        }

    @staticmethod
    def _pair_metrics(candidate: dict, reference: dict) -> dict:
        c_contrib = candidate["contrib"]
        if candidate is reference:
            return {
                "mean_abs_diff": 0.0,
                "relative_mae": 0.0,
                "sign_agreement": float(sign_agreement(c_contrib, c_contrib)),
                "mean_sample_rho": 1.0,
            }
        r_contrib = reference["contrib"]
        return {
            "mean_abs_diff": mean_abs_diff(c_contrib, r_contrib),
            "relative_mae": relative_mae(c_contrib, r_contrib),
            "sign_agreement": sign_agreement(c_contrib, r_contrib),
            "mean_sample_rho": mean_sample_rho(c_contrib, r_contrib),
        }

    def _append_to_csv(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        write_header = not Path(self.output_csv).exists()
        df.to_csv(self.output_csv, mode="a", header=write_header, index=False)
