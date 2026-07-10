import json
import time
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd

from .backends.base_backend import BaseBackend, marginal_predict, nan_result, nan_interaction_result
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


def spec_key(name: str, approximator=None, budget=None) -> str:
    """pairwise_metrics key for one run spec — single source of truth, shared
    with scripts/recompute_pairwise_metrics.py so offline recomputes emit the
    exact keys fresh runs do. True-value backends run once per class and keep
    the bare backend name (merge_fasttreeshap_repair.py looks entries up by
    name). Approximation specs can share a class (same library, different
    approximator/budget), so the key must carry both or the specs would
    overwrite each other's pairwise entries. Budgets are normalized through
    int() because a CSV round-trip turns 256 into 256.0."""
    if approximator is None or (isinstance(approximator, float) and np.isnan(approximator)):
        return name
    if isinstance(budget, float) and budget == int(budget):
        budget = int(budget)
    return f"{name}|{approximator}|{budget}"


class BenchmarkRunner:
    """Runs all backends per (model, data) cell and emits pairwise comparison rows.

    Every backend runs once. Then for each ordered pair (candidate, reference)
    the four accuracy metrics are computed and one CSV row is emitted.
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
        if self.n_eval is None:
            X_eval = X.iloc[self.n_background:]
        else:
            X_eval = X.iloc[self.n_background:self.n_background + self.n_eval]

        # Reference predictions/baseline for the additivity check, using mean
        # f(background) as the fallback base value for marginal-game backends.
  
        f = marginal_predict(model, X.columns)
        eval_preds = np.asarray(f(X_eval), dtype=float)
        baseline = float(np.mean(np.asarray(f(background), dtype=float)))

        # --- run every backend once, record contributions and metadata ---
        results: list[dict] = []

        true_value_config = {"n_background": self.n_background}
        if self.seed is not None:
            true_value_config["seed"] = self.seed

        for cls in self.true_value_backends:
            backend = cls(model, background, true_value_config)
            t0 = time.perf_counter()
            try:
                with time_limit(self.backend_timeout_s):
                    contrib = cls(model, background, true_value_config).run_explainer(X_eval)
            except BackendTimeout:
                print(f"  [SKIP] {cls.name}: exceeded {self.backend_timeout_s}s timeout")
                contrib = self._nan_contrib(cls, X_eval)
            except Exception as e:
                print(f"  [BUG] {cls.name} crashed: {e.__class__.__name__}: {e}")
                contrib = self._nan_contrib(cls, X_eval)
            runtime = time.perf_counter() - t0
            results.append({
                "cls": cls,
                "config": {},
                "contrib": contrib,
                "runtime": runtime,
                "n_model_evals": float("nan"),
                "baseline": backend.baseline_,
            })

        for cls, config in self.approximation_specs:
            counter = CountingModel(model)
            run_config = {**config}
            if self.seed is not None:
                run_config["seed"] = self.seed
            if self.imputer is not None:
                run_config["imputer"] = self.imputer
            t0 = time.perf_counter()
            # Same timeout + crash isolation as the true-value loop: one hung or
            # crashing (library, approximator, budget) combination gets an
            # all-NaN row instead of killing the whole (model, dataset) cell.
            # A timed-out row records runtime_s ~= the timeout (right-censored)
            # — filter NaN-value rows out of mean-runtime aggregations.
            backend = cls(counter, background, run_config)
            try:
                with time_limit(self.backend_timeout_s):
                    contrib = backend.run_explainer(X_eval)
            except BackendTimeout:
                print(f"  [SKIP] {cls.name} ({config}): exceeded {self.backend_timeout_s}s timeout")
                contrib = nan_result(X_eval)
            except Exception as e:
                print(f"  [BUG] {cls.name} ({config}) crashed: {e.__class__.__name__}: {e}")
                contrib = nan_result(X_eval)
            runtime = time.perf_counter() - t0
            results.append({
                "cls": cls,
                "config": config,
                "contrib": contrib,
                "runtime": runtime,
                "n_model_evals": counter.n_rows,
                "baseline": backend.baseline_,
            })

        # --- emit one row per backend with pairwise metrics dict ---
        rows: list[dict] = []
        for candidate in results:
            rows.append(self._row(run_meta, candidate, results, eval_preds, baseline))

        self._append_to_csv(rows)

    @staticmethod
    def _nan_contrib(cls: Type[BaseBackend], X_eval: pd.DataFrame) -> pd.DataFrame:
        return nan_result(X_eval) if cls.order == 1 else nan_interaction_result(X_eval)

    @staticmethod
    def _ref_key(result: dict) -> str:
        """Unique key for a backend run within a cell, as a backend
        name is not unique inside a cell: one name (e.g. "shap_approx")
        has several runs differing by approximator + budget. Keying pairwise
        metrics by name alone collapses them so only the last-listed run survives,
        which both loses comparisons and turns the self-entry non-zero.
        True-value backends carry no such config and key on the
        name alone (e.g. "shapiq_approx|kernel|256", "shap_true_value").
        """
        name = result["cls"].name
        approximator = result["config"].get("approximator")
        budget = result["config"].get("budget")
        parts = [name]
        if approximator is not None:
            parts.append(str(approximator))
        if budget is not None:
            parts.append(format(budget, "g") if isinstance(budget, float) else str(budget))
        return "|".join(parts)

    @staticmethod
    def _comparable(candidate: dict, reference: dict) -> bool:
        """Checks whether values should be compared for pairwise metrics.

        A row's Shapley values are only compared against runs that share every
        setting except the library: Any row is always compared against the exact/true-value backends (the
        ground truth each approximation is measured against) and against itself.
        """
        if candidate is reference:
            return True
        if reference["cls"].computation_type == "true_value" or candidate["cls"].computation_type == "true_value":
            return True
        cand_cfg, ref_cfg = candidate["config"], reference["config"]
        return (cand_cfg.get("approximator") == ref_cfg.get("approximator")
                and cand_cfg.get("budget") == ref_cfg.get("budget"))

    def _row(self, run_meta, candidate, all_results, eval_preds, baseline) -> dict:
        c_contrib = candidate["contrib"]
        cls = candidate["cls"]
        config = candidate["config"]
        # The backend's own game's base value when reported (path-dependent tree
        # backends), else the marginal game's (mean f over the background).
        base = candidate["baseline"] if candidate["baseline"] is not None else baseline
        gap = additivity_gap(c_contrib, eval_preds, base)

        pairwise = {}
        for reference in all_results:
            # Only compare like-for-like: exact/true-value references always, and
            # other approximations only at the same approximator+budget (see
            # _comparable). Different-setting pairs (kernel@128 vs permutation@2048)
            # are meaningless and skipped.
            if not self._comparable(candidate, reference):
                continue
            # Key by name + config: within one cell a single backend name has
            # several runs (approximator x budget), so keying by name alone
            # collapses them and only the last-listed run survives.
            ref_key = self._ref_key(reference)
            if candidate is reference:
                pairwise[ref_key] = {
                    "mean_abs_diff": 0.0,
                    "relative_mae": 0.0,
                    "sign_agreement": float(sign_agreement(c_contrib, c_contrib)),
                    "mean_sample_rho": 1.0,
                }
            else:
                r_contrib = reference["contrib"]
                pairwise[ref_key] = {
                    "mean_abs_diff": mean_abs_diff(c_contrib, r_contrib),
                    "relative_mae": relative_mae(c_contrib, r_contrib),
                    "sign_agreement": sign_agreement(c_contrib, r_contrib),
                    "mean_sample_rho": mean_sample_rho(c_contrib, r_contrib),
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

    def _append_to_csv(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        write_header = not Path(self.output_csv).exists()
        df.to_csv(self.output_csv, mode="a", header=write_header, index=False)
