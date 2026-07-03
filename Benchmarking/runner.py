import json
import time
from pathlib import Path
from typing import Type

import pandas as pd

from .backends.base_backend import BaseBackend, nan_result, nan_interaction_result
from .eval_counter import CountingModel
from .metrics import mean_abs_diff, relative_mae, sign_agreement, mean_sample_rho
from .timeout import BackendTimeout, time_limit


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

        # --- run every backend once, record contributions and metadata ---
        results: list[dict] = []

        for cls in self.true_value_backends:
            t0 = time.perf_counter()
            try:
                with time_limit(self.backend_timeout_s):
                    contrib = cls(model, background).run_explainer(X_eval)
            except BackendTimeout:
                print(f"  [SKIP] {cls.name}: exceeded {self.backend_timeout_s}s timeout")
                contrib = nan_result(X_eval) if cls.order == 1 else nan_interaction_result(X_eval)
            runtime = time.perf_counter() - t0
            results.append({
                "cls": cls,
                "config": {},
                "contrib": contrib,
                "runtime": runtime,
                "n_model_evals": float("nan"),
            })

        for cls, config in self.approximation_specs:
            counter = CountingModel(model)
            run_config = {**config}
            if self.seed is not None:
                run_config["seed"] = self.seed
            if self.imputer is not None:
                run_config["imputer"] = self.imputer
            t0 = time.perf_counter()
            contrib = cls(counter, background, run_config).run_explainer(X_eval)
            runtime = time.perf_counter() - t0
            results.append({
                "cls": cls,
                "config": config,
                "contrib": contrib,
                "runtime": runtime,
                "n_model_evals": counter.n_rows,
            })

        # --- emit one row per backend with pairwise metrics dict ---
        rows: list[dict] = []
        for candidate in results:
            rows.append(self._row(run_meta, candidate, results))

        self._append_to_csv(rows)

    def _row(self, run_meta, candidate, all_results) -> dict:
        c_contrib = candidate["contrib"]
        cls = candidate["cls"]
        config = candidate["config"]

        pairwise = {}
        for reference in all_results:
            ref_name = reference["cls"].name
            if candidate is reference:
                pairwise[ref_name] = {
                    "mean_abs_diff": 0.0,
                    "relative_mae": 0.0,
                    "sign_agreement": float(sign_agreement(c_contrib, c_contrib)),
                    "mean_sample_rho": 1.0,
                }
            else:
                r_contrib = reference["contrib"]
                pairwise[ref_name] = {
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
            "shapley_values": json.dumps(c_contrib.values.flatten().tolist()),
            "shapley_n_eval": c_contrib.shape[0],
            "shapley_n_features": c_contrib.shape[1],
            "pairwise_metrics": json.dumps(pairwise),
        }

    def _append_to_csv(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        write_header = not Path(self.output_csv).exists()
        df.to_csv(self.output_csv, mode="a", header=write_header, index=False)
