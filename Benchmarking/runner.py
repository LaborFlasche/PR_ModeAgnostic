import time
from pathlib import Path
from typing import Type

import pandas as pd

from .backends.base_backend import BaseBackend
from .eval_counter import CountingModel
from .metrics import mean_abs_diff, relative_mae, sign_agreement, mean_sample_rho


class BenchmarkRunner:
    """Runs one exact reference + many approximation configs per (model, data) cell.

    The exact value is cheap (TreeSHAP/LinearSHAP via ``shap.Explainer``), so it is
    recomputed in every cell and used as the in-memory ground truth — no caching.
    Each approximation spec is a ``(backend_class, config)`` pair; the model is
    wrapped in a ``CountingModel`` per spec so the real number of model evaluations
    is recorded as the fair, library-agnostic budget axis.
    """

    def __init__(
        self,
        true_value_backends: list[Type[BaseBackend]],
        approximation_specs: list[tuple[Type[BaseBackend], dict]],
        output_csv: str,
        n_background: int = 100,
        n_eval: int | None = None,
    ):
        self.true_value_backends = true_value_backends
        self.approximation_specs = approximation_specs
        self.output_csv = output_csv
        self.n_background = n_background
        self.n_eval = n_eval

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

        rows: list[dict] = []
        oracle_name = self._oracle_name()

        # --- exact reference backend(s): the first shap true_value is the oracle ---
        true_contributions: dict[str, pd.DataFrame] = {}
        true_runtimes: dict[str, float] = {}
        for cls in self.true_value_backends:
            t0 = time.perf_counter()
            true_contributions[cls.name] = cls(model, background).run_explainer(X_eval)
            true_runtimes[cls.name] = time.perf_counter() - t0

        oracle = true_contributions.get(oracle_name)
        for cls in self.true_value_backends:
            is_oracle = cls.name == oracle_name
            rows.append(self._row(
                run_meta, cls,
                contrib=true_contributions[cls.name],
                reference=None if is_oracle else oracle,
                runtime=true_runtimes[cls.name],
                approximator=None, budget=None, n_model_evals=None,
                reference_backend=None if is_oracle else oracle_name,
            ))

        # --- approximation specs, each measured against the oracle ---
        for cls, config in self.approximation_specs:
            counter = CountingModel(model)
            t0 = time.perf_counter()
            contrib = cls(counter, background, config).run_explainer(X_eval)
            runtime = time.perf_counter() - t0
            rows.append(self._row(
                run_meta, cls,
                contrib=contrib, reference=oracle, runtime=runtime,
                approximator=config.get("approximator"), budget=config.get("budget"),
                n_model_evals=counter.n_rows, reference_backend=oracle_name,
            ))

        self._append_to_csv(rows)

    def _row(self, run_meta, cls, *, contrib, reference, runtime, approximator,
             budget, n_model_evals, reference_backend) -> dict:
        if reference is not None:
            mad = mean_abs_diff(contrib, reference)
            rmae = relative_mae(contrib, reference)
            sa = sign_agreement(contrib, reference)
            msr = mean_sample_rho(contrib, reference)
        else:
            mad = rmae = sa = msr = float("nan")
        return {
            **run_meta,
            "backend": cls.name,
            "library": cls.library,
            "computation_type": cls.computation_type,
            "approximator": approximator if approximator is not None else float("nan"),
            "budget": budget if budget is not None else float("nan"),
            "n_eval": len(contrib),
            "runtime_s": round(runtime, 4),
            "n_model_evals": n_model_evals if n_model_evals is not None else float("nan"),
            "mean_abs_diff": mad,
            "relative_mae": rmae,
            "sign_agreement": sa,
            "mean_sample_rho": msr,
            "reference_backend": reference_backend if reference_backend is not None else float("nan"),
        }

    def _oracle_name(self) -> str | None:
        for cls in self.true_value_backends:
            if cls.library == "shap":
                return cls.name
        return self.true_value_backends[0].name if self.true_value_backends else None

    def _append_to_csv(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        write_header = not Path(self.output_csv).exists()
        df.to_csv(self.output_csv, mode="a", header=write_header, index=False)
