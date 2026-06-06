import time
from pathlib import Path
from typing import Type

import pandas as pd

from .backends.base_backend import BaseBackend
from .metrics import mean_abs_diff, sign_agreement, mean_sample_rho


class BenchmarkRunner:
    def __init__(
        self,
        backends: list[Type[BaseBackend]],
        output_csv: str,
        n_background: int = 100,
        n_eval: int | None = None,
        compare_with_reference: bool = True,
    ):
        self.backends = backends
        self.output_csv = output_csv
        self.n_background = n_background
        self.n_eval = n_eval
        self.compare_with_reference = compare_with_reference

    def run(self, model, X: pd.DataFrame, run_meta: dict) -> None:
        if len(X) <= self.n_background:
            raise ValueError(
                f"X has {len(X)} rows but n_background={self.n_background}; "
                "no evaluation rows remain."
            )
        background = X.iloc[:self.n_background]
        if self.n_eval is None:
            X_eval = X.iloc[self.n_background:]
        else:
            X_eval = X.iloc[self.n_background:self.n_background + self.n_eval]

        instances = {cls.name: cls(model, background) for cls in self.backends}

        contributions: dict[str, pd.DataFrame] = {}
        runtimes: dict[str, float] = {}
        for name, backend in instances.items():
            t0 = time.perf_counter()
            contributions[name] = backend.run_explainer(X_eval)
            runtimes[name] = time.perf_counter() - t0

        rows = []
        for cls in self.backends:
            name = cls.name
            chosen_method = getattr(instances[name], "chosen_method", None)

            if self.compare_with_reference:
                ref_name = self._resolve_reference(cls)
                if ref_name is not None and ref_name in contributions:
                    ref = contributions[ref_name]
                    cur = contributions[name]
                    mad = mean_abs_diff(cur, ref)
                    sa = sign_agreement(cur, ref)
                    msr = mean_sample_rho(cur, ref)
                else:
                    if ref_name is not None:
                        print(f"Warning: reference backend '{ref_name}' not in run — accuracy metrics NaN for '{name}'")
                    mad = sa = msr = float("nan")
            else:
                ref_name = None
                mad = sa = msr = float("nan")

            rows.append({
                **run_meta,
                "backend": name,
                "library": cls.library,
                "computation_type": cls.computation_type,
                "chosen_method": chosen_method if chosen_method is not None else float("nan"),
                "n_eval": len(X_eval),
                "runtime_s": round(runtimes[name], 4),
                "mean_abs_diff": mad,
                "sign_agreement": sa,
                "mean_sample_rho": msr,
                "reference_backend": ref_name if ref_name is not None else float("nan"),
            })

        self._append_to_csv(rows)

    def _resolve_reference(self, backend_cls: Type[BaseBackend]) -> str | None:
        if backend_cls.library == "shap" and backend_cls.computation_type == "true_value":
            return None
        if backend_cls.computation_type == "approximation":
            for cls in self.backends:
                if cls.library == backend_cls.library and cls.computation_type == "true_value":
                    return cls.name
            # Fall back to shap true_value as the universal ground truth
            for cls in self.backends:
                if cls.library == "shap" and cls.computation_type == "true_value":
                    return cls.name
            return None
        # true_value, non-shap: reference is shap true_value
        for cls in self.backends:
            if cls.library == "shap" and cls.computation_type == "true_value":
                return cls.name
        return None

    def _append_to_csv(self, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        write_header = not Path(self.output_csv).exists()
        df.to_csv(self.output_csv, mode="a", header=write_header, index=False)
