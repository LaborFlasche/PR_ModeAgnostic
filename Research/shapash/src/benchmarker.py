"""
Benchmarker — runs multiple attribution backends on the same data and
collects contributions, global importance, runtimes, and comparison metrics.

Usage
-----
    from src.benchmarker import Benchmarker
    from src.backends import ShapBackend, ShapIQBackend

    bench = Benchmarker(
        model=clf,
        backends=[ShapBackend(clf, masker=X_train), ShapIQBackend(clf, data=X_train)],
    )
    results = bench.run(X_test)
    print(results.summary())
    print(results.pairwise_metrics())
"""

import time
from dataclasses import dataclass, field

import pandas as pd

from .backends.base_backend import BaseBackend
from .metrics import pairwise_summary


@dataclass
class BenchmarkResult:
    """Container for all outputs of a single Benchmarker.run() call."""

    # Per-backend outputs
    contributions: dict[str, pd.DataFrame] = field(default_factory=dict)
    importances: dict[str, pd.Series] = field(default_factory=dict)
    runtimes: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """One-row-per-backend summary: runtime and top-5 important features."""
        rows = []
        for name in self.contributions:
            imp = self.importances[name]
            rows.append(
                {
                    "backend": name,
                    "runtime_s": round(self.runtimes[name], 3),
                    **{f"rank_{i+1}": feat for i, feat in enumerate(imp.index[:5])},
                }
            )
        return pd.DataFrame(rows).set_index("backend")

    def pairwise_metrics(self, top_k: int = 5) -> pd.DataFrame:
        """Pairwise comparison metrics across all backend pairs."""
        return pairwise_summary(self.contributions, self.importances, top_k=top_k)

    def contributions_df(self) -> pd.DataFrame:
        """All contributions concatenated into one DataFrame with a 'backend' level."""
        return pd.concat(self.contributions, axis=0, names=["backend"])


class Benchmarker:
    """Runs a list of attribution backends on the same dataset and compares results.

    Parameters
    ----------
    model :
        Trained model (sklearn-compatible or torch.nn.Module for Captum).
    backends : list[BaseBackend]
        Pre-instantiated backend objects.  Each backend's `.name` attribute is used
        as the key in result dicts — give backends unique names if you want multiple
        instances of the same library (e.g. two SHAP variants).
    """

    def __init__(self, model, backends: list[BaseBackend]):
        self.model = model
        self.backends = backends
        self._check_unique_names()

    def _check_unique_names(self) -> None:
        names = [b.name for b in self.backends]
        if len(names) != len(set(names)):
            raise ValueError(
                "All backends must have unique `.name` values.  "
                "Subclass and override `name` to distinguish two instances of the same library."
            )

    def run(self, x: pd.DataFrame) -> BenchmarkResult:
        """Run all backends on `x` and return a BenchmarkResult.

        Parameters
        ----------
        x : pd.DataFrame
            Input samples, shape (n_samples, n_features).

        Returns
        -------
        BenchmarkResult
        """
        result = BenchmarkResult()

        for backend in self.backends:
            name = backend.name
            print(f"[Benchmarker] Running '{name}' ...")

            t0 = time.perf_counter()
            contrib = backend.get_contributions(x)
            elapsed = time.perf_counter() - t0

            # For multi-class (list of DataFrames) take class-1 for scalar comparison.
            # The full list is preserved in contributions_raw if needed.
            if isinstance(contrib, list):
                contrib_scalar = contrib[1] if len(contrib) == 2 else contrib[0]
            else:
                contrib_scalar = contrib

            result.contributions[name] = contrib_scalar
            result.importances[name] = backend.get_global_importance(contrib_scalar)
            result.runtimes[name] = elapsed

            print(f"[Benchmarker]   done in {elapsed:.2f}s")

        return result
