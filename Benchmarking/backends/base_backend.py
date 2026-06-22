from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd


class BaseBackend(ABC):
    name: str
    library: str
    computation_type: Literal["true_value", "approximation"]
    # 1 = first-order Shapley values, 2 = pairwise interactions. Purely informational
    # for callers (e.g. slurm/run_benchmark.py picks which map/oracle to use) —
    # runner.py itself is agnostic to this; it just compares same-shaped DataFrames.
    order: int = 1

    def __init__(self, model, background: pd.DataFrame, config: dict | None = None):
        self.model = model
        self.background = background
        self.config = config or {}

    @abstractmethod
    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return contributions as a DataFrame of shape (n_samples, n_features)."""


def marginal_predict(model, columns):
    """Scalar value function in each model's *natural additive* output space.

    The exact oracle (``ShapTrueValueBackend``) uses shap's model-specific explainer,
    which attributes whatever output space that model is additive in:

    * regressors -> the raw ``predict`` value;
    * ``XGBClassifier`` -> the raw margin via ``predict(output_margin=True)``. Unlike
      sklearn/LightGBM classifiers it has no ``decision_function``, so it would
      otherwise fall through to ``predict_proba`` below — but shap's TreeExplainer
      defaults XGBoost to margin space (see ``_model_raw_output`` in TreeSHAPBench's
      benchmark_utils.py), so this must be checked first. Detected via the model's
      module name rather than ``isinstance(model, xgboost.XGBClassifier)`` to avoid
      importing xgboost here: doing so unconditionally loads xgboost's native runtime
      into every process before shapiq, which segfaults shapiq's interventional
      TreeExplainer even on unrelated (non-XGBoost) models — see
      ``ShapIQTreeInterventionalBackend`` in tree_shapiq_backend.py for the same
      avoidance and a fuller explanation;
    * models with ``decision_function`` (gradient boosting, linear classifiers) -> the
      **log-odds margin**, the only space where TreeSHAP/LinearSHAP are *exact* for
      them (in probability space they are not, and there is no polynomial exact method
      at high feature counts);
    * the remaining classifiers (RandomForest / DecisionTree / LightGBM, whose leaves
      already store class probabilities) -> ``predict_proba``.

    Every approximation backend calls this, so within each (model, dataset) cell it
    targets the *same* function as the oracle — the only thing the accuracy metrics
    compare. (Targeting ``predict_proba`` for boosting/linear classifiers would chase a
    different function than the margin-space oracle, making those rows meaningless.)
    Binary -> class 1, multiclass -> class 0, mirroring the oracle's reduction. Inputs
    are wrapped back into a DataFrame with the original column names to avoid sklearn's
    "missing feature names" warning and keep encodings aligned.
    """

    def f(X) -> np.ndarray:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=columns)
        if type(model).__module__.startswith("xgboost") and hasattr(model, "predict_proba"):
            out = np.asarray(model.predict(df, output_margin=True), dtype=float)
            return out if out.ndim == 1 else out[:, 0]
        if hasattr(model, "decision_function"):
            out = np.asarray(model.decision_function(df), dtype=float)
            return out if out.ndim == 1 else out[:, 0]
        if hasattr(model, "predict_proba"):
            out = np.asarray(model.predict_proba(df))
            return out[:, 1] if out.shape[1] == 2 else out[:, 0]
        return np.asarray(model.predict(df), dtype=float)

    return f


def nan_result(x: pd.DataFrame) -> pd.DataFrame:
    """All-NaN contribution frame for a backend that could not run on this cell.

    Used by tree backends with known model-compatibility gaps (e.g. woodelf's
    multiclass restriction, fasttreeshap's XGBoost-3.x incompatibility) so one
    unsupported (model, dataset) combination logs a skip instead of crashing the
    whole sweep.
    """
    return pd.DataFrame(np.nan, index=x.index, columns=x.columns)


def reduce_multiclass(values: np.ndarray | list, order: int = 1) -> np.ndarray:
    """Normalize a tree-explainer's raw output to a single (n_samples, ...) array.

    Mirrors ``to_sv_array`` in TreeSHAPBench's benchmark_utils.py: some libraries
    return a list of K per-class arrays for classifiers (older shap, woodelf), others
    a trailing-class-axis array — (n, d, K) for first-order (shap>=0.44+), (n, d, d, K)
    for interactions. Binary -> class 1, multiclass -> class 0, matching
    ``ShapTrueValueBackend``'s reduction so accuracy metrics compare the same quantity.

    ``order`` (1 for Shapley values, 2 for pairwise interactions) disambiguates a
    trailing class axis from a same-sized feature axis: a *regression* interaction
    array is already (n, d, d) — ndim 3, same as a *multiclass* first-order array —
    so ndim alone can't tell them apart; only "ndim beyond order+1 dims means a
    class axis is present" can.
    """
    if isinstance(values, list):
        idx = 1 if len(values) == 2 else 0
        return np.asarray(values[idx])
    arr = np.asarray(values)
    if arr.ndim > order + 1:
        return arr[..., 1] if arr.shape[-1] == 2 else arr[..., 0]
    return arr


def flatten_interactions(values: np.ndarray, x: pd.DataFrame) -> pd.DataFrame:
    """Flatten a (n_samples, n_features, n_features) interaction array into a
    (n_samples, n_features**2) DataFrame with paired column names (e.g. "f0__f1").

    Lets pairwise-interaction backends reuse runner.py/metrics.py completely
    unchanged: every metric there (mean_abs_diff, sign_agreement, ...) operates
    elementwise/row-wise on whatever-shaped DataFrame a backend returns and never
    assumes the column count equals n_features.
    """
    n, d, _ = values.shape
    cols = [f"{a}__{b}" for a in x.columns for b in x.columns]
    return pd.DataFrame(values.reshape(n, d * d), index=x.index, columns=cols)


def nan_interaction_result(x: pd.DataFrame) -> pd.DataFrame:
    """``nan_result`` analog for interaction backends — matches flatten_interactions'
    column shape (n_features**2), not x's own n_features columns."""
    cols = [f"{a}__{b}" for a in x.columns for b in x.columns]
    return pd.DataFrame(np.nan, index=x.index, columns=cols)


def cuda_available() -> bool:
    """Single source of truth for "is there a usable CUDA device in this process".

    xgboost's own device="cuda" does *not* raise when no GPU is present — it
    silently warns and falls back to CPU (confirmed live) — so GPU-gated backends
    must check this explicitly rather than relying on a library's own fallback.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
