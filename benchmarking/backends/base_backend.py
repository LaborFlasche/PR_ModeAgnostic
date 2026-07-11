from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd


class BaseBackend(ABC):
    name: str
    library: str
    computation_type: Literal["true_value", "approximation"]
    order: int = 1  # 1 = Shapley values, 2 = pairwise interactions

    def __init__(self, model, background: pd.DataFrame, config: dict | None = None):
        self.model = model
        self.background = background
        self.config = config or {}
        # Base value (empty-coalition value) of the game THIS backend explains,
        # set by run_explainer when the library reports one. Path-dependent
        # tree backends explain a different game than the marginal one (their
        # base value is the cover-weighted training expectation, not the mean
        # prediction over the background), so the runner must check additivity
        # against this, not the shared background baseline. None = the backend
        # explains the marginal game over the runner's background; the runner
        # falls back to mean f(background), which is that game's base value.
        self.baseline_: float | None = None

    @abstractmethod
    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return contributions as a DataFrame of shape (n_samples, n_features)."""

    def load_config(self) -> dict:
        """Config dict with backend-specific defaults filled in; backends that
        take tuning parameters override this. Not abstract: tree backends have
        no tunables and don't need it."""
        return self.config


def marginal_predict(model, columns):
    """Build the scalar value function every backend explains the same game on.

    @model: fitted model to wrap.
    @columns: feature column names, used to rebuild a DataFrame from raw arrays.

    XGBoost/LightGBM are detected via module name, not isinstance, to avoid
    importing them here — importing xgboost before shapiq segfaults shapiq's
    interventional TreeExplainer later in the process (see trees/shapiq_backend.py).
    LGBMClassifier needs the margin branch just like XGBClassifier: its leaves
    store log-odds, so the shap oracle (and every tree backend) explains the
    margin — but it has no decision_function, so without raw_score=True it would
    fall through to predict_proba and the additivity metrics would compare
    margin-space Shapley sums against probability-space predictions.

    The module-name check is done on ``model._model`` when present (e.g.
    ``CountingModel``, whose whole point is to proxy calls through so they get
    counted) because ``type(proxy).__module__`` reflects the proxy's own class,
    not the wrapped model's — checking the proxy directly would always miss the
    xgboost/lightgbm branches and silently fall through to predict_proba.
    """

    def f(X) -> np.ndarray:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=columns)
        real_model = getattr(model, "_model", model)
        if type(real_model).__module__.startswith("xgboost") and hasattr(model, "predict_proba"):
            # XGBClassifier -> raw margin (pre-sigmoid), not probability.
            out = np.asarray(model.predict(df, output_margin=True), dtype=float)
            return out if out.ndim == 1 else out[:, 0]
        if type(real_model).__module__.startswith("lightgbm") and hasattr(model, "predict_proba"):
            out = np.asarray(model.predict(df, raw_score=True), dtype=float)
            return out if out.ndim == 1 else out[:, 0]
        if hasattr(model, "decision_function"):
            # Other classifiers with a decision_function -> log-odds margin.
            out = np.asarray(model.decision_function(df), dtype=float)
            return out if out.ndim == 1 else out[:, 0]
        if hasattr(model, "predict_proba"):
            # Remaining classifiers -> predict_proba; class 1 if binary, else class 0.
            out = np.asarray(model.predict_proba(df))
            return out[:, 1] if out.shape[1] == 2 else out[:, 0]
        # Regressors -> predict directly.
        out = np.asarray(model.predict(df), dtype=float)
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        if out.ndim != 1:
            raise ValueError(
                f"marginal_predict requires 1D predictions; model.predict "
                f"returned shape {out.shape} (multi-output regressors are "
                "unsupported)"
            )
        return out

    return f


def select_base_value(expected_value) -> float:
    """Reduce a library-reported expected/base value to the scalar for the class
    convention every backend uses (see reduce_multiclass): scalar/regression as
    is, binary -> class 1, multiclass -> class 0."""
    arr = np.ravel(np.asarray(expected_value, dtype=float))
    if arr.size == 2:
        return float(arr[1])
    return float(arr[0])


def nan_result(x: pd.DataFrame) -> pd.DataFrame:
    """All-NaN frame for a backend that can't run on this (model, dataset) cell,
    so one unsupported combination skips instead of crashing the whole sweep."""
    return pd.DataFrame(np.nan, index=x.index, columns=x.columns)


def reduce_multiclass(values: np.ndarray | list, order: int = 1) -> np.ndarray:
    """Normalize a tree-explainer's raw output to a single (n_samples, ...) array:
    some libraries return a list of per-class arrays, others a trailing class
    axis. Binary -> class 1, multiclass -> class 0, matching ShapTrueValueBackend.

    ``order`` disambiguates a trailing class axis from a same-sized feature axis
    (a regression interaction array is already (n, d, d), same ndim as a
    multiclass first-order array) — only "ndim beyond order+1" means a class axis.
    """
    if isinstance(values, list):
        idx = 1 if len(values) == 2 else 0
        return np.asarray(values[idx])
    arr = np.asarray(values)
    if arr.ndim > order + 1:
        return arr[..., 1] if arr.shape[-1] == 2 else arr[..., 0]
    return arr


def flatten_interactions(values: np.ndarray, x: pd.DataFrame) -> pd.DataFrame:
    """Flatten a (n, d, d) interaction array into a (n, d**2) DataFrame with
    paired column names (e.g. "f0__f1"), so interaction backends reuse
    runner.py/metrics.py unchanged."""
    n, d, _ = values.shape
    cols = [f"{a}__{b}" for a in x.columns for b in x.columns]
    return pd.DataFrame(values.reshape(n, d * d), index=x.index, columns=cols)


def nan_interaction_result(x: pd.DataFrame) -> pd.DataFrame:
    """``nan_result`` for interaction backends (d**2 columns)."""
    cols = [f"{a}__{b}" for a in x.columns for b in x.columns]
    return pd.DataFrame(np.nan, index=x.index, columns=cols)


def cuda_available() -> bool:
    """xgboost's device="cuda" silently falls back to CPU instead of raising
    when no GPU is present, so GPU-gated backends check this explicitly."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
