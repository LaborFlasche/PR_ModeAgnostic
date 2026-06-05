from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd


class BaseBackend(ABC):
    name: str
    library: str
    computation_type: Literal["true_value", "approximation"]

    def __init__(self, model, background: pd.DataFrame, config: dict | None = None):
        self.model = model
        self.background = background
        # Per-run knobs (e.g. {"approximator": "kernel", "budget": 512}); empty
        # for the exact true-value backends, which take no tuning.
        self.config = config or {}

    @abstractmethod
    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return contributions as a DataFrame of shape (n_samples, n_features)."""


def marginal_predict(model, columns):
    """Scalar prediction function on the marginal value-function output space.

    Mirrors ``ShapTrueValueBackend``: explains class 1 for binary classifiers,
    class 0 for multiclass, and the raw prediction for regressors — so every
    approximation backend targets the same number as the shap exact reference.
    Wraps inputs back into a DataFrame with the original column names to avoid
    sklearn's "missing feature names" warning and keep encodings aligned.
    """

    def f(X) -> np.ndarray:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=columns)
        if hasattr(model, "predict_proba"):
            out = np.asarray(model.predict_proba(df))
            return out[:, 1] if out.shape[1] == 2 else out[:, 0]
        return np.asarray(model.predict(df), dtype=float)

    return f
