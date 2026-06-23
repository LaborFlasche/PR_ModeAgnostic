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
        self.config = config or {}

    @abstractmethod
    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return contributions as a DataFrame of shape (n_samples, n_features)."""


def marginal_predict(model, columns):
    """Scalar value function in each model's *natural additive* output space.

    The exact oracle (``ShapTrueValueBackend``) uses shap's model-specific explainer,
    which attributes whatever output space that model is additive in:

    * regressors -> the raw ``predict`` value;
    * models with ``decision_function`` (gradient boosting, linear classifiers) -> the
      **log-odds margin**, the only space where TreeSHAP/LinearSHAP are *exact* for
      them (in probability space they are not, and there is no polynomial exact method
      at high feature counts);
    * the remaining classifiers (RandomForest / DecisionTree, whose leaves already
      store class probabilities) -> ``predict_proba``.

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
        if hasattr(model, "decision_function"):
            out = np.asarray(model.decision_function(df), dtype=float)
            return out if out.ndim == 1 else out[:, 0]
        if hasattr(model, "predict_proba"):
            out = np.asarray(model.predict_proba(df))
            return out[:, 1] if out.shape[1] == 2 else out[:, 0]
        return np.asarray(model.predict(df), dtype=float)

    return f
