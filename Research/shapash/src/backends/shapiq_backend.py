"""
ShapIQ backend.

ShapIQ computes Shapley Interaction Values but also standard Shapley Values (SV).
We use SV (order=1) here so the output is directly comparable to SHAP.

Install: pip install shapiq
Docs:    https://shapiq.readthedocs.io
"""

import numpy as np
import pandas as pd

from .base_backend import BaseBackend

try:
    import shapiq
    _SHAPIQ_AVAILABLE = True
except ImportError:
    _SHAPIQ_AVAILABLE = False


class ShapIQBackend(BaseBackend):
    """Attribution backend using the ShapIQ library.

    Uses ShapIQ's TabularExplainer with index='SV' (standard Shapley Values)
    so results are comparable to SHAP output.

    Supports any sklearn-compatible model via a predict wrapper.
    """

    name = "shapiq"

    def __init__(self, model, data: pd.DataFrame | None = None, index: str = "SV", **explainer_kwargs):
        """
        Parameters
        ----------
        model :
            Trained sklearn-compatible model.
        data : pd.DataFrame, optional
            Background dataset used by the explainer.
            Pass X_train or a representative sample.  Falls back to X_test if None.
        index : str, default 'SV'
            ShapIQ interaction index.  'SV' = standard Shapley Values.
            Others: 'k-SII', 'STII', 'FSI' (higher-order interactions).
        **explainer_kwargs :
            Forwarded to shapiq.TabularExplainer.
        """
        if not _SHAPIQ_AVAILABLE:
            raise ImportError("shapiq is not installed. Run: pip install shapiq")
        super().__init__(model)
        self.data = data
        self.index = index
        self.explainer_kwargs = explainer_kwargs

    def _make_predict_fn(self, x: pd.DataFrame):
        """Wrap the sklearn model so ShapIQ gets a callable that accepts numpy arrays."""
        columns = x.columns

        def predict_fn(X: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(X, columns=columns)
            if hasattr(self.model, "predict_proba"):
                out = self.model.predict_proba(df)
                # Binary: return P(class=1); multi-class: return full matrix
                return out[:, 1] if out.shape[1] == 2 else out
            return self.model.predict(df).astype(float)

        return predict_fn

    def run_explainer(self, x: pd.DataFrame) -> dict:
        background = self.data if self.data is not None else x
        predict_fn = self._make_predict_fn(x)

        explainer = shapiq.TabularExplainer(
            model=predict_fn,
            data=background.values.astype(float),
            index=self.index,
            max_order=1,
            **self.explainer_kwargs,
        )

        rows = []
        for i in range(len(x)):
            iv = explainer.explain(x.iloc[i].values.astype(float))
            # InteractionValues stores SV at order 1.
            # shapiq >= 0.1: iv.get_n_order_values(1) returns a 1-D array indexed by feature.
            sv = iv.get_n_order_values(1)
            rows.append(sv)

        contributions = pd.DataFrame(rows, index=x.index, columns=x.columns)
        return {"contributions": contributions, "explainer": explainer}
