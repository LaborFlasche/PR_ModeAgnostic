"""
SHAP backend — mirrors Shapash's ShapBackend with minimal adaptation.

Shapash source: shapash/backend/shap_backend.py
Key difference: we drop Shapash's preprocessing inversion (not needed for benchmarking).
"""

import numpy as np
import pandas as pd
import shap

from .base_backend import BaseBackend


class ShapBackend(BaseBackend):
    """Attribution backend using the SHAP library.

    Automatically selects the best SHAP explainer type for the model:
    - Tree models  → shap.TreeExplainer  (fast, exact)
    - Linear models → shap.LinearExplainer
    - Any other model → shap.KernelExplainer / shap.Explainer (slower)

    This mirrors Shapash's auto-detection logic in ShapBackend.__init__.
    """

    name = "shap"

    def __init__(self, model, masker: pd.DataFrame | None = None, **explainer_kwargs):
        """
        Parameters
        ----------
        model :
            Trained sklearn-compatible model.
        masker : pd.DataFrame, optional
            Background dataset used by linear/kernel explainers.
            Pass X_train or a representative sample of it.
        **explainer_kwargs :
            Forwarded to shap.Explainer if you need fine-grained control.
        """
        super().__init__(model)
        self.masker = masker
        self.explainer_kwargs = explainer_kwargs
        self.explainer = None  # built lazily on first call

    def _build_explainer(self, x: pd.DataFrame) -> None:
        background = self.masker if self.masker is not None else x
        if self.explainer_kwargs:
            self.explainer = shap.Explainer(model=self.model, **self.explainer_kwargs)
        elif shap.explainers.Tree.supports_model_with_masker(self.model, None):
            self.explainer = shap.Explainer(model=self.model)
        elif shap.explainers.Linear.supports_model_with_masker(self.model, background):
            self.explainer = shap.Explainer(model=self.model, masker=background)
        elif hasattr(self.model, "predict_proba"):
            self.explainer = shap.Explainer(model=self.model.predict_proba, masker=background)
        else:
            self.explainer = shap.Explainer(model=self.model.predict, masker=background)

    def run_explainer(self, x: pd.DataFrame) -> dict:
        if self.explainer is None:
            self._build_explainer(x)

        sv = self.explainer(x)
        values = sv.values  # shape: (n_samples, n_features) or (n_samples, n_features, n_classes)

        if values.ndim == 3:
            # Multi-class: return list of DataFrames, one per class
            contributions = [
                pd.DataFrame(values[:, :, k], index=x.index, columns=x.columns)
                for k in range(values.shape[2])
            ]
        else:
            contributions = pd.DataFrame(values, index=x.index, columns=x.columns)

        return {"contributions": contributions, "explainer": self.explainer}
