import pandas as pd
import shap

from .base_backend import BaseBackend


class ShapTrueValueBackend(BaseBackend):
    name = "shap_true_value"
    library = "shap"
    computation_type = "true_value"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.Explainer(self.model, self.background)
        sv = explainer(x)
        values = sv.values
        if values.ndim == 3:
            # multi-class: use class 1 for binary, class 0 otherwise
            values = values[:, :, 1] if values.shape[2] == 2 else values[:, :, 0]
        return pd.DataFrame(values, index=x.index, columns=x.columns)
