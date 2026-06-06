import pandas as pd
import shap

from .base_backend import BaseBackend


class ShapModeAgnosticBackend(BaseBackend):
    name = "shap_mode_agnostic"
    library = "shap"
    computation_type = "approximation"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.Explainer(self.model)
        self.chosen_method = type(explainer).__name__
        sv = explainer(x)
        return pd.DataFrame(sv.values, index=x.index, columns=x.columns)
