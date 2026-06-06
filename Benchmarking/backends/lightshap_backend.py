import pandas as pd

from .base_backend import BaseBackend
from lightshap import explain_any


class LightShapValueBackend(BaseBackend):
    name = "light_shap_value"
    library = "LightShap"
    computation_type = "approximation"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explanation = explain_any(self.model.predict, x, bg_X=self.background)
        return pd.DataFrame(explanation.shap_values, columns=x.columns, index=x.index)

