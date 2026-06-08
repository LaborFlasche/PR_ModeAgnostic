import numpy as np
import pandas as pd
import shap

from .base_backend import BaseBackend


class GpuTreeShap(BaseBackend):
    name = "gputreeshap"
    library = "shap"
    computation_type = "true_value"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.explainers.GPUTree(self.model, self.background)
        try:
            sv = explainer(x, check_additivity=False)
        except TypeError:
            try:
                sv = explainer(x)
            except TypeError:
                values = np.asarray(explainer.shap_values(x))
            else:
                values = np.asarray(getattr(sv, "values", sv))
        else:
            values = np.asarray(getattr(sv, "values", sv))

        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[2] == 2 else values[:, :, 0]
        return pd.DataFrame(values, index=x.index, columns=x.columns)
