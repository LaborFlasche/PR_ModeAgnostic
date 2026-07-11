import numpy as np
import pandas as pd
from lightshap import explain_any

from ...base_backend import BaseBackend, marginal_predict


class LightShapExactBackend(BaseBackend):
    """LightShap's exact SHAP on the marginal value function."""

    name = "lightshap_exact"
    library = "lightshap"
    computation_type = "true_value"

    _EXACT_MAX_FEATURES = 14

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("LightShapExactBackend requires a 'seed' in the config.")
        return {
            "random_state": seed,
        }

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        config = self.load_config()
        n_features = x.shape[1]
        if n_features > self._EXACT_MAX_FEATURES:
            print(
                f"  [WARN] {self.name}: n_features={n_features} > "
                f"{self._EXACT_MAX_FEATURES} — how='exact' enumerates all "
                f"2^{n_features} coalitions with no budget cap; this may be "
                "extremely slow or infeasible."
            )
        f = marginal_predict(self.model, x.columns)

        explanation = explain_any(
            f,
            x,
            bg_X=self.background,
            how="exact",
            verbose=False,
            **config,
        )
        values = np.asarray(explanation.shap_values)
        return pd.DataFrame(values, index=x.index, columns=x.columns)
