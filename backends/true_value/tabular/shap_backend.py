import numpy as np
import pandas as pd
import shap

from ...base_backend import BaseBackend, marginal_predict


class ShapTrueValueBackend(BaseBackend):
    name = "shap_true_value"
    library = "shap"
    computation_type = "true_value"

    _EXACT_MAX_FEATURES = 14

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("ShapTrueValueBackend requires a 'seed' in the config.")
        return {
            "random_state": seed,
            "n_background": self.config.get("n_background", 100),
        }


    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        config = self.load_config()
        n_features = x.shape[1]
        if n_features > self._EXACT_MAX_FEATURES:
            print(
                f"  [WARN] {self.name}: n_features={n_features} > "
                f"{self._EXACT_MAX_FEATURES} — shap.explainers.Exact enumerates all "
                f"2^{n_features} coalitions with no budget cap; this may be "
                "extremely slow or infeasible."
            )

        # Because shap.Exact does not take a seed argument, we seed the global RNG for reproducibility.
        np.random.seed(config["random_state"])
        f = marginal_predict(self.model, x.columns)
        # shap.Exact needs a masker, but it's never actually invoked here because
        # max_samples == len(self.background), so every row is used directly.
        explainer = shap.explainers.Exact(f, shap.maskers.Independent(self.background, max_samples=len(self.background)))
        sv = explainer(x)
        values = sv.values
        if values.ndim == 3:
            # multi-class: use class 1 for binary, class 0 otherwise
            values = values[:, :, 1] if values.shape[2] == 2 else values[:, :, 0]
        return pd.DataFrame(values, index=x.index, columns=x.columns)
