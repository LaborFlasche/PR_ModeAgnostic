import numpy as np
import pandas as pd
from lightshap import explain_any

from .base_backend import BaseBackend, marginal_predict


class LightShapExactBackend(BaseBackend):
    """
    LightShap's exact SHAP on the marginal value function.
    """

    name = "lightshap_exact"
    library = "lightshap"
    computation_type = "true_value"

    # To keep everything computational we limit num of features for exact backend to 14 -> we throw a warning
    _EXACT_MAX_FEATURES = 14

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("LightShapExactBackend requires a 'seed' in the config.")
        return {
            "random_state": seed, # seed for lightshap
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

class LightShapApproxBackend(BaseBackend):
    """lightshap's Kernel / Permutation SHAP on the marginal value function.

    config:
        approximator: "kernel" or "permutation" (lightshap's ``method``)
        budget: maximum number of sampling iterations (lightshap's ``max_iter``).
            One iteration is a forward+backward pass over a random permutation, so
            this is not directly comparable to other libraries' budgets — rely on
            the measured model-evaluation count for cross-library comparison.

    ``how="sampling"`` forces the iterative approximation path (lightshap would
    otherwise compute exact values for small feature counts). lightshap's permutation
    sampling requires at least 4 features, which the config's ``n_features`` floor
    guarantees.
    """

    name = "lightshap_approx"
    library = "lightshap"
    computation_type = "approximation"
    SUPPORTED_APPROXIMATORS = ("kernel", "permutation")

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        approximator = self.config.get("approximator", "permutation")
        budget = self.config.get("budget")
        seed = self.config.get("seed")
        imputer = self.config.get("imputer", "marginal")
        if approximator not in ("kernel", "permutation"):
            raise ValueError(f"Unknown lightshap approximator '{approximator}' (use 'kernel' or 'permutation')")
        if imputer != "marginal":
            raise ValueError(
                f"lightshap backend only supports imputer='marginal' (got {imputer!r}): "
                "explain_any imputes masked features from bg_X (the marginal value function)."
            )

        f = marginal_predict(self.model, x.columns)
        kwargs = {}
        if budget is not None:
            kwargs["max_iter"] = budget

        explanation = explain_any(
            f,
            x,
            bg_X=self.background,
            method=approximator,
            how="sampling",
            random_state=seed,
            verbose=False,
            **kwargs,
        )
        values = np.asarray(explanation.shap_values)
        return pd.DataFrame(values, index=x.index, columns=x.columns)
