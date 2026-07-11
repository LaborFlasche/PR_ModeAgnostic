import numpy as np
import pandas as pd
from lightshap import explain_any

from ..base_backend import BaseBackend, marginal_predict


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

    ``tol=0`` disables lightshap's own convergence check (default ``tol=0.01``
    stops iterating once estimated standard errors are small enough, independent
    of ``max_iter``), so ``budget`` is the only thing controlling how much work is
    done — otherwise most cells converge early and different budgets in the sweep
    produce identical results.
    """

    name = "lightshap_approx"
    library = "lightshap"
    computation_type = "approximation"
    SUPPORTED_APPROXIMATORS = ("kernel", "permutation")

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("LightShapApproxBackend requires a 'seed' in the config.")
        assert self.config.get("approximator", "permutation") in self.SUPPORTED_APPROXIMATORS, \
            f"shap approximator must be one of {self.SUPPORTED_APPROXIMATORS} (got {self.config.get('approximator')!r})"
        
        return {
            "random_state": seed, # seed for shapiq,
            "method": self.config.get("approximator", "permutation"), # shap approximator to use
            "max_iter": self.config.get("budget"),
            "tol": 0, # disable early stopping so max_iter/budget is the real cap
        }

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        config = self.load_config()
        n_features = x.shape[1]
        # LightShap sampling requires at least 4 features, so we enforce that here
        if n_features < 4:
            raise ValueError(f"LightShapApproxBackend requires at least 4 features (got {n_features})")

        f = marginal_predict(self.model, x.columns)

        explanation = explain_any(
            f,
            x,
            bg_X=self.background,
            how="sampling",
            verbose=False,
            **config,
        )
        values = np.asarray(explanation.shap_values)
        return pd.DataFrame(values, index=x.index, columns=x.columns)
