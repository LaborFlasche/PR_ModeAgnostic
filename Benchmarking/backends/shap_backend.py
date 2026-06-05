import numpy as np
import pandas as pd
import shap

from .base_backend import BaseBackend, marginal_predict


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


class ShapApproxBackend(BaseBackend):
    """Model-agnostic KernelSHAP / Permutation SHAP on the marginal value function.

    config:
        approximator: "kernel" (KernelExplainer) or "permutation" (PermutationExplainer)
        budget: nominal evaluation budget (nsamples / max_evals). The real cost is
            recorded separately via the model evaluation counter, so the budget is
            only a nominal request — it is clamped up to each algorithm's minimum to
            avoid crashes on high-dimensional inputs.
    """

    name = "shap_approx"
    library = "shap"
    computation_type = "approximation"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        approximator = self.config.get("approximator", "permutation")
        budget = self.config.get("budget")
        f = marginal_predict(self.model, x.columns)
        n_features = x.shape[1]

        if approximator == "kernel":
            explainer = shap.KernelExplainer(f, self.background, silent=True)
            nsamples = budget if budget is not None else "auto"
            values = np.asarray(explainer.shap_values(x, nsamples=nsamples, silent=True))
        elif approximator == "permutation":
            masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
            explainer = shap.PermutationExplainer(f, masker, silent=True)
            # PermutationExplainer needs at least 2*n_features+1 evals per instance.
            max_evals = max(budget, 2 * n_features + 1) if budget is not None else "auto"
            values = np.asarray(explainer(x, max_evals=max_evals, silent=True).values)
        else:
            raise ValueError(f"Unknown shap approximator '{approximator}' (use 'kernel' or 'permutation')")

        return pd.DataFrame(values, index=x.index, columns=x.columns)
