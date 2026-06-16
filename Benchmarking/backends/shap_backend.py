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
        # check_additivity=False: interventional TreeSHAP on HistGradientBoosting
        # (the gradient_boosting model) trips shap's additivity check — its binned
        # histogram trees make shap's traversal sum mismatch the raw model output by
        # a small margin. Only TreeExplainer runs this check; LinearExplainer and the
        # exact path don't accept the kwarg, so fall back to the plain call there. The
        # oracle-validation cell independently confirms the values are correct.
        try:
            sv = explainer(x, check_additivity=False)
        except TypeError:
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
        seed = self.config.get("seed")
        imputer = self.config.get("imputer", "marginal")
        if imputer != "marginal":
            raise ValueError(
                f"shap backend only supports imputer='marginal' (got {imputer!r}): both the "
                "Independent masker and KernelExplainer integrate over the background = the "
                "marginal value function; no conditional masker is wired."
            )
        f = marginal_predict(self.model, x.columns)
        n_features = x.shape[1]

        if approximator == "kernel":
            # KernelExplainer integrates over the full background set = the marginal
            # imputer (the shared value function). It exposes no seed argument, so seed
            # the global RNG it samples coalitions from for reproducibility.
            if seed is not None:
                np.random.seed(seed)
            explainer = shap.KernelExplainer(f, self.background, silent=True)
            nsamples = budget if budget is not None else "auto"
            values = np.asarray(explainer.shap_values(x, nsamples=nsamples, silent=True))
        elif approximator == "permutation":
            # Independent masker over the full background = marginal imputer, matching
            # the other libraries' value function. seed makes the sampled permutations
            # reproducible.
            masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
            explainer = shap.PermutationExplainer(f, masker, seed=seed, silent=True)
            # PermutationExplainer needs at least 2*n_features+1 evals per instance.
            max_evals = max(budget, 2 * n_features + 1) if budget is not None else "auto"
            values = np.asarray(explainer(x, max_evals=max_evals, silent=True).values)
        else:
            raise ValueError(f"Unknown shap approximator '{approximator}' (use 'kernel' or 'permutation')")

        return pd.DataFrame(values, index=x.index, columns=x.columns)
