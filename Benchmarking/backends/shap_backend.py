import numpy as np
import pandas as pd
import shap

from .base_backend import BaseBackend, marginal_predict


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
            "random_state": seed, # seed for shap,
            "n_background": self.config.get("n_background", 100), # number of background samples for shap
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
        # Shap exact explainer needs a masker but this masker is never called because len(self.backgound ) = max_samples
        explainer = shap.explainers.Exact(f, shap.maskers.Independent(self.background, max_samples=len(self.background)))
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
    SUPPORTED_APPROXIMATORS = ("kernel", "permutation")

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
            # KernelExplainer exposes no seed argument, so seed
            # the global RNG it samples coalitions from for reproducibility.
            if seed is not None:
                np.random.seed(seed)
            explainer = shap.KernelExplainer(f, self.background, silent=True)
            nsamples = budget if budget is not None else "auto"
            values = np.asarray(explainer.shap_values(x, nsamples=nsamples, l1_reg=False, silent=True))
        elif approximator == "permutation":
            masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
            explainer = shap.PermutationExplainer(f, masker, seed=seed, silent=True)
            # PermutationExplainer needs at least 2*n_features+1 evals per instance.
            max_evals = max(budget, 2 * n_features + 1) if budget is not None else "auto"
            values = np.asarray(explainer(x, max_evals=max_evals, silent=True).values)
        else:
            raise ValueError(f"Unknown shap approximator '{approximator}' (use 'kernel' or 'permutation')")

        return pd.DataFrame(values, index=x.index, columns=x.columns)
