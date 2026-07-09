import numpy as np
import pandas as pd
import shap

from ..base_backend import BaseBackend, marginal_predict


class ShapApproxBackend(BaseBackend):
    """Model-agnostic KernelSHAP / Permutation SHAP on the marginal value function.

    config:
        approximator: "kernel" (KernelExplainer) or "permutation" (PermutationExplainer)
        budget: nominal evaluation budget (nsamples / max_evals).
    """

    name = "shap_approx"
    library = "shap"
    computation_type = "approximation"
    SUPPORTED_APPROXIMATORS = ("kernel", "permutation")

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("ShapApproxBackend requires a 'seed' in the config.")
        assert self.config.get("approximator", "permutation") in self.SUPPORTED_APPROXIMATORS, \
            f"shap approximator must be one of {self.SUPPORTED_APPROXIMATORS} (got {self.config.get('approximator')!r})"
        
        return {
            "random_state": seed, # seed for shap,
            "approximator": self.config.get("approximator", "permutation"), # shap approximator to use
            "budget": self.config.get("budget"),
        }
        

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        config = self.load_config()

        f = marginal_predict(self.model, x.columns)

        np.random.seed(config["random_state"])

        if config["approximator"] == "kernel":
            explainer = shap.KernelExplainer(f, self.background, silent=True, seed=config["random_state"])
            values = np.asarray(explainer.shap_values(x, nsamples=config["budget"], silent=True))
        elif config["approximator"] == "permutation":
            masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
            explainer = shap.PermutationExplainer(f, masker, seed=config["random_state"], silent=True)
            # PermutationExplainer needs at least 2*n_features+1 evals per instance;
            # if max_evals is less than that i will throw an error resulting in NaN values
            values = np.asarray(explainer(x, max_evals=config["budget"] , l1_reg=False, silent=True).values)
        else:
            raise ValueError(f"Unknown shap approximator '{config['approximator']}' (use 'kernel' or 'permutation')")

        return pd.DataFrame(values, index=x.index, columns=x.columns)
