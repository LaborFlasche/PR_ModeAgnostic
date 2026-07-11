import dalex
import pandas as pd

from ...base_backend import BaseBackend, marginal_predict


class DalexApproxBackend(BaseBackend):
    """dalex's SHAP (Štrumbelj–Kononenko marginal sampling) on the shared value function.

    ``predict_parts(type="shap")`` averages a feature's contribution over ``B``
    random orderings, imputing removed features from the background — the same
    value function as the other backends. dalex has no kernel-weighting
    analogue, so only the **permutation** approximator slot is supported.

    dalex returns variables sorted by ``|contribution|`` with the averaged
    values at ``B == 0``; both are undone here (filter, reindex).

    config:
        approximator: must be ``"permutation"`` (the only method dalex provides).
        budget: passed directly as dalex's ``B`` (number of random orderings).
    """

    name = "dalex_approx"
    library = "dalex"
    computation_type = "approximation"
    SUPPORTED_APPROXIMATORS = ("permutation",)

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("DalexApproxBackend requires a 'seed' in the config.")
        assert self.config.get("approximator", "permutation") in self.SUPPORTED_APPROXIMATORS, \
            f"approximator must be one of {self.SUPPORTED_APPROXIMATORS} (got {self.config.get('approximator')!r})"

        return {
            "random_state": seed,
            "approximator": self.config.get("approximator", "permutation"),
            "budget": self.config.get("budget"),
        }

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        config = self.load_config()
        columns = list(x.columns)
        n_features = x.shape[1]

        B = max(1, round(config["budget"]/n_features))

        f = marginal_predict(self.model, columns)

        explainer = dalex.Explainer(
            self.model,
            self.background,
            y=None,
            predict_function=lambda m, d: f(d),
            verbose=False,
        )

        rows = []
        for i in range(len(x)):
            obs = x.iloc[i:i + 1]
            pp = explainer.predict_parts(obs, type="shap", B=B, random_state=config["random_state"])
            agg = pp.result[pp.result["B"] == 0]
            contrib = agg.set_index("variable_name")["contribution"].reindex(columns)
            rows.append(contrib.to_numpy(dtype=float))

        return pd.DataFrame(rows, index=x.index, columns=columns)
