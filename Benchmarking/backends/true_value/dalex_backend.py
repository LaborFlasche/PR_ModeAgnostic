import dalex
import pandas as pd

from ..base_backend import BaseBackend, marginal_predict


class DalexTrueBackend(BaseBackend):
    """dalex's permutation-sampling SHAP (``predict_parts(type="shap")``), run as a true-value backend.

    because dalex does not use determistic sampling so duplicates can occur. That's why dalex is not able to perform true value shapley values.
    We try to get as close as possible to the true value by setting B = 2^n_features. This is not feasible for n_features > 14, so we throw a warning in that case.

    config:
        seed: random_state for dalex.
        N: number of background samples for dalex (default 100).
    """

    name = "dalex_true_value"
    library = "dalex"
    computation_type = "true_value"

    _EXACT_MAX_FEATURES = 14

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("DalexTrueBackend requires a 'seed' in the config.")
        return {
            "random_state": seed, # seed for dalex
            "N": self.config.get("n_background", 100), # number of background samples for dalex
        }

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        config = self.load_config()

        columns = list(x.columns)
        n_features = x.shape[1]
        if n_features > self._EXACT_MAX_FEATURES:
            print(
                f"  [WARN] {self.name}: n_features={n_features} > "
                f"{self._EXACT_MAX_FEATURES} — B=2**{n_features} orderings is "
                "infeasible; results will not be close to the true value."
            )
        # Dalex does not allow true value computation -> we try to get as close as possible by setting B = 2^n_features
        B = 2**n_features

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
            pp = explainer.predict_parts(obs, type="shap", B=B, **config)
            agg = pp.result[pp.result["B"] == 0]
            contrib = agg.set_index("variable_name")["contribution"].reindex(columns)
            rows.append(contrib.to_numpy(dtype=float))

        return pd.DataFrame(rows, index=x.index, columns=columns)
