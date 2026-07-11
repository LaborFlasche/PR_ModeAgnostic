import dalex
import pandas as pd

from ...base_backend import BaseBackend, marginal_predict


class DalexTrueBackend(BaseBackend):
    """dalex's permutation-sampling SHAP (``predict_parts(type="shap")``), run as a true-value backend.

    dalex has no exact/deterministic Shapley computation, so this is an approximation
    that gets as close as possible to the true value by spending a total predict_parts
    budget of 2^n_features model evaluations (B = 2^n_features / (n_features+1)
    orderings). Infeasible for n_features > 14 — a warning is raised in that case.

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
            "random_state": seed,
            "N": self.config.get("n_background", 100),
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
        B = max(1, 2**n_features // (n_features + 1))

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
