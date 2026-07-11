import pandas as pd
import shapiq

from ...base_backend import BaseBackend, marginal_predict


class ShapIQTrueValueBackend(BaseBackend):
    """Exact Shapley values via shapiq's TabularExplainer (budget = 2^n_features)."""

    name = "shapiq_true_value"
    library = "shapiq"
    computation_type = "true_value"

    _EXACT_MAX_FEATURES = 14

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("ShapIQTrueValueBackend requires a 'seed' in the config.")
        return {
            "random_state": seed,
        }

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        # shapiq needs numpy arrays, not DataFrames.
        background_np = self.background.values.astype(float)
        x_np = x.values.astype(float)
        n_features = x_np.shape[1]
        config = self.load_config()

        # budget = 2^n enumerates every coalition (exact); capped for wide inputs
        budget = 2 ** n_features
        if n_features > self._EXACT_MAX_FEATURES:
            print(
                f"  [WARN] {self.name}: n_features={n_features} > "
                f"{self._EXACT_MAX_FEATURES} — budget capped at 2^"
                f"{self._EXACT_MAX_FEATURES}={budget}; reference values are "
                "approximate, not exact"
            )

        f = marginal_predict(self.model, columns)

        # approximator="auto" resolves to exact computation at this few features.
        explainer = shapiq.TabularExplainer(
            model=f,
            data=background_np,
            index="SV",
            approximator="auto",
            max_order=1,
            **config,
        )

        rows = []
        for i in range(len(x_np)):
            iv = explainer.explain(x_np[i], budget=budget)
            rows.append(iv.get_n_order_values(1))

        return pd.DataFrame(rows, index=x.index, columns=columns)
