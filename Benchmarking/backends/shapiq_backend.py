import numpy as np
import pandas as pd
import shapiq

from .base_backend import BaseBackend, marginal_predict


class ShapIQTrueValueBackend(BaseBackend):
    name = "shapiq_true_value"
    library = "shapiq"
    computation_type = "true_value"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        background_np = self.background.values.astype(float)
        x_np = x.values.astype(float)
        n_features = x_np.shape[1]

        # budget = 2^n for exact Shapley values; shapiq approximates if budget
        # exceeds available coalitions, so this is always the best-effort exact budget
        budget = 2 ** n_features

        def predict_fn(X: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(X, columns=columns)
            if hasattr(self.model, "predict_proba"):
                out = self.model.predict_proba(df)
                return out[:, 1] if out.shape[1] == 2 else out
            return self.model.predict(df).astype(float)

        explainer = shapiq.TabularExplainer(
            model=predict_fn,
            data=background_np,
            index="SV",
            max_order=1,
        )

        rows = []
        for i in range(len(x_np)):
            iv = explainer.explain(x_np[i], budget=budget)
            rows.append(iv.get_n_order_values(1))

        return pd.DataFrame(rows, index=x.index, columns=columns)


class ShapIQApproxBackend(BaseBackend):
    """shapiq's model-agnostic SV approximators on the marginal value function.

    config:
        approximator: "kernel" (shapiq.approximator.KernelSHAP) or
            "permutation" (shapiq.approximator.PermutationSamplingSV)
        budget: number of coalitions evaluated per instance (shapiq's budget knob).
    """

    name = "shapiq_approx"
    library = "shapiq"
    computation_type = "approximation"

    _APPROXIMATORS = {
        "kernel": shapiq.approximator.KernelSHAP,
        "permutation": shapiq.approximator.PermutationSamplingSV,
    }

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        x_np = x.values.astype(float)
        n_features = x_np.shape[1]

        approximator = self.config.get("approximator", "kernel")
        budget = self.config.get("budget", 256)
        seed = self.config.get("seed")
        imputer = self.config.get("imputer", "marginal")
        if approximator not in self._APPROXIMATORS:
            raise ValueError(f"Unknown shapiq approximator '{approximator}' (use 'kernel' or 'permutation')")

        f = marginal_predict(self.model, columns)
        appr = self._APPROXIMATORS[approximator](n=n_features, random_state=seed)
        explainer = shapiq.TabularExplainer(
            model=f,
            data=self.background.values.astype(float),
            imputer=imputer,  # shared value function across all libraries (config-driven)
            approximator=appr,
            index="SV",
            max_order=1,
            random_state=seed,
        )

        rows = []
        for i in range(len(x_np)):
            iv = explainer.explain(x_np[i], budget=budget)
            rows.append(iv.get_n_order_values(1))

        return pd.DataFrame(rows, index=x.index, columns=columns)
