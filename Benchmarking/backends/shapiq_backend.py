import numpy as np
import pandas as pd
import shapiq

from .base_backend import BaseBackend, marginal_predict


class ShapIQTrueValueBackend(BaseBackend):
    name = "shapiq_true_value"
    library = "shapiq"
    computation_type = "true_value"

    # Exact SVs need all 2**n coalitions, and shapiq's coalition sampler
    # allocates np.zeros((budget, n)) up front — beyond ~2**30 that raises
    # "Maximum allowed dimension exceeded" (ames: 2**79, gisette: 2**256).
    # Above the cap this backend is therefore a best-effort KernelSHAP
    # reference at MAX_BUDGET coalitions, not an exact oracle.
    MAX_BUDGET = 2 ** 14

    @classmethod
    def effective_budget(cls, n_features: int) -> int:
        """2**n_features (exact) while allocatable, else MAX_BUDGET."""
        return min(2 ** n_features, cls.MAX_BUDGET)

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        background_np = self.background.values.astype(float)
        x_np = x.values.astype(float)
        n_features = x_np.shape[1]

        budget = self.effective_budget(n_features)
        seed = self.config.get("seed")
        imputer = self.config.get("imputer", "marginal")

        def predict_fn(X: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(X, columns=columns)
            if hasattr(self.model, "predict_proba"):
                out = self.model.predict_proba(df)
                return out[:, 1] if out.shape[1] == 2 else out
            return self.model.predict(df).astype(float)

        explainer = shapiq.TabularExplainer(
            model=predict_fn,
            data=background_np,
            imputer=imputer,  # shared value function across all libraries (config-driven)
            index="SV",
            max_order=1,
            random_state=seed,
        )

        rows = []
        for i in range(len(x_np)):
            iv = explainer.explain(x_np[i], budget=budget)
            rows.append(iv.get_n_order_values(1))

        return pd.DataFrame(rows, index=x.index, columns=columns)


class ShapIQProxyBackend(BaseBackend):
    """shapiq's ProxySHAP on the marginal value function: fits a proxy model
    (xgboost by default) on sampled coalitions, reads SVs off the proxy, and
    applies an MSR residual adjustment. Model-agnostic, so it competes with
    kernel/permutation in the NN approximation sweep (RQ3).

    config:
        budget: number of coalitions evaluated per instance.
        proxy_model: "xgboost" (default), "lightgbm", "tree", or "linear".
            With torch already imported, "xgboost"/"lightgbm" segfault on
            macOS arm64 (duplicate OpenMP runtimes) — fine on the Linux
            cluster, where torch+xgboost already coexist in the tree sweep.
            Tests therefore run with "tree".
    """

    name = "shapiq_proxy"
    library = "shapiq_proxy"
    computation_type = "approximation"

    SUPPORTED_APPROXIMATORS = ("proxy",)

    _PROXY_MODELS = ("xgboost", "lightgbm", "tree", "linear")

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        x_np = x.values.astype(float)
        n_features = x_np.shape[1]

        budget = self.config.get("budget", 256)
        seed = self.config.get("seed")
        imputer = self.config.get("imputer", "marginal")
        proxy_model = self.config.get("proxy_model", "xgboost")
        # shapiq silently accepts unknown proxy_model strings and still returns
        # values, so a config typo would mislabel benchmark results — reject here.
        if proxy_model not in self._PROXY_MODELS:
            raise ValueError(
                f"Unknown ProxySHAP proxy_model '{proxy_model}' (use one of {self._PROXY_MODELS})"
            )

        f = marginal_predict(self.model, columns)
        # ProxySHAP defaults to index="k-SII", max_order=2 — pin both to plain
        # first-order Shapley values like every other approximator here.
        appr = shapiq.approximator.ProxySHAP(
            n=n_features, index="SV", max_order=1,
            proxy_model=proxy_model, random_state=seed,
        )
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

    SUPPORTED_APPROXIMATORS = ("kernel", "permutation")

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
