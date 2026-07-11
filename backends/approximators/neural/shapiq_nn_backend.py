import pandas as pd
import shapiq

from ...base_backend import BaseBackend, marginal_predict


class ShapIQNNApproxBackend(BaseBackend):
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
