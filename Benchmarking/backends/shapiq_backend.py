import numpy as np
import pandas as pd
import shapiq

from .base_backend import BaseBackend, marginal_predict


class ShapIQTrueValueBackend(BaseBackend):
    """Exact Shapley values via full coalition enumeration, feasible only up to
    ``_EXACT_MAX_FEATURES``. Above that the budget is capped at
    ``2**_EXACT_MAX_FEATURES``: shapiq 1.5.x otherwise tries to allocate a
    2^n-element array and crashes ("Maximum allowed dimension exceeded"), and
    exact values are computationally impossible anyway. Capped cells are a
    budget-capped *reference*, not ground truth — interpret accuracy metrics
    against them as agreement, not error."""

    name = "shapiq_true_value"
    library = "shapiq"
    computation_type = "true_value"

    # 2^14 = 16384 coalitions: exact for adult_census (14 features, the widest
    # cell meant to have a true oracle) and still tractable as a capped
    # reference budget for the wide NN cells (ames 79, gisette 256).
    _EXACT_MAX_FEATURES = 14

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        background_np = self.background.values.astype(float)
        x_np = x.values.astype(float)
        n_features = x_np.shape[1]

        # budget = 2^n enumerates every coalition (exact); capped for wide inputs
        budget = 2 ** min(n_features, self._EXACT_MAX_FEATURES)
        if n_features > self._EXACT_MAX_FEATURES:
            print(
                f"  [WARN] {self.name}: n_features={n_features} > "
                f"{self._EXACT_MAX_FEATURES} — budget capped at 2^"
                f"{self._EXACT_MAX_FEATURES}={budget}; reference values are "
                "approximate, not exact"
            )

        def predict_fn(X: np.ndarray) -> np.ndarray:
            # Must return a 1D scalar game like every other backend's value function
            # (see marginal_predict): a 2D multiclass return would fall through to
            # shapiq's default class_index=1, silently explaining a different class
            # than the class-0 convention every other backend uses.
            df = pd.DataFrame(X, columns=columns)
            if hasattr(self.model, "predict_proba"):
                out = self.model.predict_proba(df)
                return out[:, 1] if out.shape[1] == 2 else out[:, 0]
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
