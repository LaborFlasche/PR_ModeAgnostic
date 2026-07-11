import pandas as pd
import shapiq

from ...base_backend import BaseBackend, marginal_predict


class ShapIQApproxBackend(BaseBackend):
    """shapiq's model-agnostic SV approximators on the marginal value function.

    config:
        approximator: "kernel" (shapiq.approximator.KernelSHAP) or
            "permutation" (shapiq.approximator.PermutationSamplingSV)
        budget: number of coalitions evaluated per instance (shapiq's budget knob).
        imputer: value function passed to TabularExplainer ("marginal" default,
            injected from the config by BenchmarkRunner).
    """

    name = "shapiq_approx"
    library = "shapiq"
    computation_type = "approximation"

    SUPPORTED_APPROXIMATORS = ("kernel", "permutation")

    _APPROXIMATORS = {
        "kernel": shapiq.approximator.KernelSHAP,
        "permutation": shapiq.approximator.PermutationSamplingSV,
    }

    def load_config(self):
        if "seed" in self.config and self.config["seed"] is not None:
            seed = self.config["seed"]
        else:
            raise ValueError("ShapIQApproxBackend requires a 'seed' in the config.")
        assert self.config.get("approximator", "permutation") in self.SUPPORTED_APPROXIMATORS, \
            f"approximator must be one of {self.SUPPORTED_APPROXIMATORS} (got {self.config.get('approximator')!r})"

        return {
            "random_state": seed,
            "approximator": self.config.get("approximator", "permutation"),
            "budget": self.config.get("budget"),
            "imputer": self.config.get("imputer", "marginal"), # value function (config-driven)
        }

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        columns = x.columns
        x_np = x.values.astype(float)
        n_features = x_np.shape[1]
        config = self.load_config()

        f = marginal_predict(self.model, columns)
        appr = self._APPROXIMATORS[config["approximator"]](n=n_features, random_state=config["random_state"])
        explainer = shapiq.TabularExplainer(
            model=f,
            data=self.background.values.astype(float),
            imputer=config["imputer"],
            approximator=appr,
            index="SV",
            max_order=1,
            random_state=config["random_state"],
        )

        rows = []
        for i in range(len(x_np)):
            iv = explainer.explain(x_np[i], budget=config["budget"])
            rows.append(iv.get_n_order_values(1))

        return pd.DataFrame(rows, index=x.index, columns=columns)
