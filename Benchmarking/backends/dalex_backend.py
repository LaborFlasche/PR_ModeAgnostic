import dalex
import numpy as np
import pandas as pd

from .base_backend import BaseBackend, marginal_predict


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


class DalexApproxBackend(BaseBackend):
    """dalex's SHAP (Štrumbelj–Kononenko marginal sampling) on the shared value function.

    dalex exposes a single Shapley method: ``predict_parts(type="shap")`` averages a
    feature's contribution over ``B`` random feature orderings, imputing the "removed"
    features from the background ``data``. That is marginal / interventional sampling —
    the same value-function family as the other backends — so within each cell it
    targets the *same* ``marginal_predict`` function as the oracle (margin space for
    gradient boosting / linear, probability for RF / DecisionTree, raw for regressors).
    Because dalex has no kernel-weighting analogue, it is wired to the **permutation**
    approximator slot only (see ``SUPPORTED_APPROXIMATORS``).

    Unlike the coalition-budget samplers, dalex has no fixed coalition budget: its cost is
    ``B × n_features × n_background`` and a single ordering already costs a full sweep over
    all features. To keep its *measured* evaluations comparable to the others at a shared
    nominal ``budget`` — and to stop the eval count exploding in high dimension — the
    budget is mapped to ``B = max(1, round(budget / n_features))``. This holds the total
    near ``budget × n_background`` until ``B`` floors at 1, beyond which dalex is
    unavoidably more expensive than a fixed coalition budget (a real property the
    benchmark records via ``n_model_evals``, the comparable cross-library axis).

    dalex returns variables ordered by ``|contribution|``, with the averaged values at
    ``B == 0``; both are undone here (filter ``B == 0``, reindex by ``variable_name``).

    config:
        approximator: must be ``"permutation"`` (the only method dalex provides).
        budget: shared nominal budget, converted to dalex's ``B`` as above.
    """

    name = "dalex_approx"
    library = "dalex"
    computation_type = "approximation"
    SUPPORTED_APPROXIMATORS = ("permutation",)

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        approximator = self.config.get("approximator", "permutation")
        if approximator not in self.SUPPORTED_APPROXIMATORS:
            raise ValueError(
                f"dalex supports only {self.SUPPORTED_APPROXIMATORS} (got '{approximator}'): "
                "it has a single ordering-sampling SHAP method, with no kernel variant."
            )
        imputer = self.config.get("imputer", "marginal")
        if imputer != "marginal":
            raise ValueError(
                f"dalex backend only supports imputer='marginal' (got {imputer!r}): "
                "predict_parts(type='shap') imputes removed features from the background "
                "distribution (the marginal value function)."
            )
        columns = list(x.columns)
        n_features = x.shape[1]
        budget = self.config.get("budget")
        seed = self.config.get("seed")
        B = max(1, round(budget / n_features)) if budget else 25

        f = marginal_predict(self.model, columns)
        # y=None: predict_parts(type="shap") needs only the model and the background
        # distribution, not labels. predict_function routes through marginal_predict so
        # dalex explains the same scalar output space as the oracle.
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
            pp = explainer.predict_parts(obs, type="shap", B=B, random_state=seed)
            agg = pp.result[pp.result["B"] == 0]
            contrib = agg.set_index("variable_name")["contribution"].reindex(columns)
            rows.append(contrib.to_numpy(dtype=float))

        return pd.DataFrame(rows, index=x.index, columns=columns)
