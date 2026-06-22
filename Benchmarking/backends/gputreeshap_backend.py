import pandas as pd

from .base_backend import (
    BaseBackend,
    nan_result,
    nan_interaction_result,
    reduce_multiclass,
    flatten_interactions,
    cuda_available,
)


class _GPUTreeShapBackend(BaseBackend):
    """XGBoost's own native GPU SHAP path: ``Booster.predict(dmatrix,
    pred_contribs=True / pred_interactions=True, ...)`` with ``device="cuda"``.

    This is not a separate "gputreeshap" package — TreeSHAPBench's own
    extras_and_scaling.ipynb (section E3) uses exactly this XGBoost-native path
    under that name. XGBoost-only: the booster API this needs doesn't exist on
    sklearn/LightGBM models. Gated on ``cuda_available()`` *before* attempting
    anything, since xgboost's device="cuda" does not raise when no GPU is present
    — it silently warns and falls back to CPU (confirmed live) — so a try/except
    would wrongly report CPU timings as "GPU" results.

    UNVERIFIED ON REAL GPU HARDWARE: this machine has no CUDA device, so the
    actual prediction path (output shape, whether/how the trailing bias
    column/row needs dropping) has not been exercised live — only the
    model-type guard and the cuda_available() skip path are tested. Re-verify
    end-to-end on a CUDA machine before trusting its numbers.
    """

    library = "gputreeshap"
    computation_type = "true_value"

    def _check(self, x: pd.DataFrame) -> str | None:
        if not type(self.model).__module__.startswith("xgboost"):
            return f"XGBoost-only (got {type(self.model).__module__})"
        if not cuda_available():
            return "no CUDA device"
        return None


class GPUTreeShapBackend(_GPUTreeShapBackend):
    name = "gputreeshap"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        skip = self._check(x)
        if skip:
            print(f"  [SKIP] {self.name}: {skip}")
            return nan_result(x)

        import xgboost as xgb  # lazy: only reached once the model is already an
        # xgboost object and a CUDA device is confirmed present — see the module
        # docstring on why xgboost must never be imported at module level here.
        booster = self.model.get_booster()
        booster.set_param({"device": "cuda"})
        contribs = booster.predict(xgb.DMatrix(x), pred_contribs=True)
        values = reduce_multiclass(contribs[..., :-1], order=1)  # drop trailing bias column
        return pd.DataFrame(values, index=x.index, columns=x.columns)


class GPUTreeShapInteractionBackend(_GPUTreeShapBackend):
    name = "gputreeshap_interaction"
    order = 2

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        skip = self._check(x)
        if skip:
            print(f"  [SKIP] {self.name}: {skip}")
            return nan_interaction_result(x)

        import xgboost as xgb  # lazy — see GPUTreeShapBackend.run_explainer
        booster = self.model.get_booster()
        booster.set_param({"device": "cuda"})
        interactions = booster.predict(xgb.DMatrix(x), pred_interactions=True)
        values = reduce_multiclass(interactions[..., :-1, :-1], order=2)  # drop bias row/col
        return flatten_interactions(values, x)
