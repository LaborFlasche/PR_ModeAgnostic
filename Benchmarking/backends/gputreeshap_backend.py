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
    """XGBoost's own native GPU SHAP path (Booster.predict with device="cuda"),
    not a separate package. XGBoost-only. UNVERIFIED ON REAL GPU HARDWARE —
    only the guard/skip path is tested; re-verify before trusting its numbers."""

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

        import xgboost as xgb  # lazy: avoid importing xgboost at module level
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
