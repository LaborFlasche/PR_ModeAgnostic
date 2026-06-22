import pandas as pd
import shap

from .base_backend import BaseBackend, reduce_multiclass, flatten_interactions


class ShapTreePathDependentBackend(BaseBackend):
    """Path-dependent TreeSHAP: exact Shapley values from the tree's own internal
    sample-weighting, with no background/reference dataset.

    This is a *different* value definition than the existing oracle
    (``ShapTrueValueBackend``, which always passes a background and so gets
    interventional TreeSHAP for tree models) — expect systematic, non-approximation-
    error divergence against it for tree models, by design.
    """

    name = "shap_tree_path_dependent"
    library = "shap"
    computation_type = "true_value"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.TreeExplainer(self.model)
        try:
            sv = explainer(x, check_additivity=False)
        except TypeError:
            sv = explainer(x)
        values = reduce_multiclass(sv.values)
        return pd.DataFrame(values, index=x.index, columns=x.columns)


class ShapInteractionBackend(BaseBackend):
    """Pairwise (order-2) path-dependent SHAP interactions — the order-2 oracle.

    Unlike the order-1 oracle (``ShapTrueValueBackend``, interventional since it
    always gets a background), this one must be path-dependent: shap's
    interventional/independent feature-perturbation mode raises
    ``FEATURE_DEPENDENCE::independent does not support interactions!`` (confirmed
    live) — interaction values are only supported in path-dependent mode at all,
    for shap. Every order-2 backend in this codebase is path-dependent for the
    same reason, so this keeps the oracle consistent with what it's compared
    against.
    """

    name = "shap_interaction"
    library = "shap"
    computation_type = "true_value"
    order = 2

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.TreeExplainer(self.model)
        values = reduce_multiclass(explainer.shap_interaction_values(x), order=2)
        return flatten_interactions(values, x)
