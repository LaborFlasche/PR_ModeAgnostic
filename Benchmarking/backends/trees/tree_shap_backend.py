import pandas as pd
import shap

from ..base_backend import BaseBackend, reduce_multiclass, flatten_interactions, select_base_value


class ShapTreePathDependentBackend(BaseBackend):
    """Path-dependent TreeSHAP (no background) — a different value definition
    than the interventional oracle, so expect systematic divergence by design."""

    name = "shap_tree_path_dependent"
    library = "shap"
    computation_type = "true_value"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.TreeExplainer(self.model)
        try:
            sv = explainer(x, check_additivity=False)
        except TypeError:
            sv = explainer(x)
        self.baseline_ = select_base_value(explainer.expected_value)
        values = reduce_multiclass(sv.values)
        return pd.DataFrame(values, index=x.index, columns=x.columns)


class ShapInteractionBackend(BaseBackend):
    """Pairwise (order-2) path-dependent SHAP interactions — the order-2 oracle.
    Must be path-dependent: shap's interventional mode raises
    "FEATURE_DEPENDENCE::independent does not support interactions!"."""

    name = "shap_interaction"
    library = "shap"
    computation_type = "true_value"
    order = 2

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.TreeExplainer(self.model)
        self.baseline_ = select_base_value(explainer.expected_value)
        values = reduce_multiclass(explainer.shap_interaction_values(x), order=2)
        return flatten_interactions(values, x)
