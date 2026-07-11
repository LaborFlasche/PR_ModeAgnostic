import numpy as np
import pandas as pd
import shapiq

from ...base_backend import BaseBackend, flatten_interactions


def _class_index(model) -> int:
    """Class whose output shapiq explains: 1 for binary, 0 for multiclass,
    matching ShapTrueValueBackend/reduce_multiclass. Uses ``classes_`` (not
    ``n_classes_``, which HistGradientBoosting lacks) so a missing attribute
    can't silently default to binary. Regressors (no ``classes_``) get 1,
    which shapiq ignores for single-output models."""
    n_classes = len(getattr(model, "classes_", ()))
    return 1 if n_classes in (0, 2) else 0


class _ShapIQTreeBackend(BaseBackend):
    """shapiq's tree-specific explainer (distinct from ShapIQTrueValueBackend's
    model-agnostic TabularExplainer)."""

    library = "shapiq"
    computation_type = "true_value"
    mode: str  # "pathdependent" | "interventional"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        reference_dataset = self.background.values if self.mode == "interventional" else None
        class_index = _class_index(self.model)
        explainer = shapiq.TreeExplainer(
            self.model,
            mode=self.mode,
            reference_dataset=reference_dataset,
            max_order=1,
            min_order=1,
            index="SV",
            class_index=class_index,
        )
        results = explainer.explain_X(x.values, n_jobs=1)
        self.baseline_ = float(results[0].baseline_value)
        rows = [np.asarray(iv.get_n_order_values(1)).ravel() for iv in results]
        return pd.DataFrame(rows, index=x.index, columns=x.columns)


class ShapIQTreePathDependentBackend(_ShapIQTreeBackend):
    name = "shapiq_tree_path_dependent"
    mode = "pathdependent"


class ShapIQTreeInterventionalBackend(_ShapIQTreeBackend):
    """shapiq's interventional TreeExplainer. Previously excluded (reported
    topology-dependent hangs/segfaults on shapiq 1.5.0); re-verified clean on
    1.5.2 across config-tree.yaml's full grid (both models, every depth/feature
    count), so it's wired back in. backend_timeout_s remains the safety net if
    an untested topology still trips it."""

    name = "shapiq_tree_interventional"
    mode = "interventional"


class ShapIQInteractionBackend(BaseBackend):
    """Pairwise (order-2, SII) shapiq interactions, path-dependent only
    (interventional order-1 already crashes unreliably; assume order-2 does too).

    Uses index="SII", not shapiq's default "k-SII": SII is the same classical
    Shapley Interaction Index as shap's shap_interaction_values (the oracle),
    so after the /2 correction below they agree exactly. k-SII is a different,
    efficiency-corrected index and would never match.

    Two corrections needed to match shap's convention:
      1. shapiq's get_n_order_values(2) returns the full (unhalved) interaction
         value in both (i, j) and (j, i); shap splits it symmetrically (half in
         each cell). Divide by 2.
      2. shapiq's diagonal is zero; shap/woodelf fold the remaining main effect
         onto the diagonal so a row sums to the first-order value. Fill it:
         diag[i] = SV[i] - sum_{j!=i} interaction[i,j], using this call's own
         order-1 byproduct (which matches shap's SV exactly under SII).
    """

    name = "shapiq_interaction"
    library = "shapiq"
    computation_type = "true_value"
    order = 2

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        class_index = _class_index(self.model)
        explainer = shapiq.TreeExplainer(
            self.model,
            mode="pathdependent",
            max_order=2,
            min_order=1,
            index="SII",
            class_index=class_index,
        )
        results = explainer.explain_X(x.values, n_jobs=1)
        self.baseline_ = float(results[0].baseline_value)
        matrices = []
        for iv in results:
            m2 = np.asarray(iv.get_n_order_values(2)) / 2.0
            sv = np.asarray(iv.get_n_order_values(1))
            np.fill_diagonal(m2, sv - m2.sum(axis=1))
            matrices.append(m2)
        values = np.stack(matrices)
        return flatten_interactions(values, x)
