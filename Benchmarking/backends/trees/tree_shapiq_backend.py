import numpy as np
import pandas as pd
import shapiq

from ..base_backend import BaseBackend, flatten_interactions


class _ShapIQTreeBackend(BaseBackend):
    """shapiq's tree-specific explainer (distinct from ShapIQTrueValueBackend's
    model-agnostic TabularExplainer)."""

    library = "shapiq"
    computation_type = "true_value"
    mode: str  # "pathdependent" | "interventional"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        reference_dataset = self.background.values if self.mode == "interventional" else None
        n_classes = getattr(self.model, "n_classes_", 2)
        class_index = 1 if n_classes == 2 else 0
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
        rows = [np.asarray(iv.get_n_order_values(1)).ravel() for iv in results]
        return pd.DataFrame(rows, index=x.index, columns=x.columns)


class ShapIQTreePathDependentBackend(_ShapIQTreeBackend):
    name = "shapiq_tree_path_dependent"
    mode = "pathdependent"


class ShapIQTreeInterventionalBackend(_ShapIQTreeBackend):
    """shapiq's interventional TreeExplainer. Previously excluded from
    TREE_TRUE_VALUE_MAP: reported to hang on real XGBoost/LightGBM and segfault
    on plain sklearn models (non-deterministically, depending on tree topology)
    against shapiq 1.5.0 + numba 0.65.1 + llvmlite 0.47.0 (macOS arm64) — not an
    import-order issue. Re-verified against shapiq 1.5.2 (same numba/llvmlite,
    same machine) across configs/RQ4-tree/config-tree.yaml's full grid — both models,
    every max_depth (4-50), every n_features (4-256) — with no hangs or
    crashes, so it's wired back in. Since the original report was
    topology-dependent, BenchmarkRunner's backend_timeout_s is the safety net
    if an untested topology still trips it."""

    name = "shapiq_tree_interventional"
    mode = "interventional"


class ShapIQInteractionBackend(BaseBackend):
    """Pairwise (order-2, SII) shapiq interactions, path-dependent only
    (interventional order-1 already crashes unreliably; assume order-2 does too).

    Uses index="SII" (not shapiq's own default of "k-SII"): SII is the classical
    Shapley Interaction Index shap's shap_interaction_values (the order-2 oracle)
    also computes — they're the same quantity by definition, and (after the /2
    below) now agree exactly (mean_abs_diff ~0.0 on a toy model), confirming this
    is a real bug fix, not just a closer index label. k-SII is a genuinely
    different, efficiency-corrected index and was never expected to match.

    Two corrections vs. shapiq's raw output are needed to recover shap's
    convention:
      1. shapiq's get_n_order_values(2) returns the FULL (unhalved) interaction
         value in both (i, j) and (j, i); shap splits it symmetrically, storing
         HALF in each of the two cells (so the pair's total effect is
         cell[i,j] + cell[j,i], each equal). Confirmed empirically: every
         off-diagonal cell of shapiq's raw matrix was exactly 2x shap's — not an
         approximate ratio, a fixed factor. Divide by 2 before anything else.
      2. shapiq's diagonal is zero (pure pairwise terms only), unlike shap/
         woodelf which fold the remaining main effect onto the diagonal so a row
         sums to the first-order value. Filled in the same way:
         diag[i] = SV[i] - sum_{j!=i} interaction[i,j] (using the already-halved
         off-diagonal values), from the order-1 byproduct of this same call —
         which, with index="SII", now also matches shap's SV exactly (unlike
         k-SII's byproduct, which the previous version of this docstring noted
         differed by ~0.03-0.08 on a toy model; that discrepancy doesn't occur
         with SII).
    """

    name = "shapiq_interaction"
    library = "shapiq"
    computation_type = "true_value"
    order = 2

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        n_classes = getattr(self.model, "n_classes_", 2)
        class_index = 1 if n_classes == 2 else 0
        explainer = shapiq.TreeExplainer(
            self.model,
            mode="pathdependent",
            max_order=2,
            min_order=1,
            index="SII",
            class_index=class_index,
        )
        results = explainer.explain_X(x.values, n_jobs=1)
        matrices = []
        for iv in results:
            m2 = np.asarray(iv.get_n_order_values(2)) / 2.0
            sv = np.asarray(iv.get_n_order_values(1))
            np.fill_diagonal(m2, sv - m2.sum(axis=1))
            matrices.append(m2)
        values = np.stack(matrices)
        return flatten_interactions(values, x)
