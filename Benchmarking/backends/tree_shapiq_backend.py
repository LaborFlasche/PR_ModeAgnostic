import numpy as np
import pandas as pd
import shapiq

from .base_backend import BaseBackend, flatten_interactions


class _ShapIQTreeBackend(BaseBackend):
    """Shared logic for shapiq's tree-specific explainer (distinct from
    ``ShapIQTrueValueBackend`` in shapiq_backend.py, which brute-forces the
    model-agnostic ``TabularExplainer`` instead of exploiting tree structure).
    """

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
    """shapiq's interventional TreeExplainer.

    NOT WIRED INTO PRODUCTION (intentionally excluded from
    slurm/run_benchmark.py's TREE_TRUE_VALUE_MAP and from configs/config.yaml) due
    to firsthand-reproduced crashes in this dependency combination (shapiq 1.5.0 +
    numba 0.65.1 + llvmlite 0.47.0 on macOS arm64):

    * hangs indefinitely on actual XGBoost/LightGBM models;
    * segfaults on plain sklearn models (RandomForestRegressor etc.) too — not
      just a boosting-library issue. Confirmed non-deterministically depending on
      tree topology: passes on one toy RandomForest, segfaults on another trained
      on real data with the same hyperparameters. This is not import-order
      contamination (reproduced with no xgboost/lightgbm touched in the process
      at all) — it is a bug in shapiq's interventional TreeExplainer itself in
      this environment.

    Path-dependent mode (``ShapIQTreePathDependentBackend``), woodelf (both
    modes), and fasttreeshap are all unaffected — verified individually and
    repeatedly. This class is kept for reference/future shapiq versions, but
    should not be re-enabled without re-verifying on the target environment.
    """

    name = "shapiq_tree_interventional"
    mode = "interventional"


class ShapIQInteractionBackend(BaseBackend):
    """Pairwise (order-2, k-SII) shapiq interactions, path-dependent only.

    Interventional order-2 is deliberately not implemented: order-1 interventional
    (``ShapIQTreeInterventionalBackend`` above) crashes unreliably in this
    dependency stack, and the same numba code path almost certainly underlies
    order-2 interventional too. This assumption hasn't been independently
    re-verified at order 2 — do so before ever wiring up an interventional
    variant here.

    shapiq's ``get_n_order_values(2)`` returns a *zero-diagonal* matrix — pure
    pairwise terms only, with the order-1 (main-effect) values kept separate.
    shap's and woodelf's ``shap_interaction_values()`` instead fold a "remaining
    main effect" onto the diagonal, so the row-sum of a feature's full row
    equals its first-order Shapley value (verified empirically: shap and woodelf
    agree on this to ~1e-8, shapiq's raw zero-diagonal output does not). The
    diagonal is filled in here the same way woodelf documents computing it:
    ``diag[i] = SV[i] - sum_{j!=i} interaction[i,j]`` — this makes all three
    libraries' order-2 output directly comparable in the benchmark, not just
    differently-shaped numbers that happen to share a column count.

    Caveat confirmed live: the ``SV`` used above is the order-1 *byproduct* of
    this same ``index="k-SII", max_order=2`` call (``get_n_order_values(1)``),
    not a separately-requested ``index="SV", max_order=1`` computation — those
    two differ by ~0.07 on a toy RandomForest even though both are nominally
    "the Shapley value" for the same model/input (verified: shapiq's max_order=1
    SV path and its k-SII order-1 component are not numerically identical, not
    just floating-point noise). So the row-sum axiom holds tightly *within* this
    backend's own single call, but comparing this backend's row-sum against
    ``ShapIQTreePathDependentBackend``'s separately-computed SV will show a
    similar ~0.03-0.08 gap — that's shapiq's own algorithm choice, not a defect
    in this integration.
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
            index="k-SII",
            class_index=class_index,
        )
        results = explainer.explain_X(x.values, n_jobs=1)
        matrices = []
        for iv in results:
            m2 = np.asarray(iv.get_n_order_values(2))
            sv = np.asarray(iv.get_n_order_values(1))
            np.fill_diagonal(m2, sv - m2.sum(axis=1))
            matrices.append(m2)
        values = np.stack(matrices)
        return flatten_interactions(values, x)
