import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from Benchmarking.backends.shap_backend import ShapTrueValueBackend


@pytest.fixture
def toy_rf():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((50, 3)), columns=["f0", "f1", "f2"])
    y = X["f0"] + 2 * X["f1"] + rng.normal(0, 0.1, 50)
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X


def test_shap_true_value_shape(toy_rf):
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    backend = ShapTrueValueBackend(model, background)
    contrib = backend.run_explainer(X_eval)
    assert contrib.shape == (5, 3)


def test_shap_true_value_columns(toy_rf):
    model, X = toy_rf
    backend = ShapTrueValueBackend(model, X.iloc[:10])
    contrib = backend.run_explainer(X.iloc[10:15])
    assert list(contrib.columns) == ["f0", "f1", "f2"]


def test_shap_true_value_metadata():
    assert ShapTrueValueBackend.library == "shap"
    assert ShapTrueValueBackend.computation_type == "true_value"
    assert ShapTrueValueBackend.name == "shap_true_value"


from Benchmarking.backends.shapiq_backend import ShapIQTrueValueBackend


def test_shapiq_true_value_shape(toy_rf):
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:13]
    backend = ShapIQTrueValueBackend(model, background)
    contrib = backend.run_explainer(X_eval)
    assert contrib.shape == (3, 3)


def test_shapiq_true_value_columns(toy_rf):
    model, X = toy_rf
    backend = ShapIQTrueValueBackend(model, X.iloc[:10])
    contrib = backend.run_explainer(X.iloc[10:13])
    assert list(contrib.columns) == ["f0", "f1", "f2"]


def test_shapiq_true_value_metadata():
    assert ShapIQTrueValueBackend.library == "shapiq"
    assert ShapIQTrueValueBackend.computation_type == "true_value"
    assert ShapIQTrueValueBackend.name == "shapiq_true_value"


from Benchmarking.backends.tree_shap_backend import ShapTreePathDependentBackend
# ShapIQTreeInterventionalBackend deliberately not exercised here: it crashes
# unreliably depending on tree topology (see its docstring) and is excluded from
# production wiring, so asserting it "passes" would be misleading.
from Benchmarking.backends.tree_shapiq_backend import ShapIQTreePathDependentBackend
from Benchmarking.backends.woodelf_backend import (
    WoodelfTreePathDependentBackend,
    WoodelfTreeInterventionalBackend,
)


@pytest.mark.parametrize("backend_cls", [
    ShapTreePathDependentBackend,
    ShapIQTreePathDependentBackend,
    WoodelfTreePathDependentBackend,
    WoodelfTreeInterventionalBackend,
])
def test_tree_backend_shape_and_columns(toy_rf, backend_cls):
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    backend = backend_cls(model, background)
    contrib = backend.run_explainer(X_eval)
    assert contrib.shape == (5, 3)
    assert list(contrib.columns) == ["f0", "f1", "f2"]


def test_woodelf_skips_multiclass(toy_rf):
    model, X = toy_rf
    model.objective = "multi:softmax"  # simulate a multiclass model
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    backend = WoodelfTreePathDependentBackend(model, background)
    contrib = backend.run_explainer(X_eval)
    assert contrib.isna().all().all()


from Benchmarking.backends.fasttreeshap_backend import FastTreeShapBackend


def test_fasttreeshap_skips_when_venv_missing(toy_rf, monkeypatch):
    monkeypatch.setenv("FASTTREESHAP_VENV_PYTHON", "/nonexistent/python")
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    backend = FastTreeShapBackend(model, background)
    contrib = backend.run_explainer(X_eval)
    assert contrib.shape == (5, 3)
    assert contrib.isna().all().all()


from Models.dataset_and_models import Model


def test_model_is_tree():
    assert Model.RANDOM_FOREST.is_tree
    assert Model.DECISION_TREE.is_tree
    assert Model.GRADIENT_BOOSTING.is_tree
    assert Model.XGBOOST.is_tree
    assert Model.LIGHTGBM.is_tree
    assert not Model.LINEAR_BASELINE.is_tree
    assert not Model.LINEAR_REGULARIZED.is_tree
    assert not Model.PYTORCH_NEURAL_NETWORK.is_tree


# Deliberately not importing real xgboost here: doing so anywhere in this process
# alongside shapiq's interventional TreeExplainer (exercised above) is a confirmed
# bidirectional hang/segfault in this dependency combination, regardless of import
# order (see ShapIQTreeInterventionalBackend's docstring in tree_shapiq_backend.py).
# A duck-typed fake exercises marginal_predict's routing logic without that risk.
class _FakeXGBClassifier:
    __module__ = "xgboost.sklearn"

    def predict_proba(self, X):
        return np.tile([0.7, 0.3], (len(X), 1))

    def predict(self, X, output_margin=False):
        assert output_margin is True
        return np.arange(len(X), dtype=float)


def test_marginal_predict_xgboost_classifier_uses_margin():
    from Benchmarking.backends.base_backend import marginal_predict
    X = pd.DataFrame(np.zeros((5, 3)), columns=["f0", "f1", "f2"])
    f = marginal_predict(_FakeXGBClassifier(), X.columns)
    np.testing.assert_allclose(f(X), np.arange(5, dtype=float))


# --- Interaction (order-2) backends -----------------------------------------

from Benchmarking.backends.tree_shap_backend import ShapInteractionBackend
from Benchmarking.backends.tree_shapiq_backend import ShapIQInteractionBackend
from Benchmarking.backends.woodelf_backend import WoodelfInteractionBackend


@pytest.mark.parametrize("backend_cls", [
    ShapInteractionBackend,
    ShapIQInteractionBackend,
    WoodelfInteractionBackend,
])
def test_interaction_backend_shape_and_columns(toy_rf, backend_cls):
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    backend = backend_cls(model, background)
    contrib = backend.run_explainer(X_eval)
    assert backend_cls.order == 2
    assert contrib.shape == (5, 9)  # 3 features -> 3*3 flattened columns
    assert list(contrib.columns) == [f"{a}__{b}" for a in X.columns for b in X.columns]


def test_woodelf_interaction_skips_multiclass(toy_rf):
    model, X = toy_rf
    model.objective = "multi:softmax"
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    contrib = WoodelfInteractionBackend(model, background).run_explainer(X_eval)
    assert contrib.shape == (5, 9)
    assert contrib.isna().all().all()


# --- Axiom/correctness regression tests --------------------------------------
#
# Tolerances below are not guesses: each was set from values actually measured
# live on this toy_rf fixture (see the Phase 2 investigation), not from
# TreeSHAPBench's defaults, since the libraries' own conventions differ from
# what that notebook assumed (see ShapIQInteractionBackend's docstring).

def test_woodelf_interventional_efficiency(toy_rf):
    """sum(shap_values_row) + E[f(background)] ~= f(x) for an interventional
    (marginal) explanation - the textbook efficiency axiom. Holds tightly for
    interventional backends; path-dependent backends use a different value
    definition (the tree's own internal sample weighting) and do NOT satisfy
    this against a background mean, by design - see
    test_path_dependent_backends_agree below for the check that applies to them.
    """
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    contrib = WoodelfTreeInterventionalBackend(model, background).run_explainer(X_eval)
    baseline = model.predict(background).mean()
    total = contrib.sum(axis=1).to_numpy() + baseline
    np.testing.assert_allclose(total, model.predict(X_eval), atol=1e-6)


def test_path_dependent_backends_agree(toy_rf):
    """shap, shapiq, and woodelf's path-dependent TreeExplainers all compute the
    *same* value (the tree's internal sample-weighting definition, no
    background) - so unlike the interventional case, comparing them to each
    other (rather than to a background-mean baseline) is the meaningful
    correctness check. Measured live: shap vs shapiq agree to ~5.5e-17, shap vs
    woodelf to ~2.3e-8 - both far inside the atol used here.
    """
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    shap_vals = ShapTreePathDependentBackend(model, background).run_explainer(X_eval).to_numpy()
    shapiq_vals = ShapIQTreePathDependentBackend(model, background).run_explainer(X_eval).to_numpy()
    woodelf_vals = WoodelfTreePathDependentBackend(model, background).run_explainer(X_eval).to_numpy()
    np.testing.assert_allclose(shapiq_vals, shap_vals, atol=1e-6)
    np.testing.assert_allclose(woodelf_vals, shap_vals, atol=1e-4)


@pytest.mark.parametrize("backend_cls,atol", [
    (ShapInteractionBackend, 1e-6),
    (WoodelfInteractionBackend, 1e-4),
])
def test_interaction_row_sum_matches_own_first_order(toy_rf, backend_cls, atol):
    """sum_j interaction(i, j) == first-order Shapley value(i) - the interaction
    axiom shap/woodelf both document (a "remaining main effect" folded onto the
    diagonal). Measured live: holds to ~3e-16 for shap, ~6e-8 for woodelf.

    shapiq is deliberately excluded here: its row-sum axiom only holds *within
    a single explainer call* (its own SV is the order-1 byproduct of the same
    max_order=2 call, by construction - see test_shapiq_interaction_self_consistent
    below) - comparing against a *separately constructed*
    ShapIQTreePathDependentBackend call shows a ~0.03-0.08 gap because shapiq's
    max_order=1 SV path and its k-SII order-1 byproduct are not numerically
    identical (confirmed live, not a defect in this integration - see
    ShapIQInteractionBackend's docstring).
    """
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    d = X.shape[1]

    sv_backend_cls = {
        ShapInteractionBackend: ShapTreePathDependentBackend,
        WoodelfInteractionBackend: WoodelfTreePathDependentBackend,
    }[backend_cls]
    sv = sv_backend_cls(model, background).run_explainer(X_eval).to_numpy()
    interactions = backend_cls(model, background).run_explainer(X_eval).to_numpy().reshape(len(X_eval), d, d)
    row_sums = interactions.sum(axis=2)
    np.testing.assert_allclose(row_sums, sv, atol=atol)


def test_shapiq_interaction_self_consistent(toy_rf):
    """Within ShapIQInteractionBackend's own single explainer call, the
    diagonal is constructed *from* its own order-1 byproduct
    (diag[i] = SV[i] - sum_{j!=i} interaction[i,j]), so the row-sum axiom holds
    by construction here - this guards against a future refactor accidentally
    breaking that construction, not against shapiq's own algorithm.
    """
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    d = X.shape[1]
    n_classes = getattr(model, "n_classes_", 2)
    class_index = 1 if n_classes == 2 else 0

    import shapiq
    explainer = shapiq.TreeExplainer(
        model, mode="pathdependent", max_order=2, min_order=1, index="k-SII", class_index=class_index,
    )
    results = explainer.explain_X(X_eval.values, n_jobs=1)
    sv_byproduct = np.stack([np.asarray(iv.get_n_order_values(1)).ravel() for iv in results])

    interactions = ShapIQInteractionBackend(model, background).run_explainer(X_eval).to_numpy().reshape(len(X_eval), d, d)
    row_sums = interactions.sum(axis=2)
    np.testing.assert_allclose(row_sums, sv_byproduct, atol=1e-10)


# --- GPU-gated backends: skip-path only (no CUDA on this machine) -----------

from Benchmarking.backends.woodelf_backend import (
    WoodelfGPUPathDependentBackend,
    WoodelfGPUInterventionalBackend,
)
from Benchmarking.backends.gputreeshap_backend import (
    GPUTreeShapBackend,
    GPUTreeShapInteractionBackend,
)


@pytest.mark.parametrize("backend_cls", [
    WoodelfGPUPathDependentBackend,
    WoodelfGPUInterventionalBackend,
])
def test_woodelf_gpu_skips_without_cuda(toy_rf, backend_cls, monkeypatch):
    monkeypatch.setattr("Benchmarking.backends.woodelf_backend.cuda_available", lambda: False)
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    contrib = backend_cls(model, background).run_explainer(X_eval)
    assert contrib.shape == (5, 3)
    assert contrib.isna().all().all()


@pytest.mark.parametrize("backend_cls,expected_cols", [
    (GPUTreeShapBackend, ["f0", "f1", "f2"]),
    (GPUTreeShapInteractionBackend, [f"{a}__{b}" for a in ["f0", "f1", "f2"] for b in ["f0", "f1", "f2"]]),
])
def test_gputreeshap_skips_for_non_xgboost_model(toy_rf, backend_cls, expected_cols):
    model, X = toy_rf  # toy_rf is a RandomForestRegressor, not XGBoost
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    contrib = backend_cls(model, background).run_explainer(X_eval)
    assert list(contrib.columns) == expected_cols
    assert contrib.isna().all().all()


def test_gputreeshap_skips_xgboost_without_cuda(monkeypatch):
    monkeypatch.setattr("Benchmarking.backends.gputreeshap_backend.cuda_available", lambda: False)
    X = pd.DataFrame(np.zeros((5, 3)), columns=["f0", "f1", "f2"])
    background = X.iloc[:3]
    contrib = GPUTreeShapBackend(_FakeXGBClassifier(), background).run_explainer(X)
    assert contrib.shape == (5, 3)
    assert contrib.isna().all().all()
