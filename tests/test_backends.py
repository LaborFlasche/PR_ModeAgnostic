# Must precede any shapiq import (incl. transitively via the shapiq_backend import
# below), or a later xgboost/lightgbm .fit() segfaults — see run_benchmark.py.
import xgboost  # noqa: F401
import lightgbm  # noqa: F401

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from backends.true_value.tabular.shap_backend import ShapTrueValueBackend


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
    backend = ShapTrueValueBackend(model, background, {"seed": 0})
    contrib = backend.run_explainer(X_eval)
    assert contrib.shape == (5, 3)


def test_shap_true_value_columns(toy_rf):
    model, X = toy_rf
    backend = ShapTrueValueBackend(model, X.iloc[:10], {"seed": 0})
    contrib = backend.run_explainer(X.iloc[10:15])
    assert list(contrib.columns) == ["f0", "f1", "f2"]


def test_shap_true_value_metadata():
    assert ShapTrueValueBackend.library == "shap"
    assert ShapTrueValueBackend.computation_type == "true_value"
    assert ShapTrueValueBackend.name == "shap_true_value"


from backends.true_value.tabular.shapiq_backend import ShapIQTrueValueBackend


def test_shapiq_true_value_shape(toy_rf):
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:13]
    backend = ShapIQTrueValueBackend(model, background, {"seed": 0})
    contrib = backend.run_explainer(X_eval)
    assert contrib.shape == (3, 3)


def test_shapiq_true_value_columns(toy_rf):
    model, X = toy_rf
    backend = ShapIQTrueValueBackend(model, X.iloc[:10], {"seed": 0})
    contrib = backend.run_explainer(X.iloc[10:13])
    assert list(contrib.columns) == ["f0", "f1", "f2"]


def test_shapiq_true_value_metadata():
    assert ShapIQTrueValueBackend.library == "shapiq"
    assert ShapIQTrueValueBackend.computation_type == "true_value"
    assert ShapIQTrueValueBackend.name == "shapiq_true_value"


from backends.true_value.trees.shap_backend import ShapTreePathDependentBackend
# ShapIQTreeInterventionalBackend not exercised here: it crashes unreliably (see its docstring).
from backends.true_value.trees.shapiq_backend import ShapIQTreePathDependentBackend
from backends.true_value.trees.woodelf_backend import (
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


def test_woodelf_skips_multiclass():
    # lightgbm multiclass specifically: confirmed a genuine upstream woodelf bug
    # (see _woodelf_multiclass_unsupported), unlike sklearn-native multiclass
    # classifiers, which are NOT skipped since they're confirmed to work correctly.
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((60, 3)), columns=["f0", "f1", "f2"])
    y = rng.integers(0, 3, 60)
    model = lightgbm.LGBMClassifier(n_estimators=5, max_depth=3, verbosity=-1, random_state=42)
    model.fit(X, y)
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    backend = WoodelfTreePathDependentBackend(model, background)
    contrib = backend.run_explainer(X_eval)
    assert contrib.isna().all().all()


from backends.true_value.trees.fasttreeshap_backend import FastTreeShapBackend


def test_fasttreeshap_skips_when_venv_missing(toy_rf, monkeypatch):
    monkeypatch.setenv("FASTTREESHAP_VENV_PYTHON", "/nonexistent/python")
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    backend = FastTreeShapBackend(model, background)
    contrib = backend.run_explainer(X_eval)
    assert contrib.shape == (5, 3)
    assert contrib.isna().all().all()


from models.model import Model


def test_model_is_tree():
    assert Model.RANDOM_FOREST.is_tree
    assert Model.DECISION_TREE.is_tree
    assert Model.GRADIENT_BOOSTING.is_tree
    assert Model.XGBOOST.is_tree
    assert Model.LIGHTGBM.is_tree
    assert not Model.LINEAR_BASELINE.is_tree
    assert not Model.LINEAR_REGULARIZED.is_tree
    assert not Model.MLP.is_tree
    assert not Model.TRANSFORMER.is_tree
    assert not Model.CNN_1D.is_tree


# Duck-typed fake, not real xgboost: importing xgboost alongside shapiq's
# interventional TreeExplainer (exercised above) segfaults in this dependency stack.
class _FakeXGBClassifier:
    __module__ = "xgboost.sklearn"

    def predict_proba(self, X):
        return np.tile([0.7, 0.3], (len(X), 1))

    def predict(self, X, output_margin=False):
        assert output_margin is True
        return np.arange(len(X), dtype=float)


def test_marginal_predict_xgboost_classifier_uses_margin():
    from backends.base_backend import marginal_predict
    X = pd.DataFrame(np.zeros((5, 3)), columns=["f0", "f1", "f2"])
    f = marginal_predict(_FakeXGBClassifier(), X.columns)
    np.testing.assert_allclose(f(X), np.arange(5, dtype=float))


# --- Interaction (order-2) backends -----------------------------------------

from backends.true_value.trees.shap_backend import ShapInteractionBackend
from backends.true_value.trees.shapiq_backend import ShapIQInteractionBackend
from backends.true_value.trees.woodelf_backend import WoodelfInteractionBackend


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


def test_woodelf_interaction_skips_multiclass():
    # See test_woodelf_skips_multiclass: lightgbm multiclass is genuinely
    # unsupported, sklearn-native multiclass is not.
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((60, 3)), columns=["f0", "f1", "f2"])
    y = rng.integers(0, 3, 60)
    model = lightgbm.LGBMClassifier(n_estimators=5, max_depth=3, verbosity=-1, random_state=42)
    model.fit(X, y)
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    contrib = WoodelfInteractionBackend(model, background).run_explainer(X_eval)
    assert contrib.shape == (5, 9)
    assert contrib.isna().all().all()


def test_woodelf_runs_sklearn_multiclass():
    """sklearn-native multiclass classifiers are the one case NOT skipped —
    confirmed to agree with the oracle's own multiclass (class-0) convention."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((150, 3)), columns=["f0", "f1", "f2"])
    y = rng.integers(0, 3, 150)
    model = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42)
    model.fit(X, y)
    background = X.iloc[:100]
    X_eval = X.iloc[100:110]
    contrib = WoodelfTreePathDependentBackend(model, background).run_explainer(X_eval)
    assert not contrib.isna().all().all()


# --- Axiom/correctness regression tests --------------------------------------
# Tolerances are set from values measured live on this toy_rf fixture, not
# guessed defaults.

def test_woodelf_interventional_efficiency(toy_rf):
    """sum(shap_values_row) + E[f(background)] ~= f(x): the efficiency axiom.
    Path-dependent backends use a different value definition and don't satisfy
    this against a background mean — see test_path_dependent_backends_agree."""
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    contrib = WoodelfTreeInterventionalBackend(model, background).run_explainer(X_eval)
    baseline = model.predict(background).mean()
    total = contrib.sum(axis=1).to_numpy() + baseline
    np.testing.assert_allclose(total, model.predict(X_eval), atol=1e-6)


def test_path_dependent_backends_agree(toy_rf):
    """shap, shapiq, and woodelf's path-dependent TreeExplainers compute the
    same value (no background), so they should agree with each other directly
    rather than against a baseline."""
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
    """sum_j interaction(i, j) == first-order Shapley value(i): the interaction
    axiom shap/woodelf both fold onto the diagonal. shapiq is excluded here —
    see test_shapiq_interaction_self_consistent and ShapIQInteractionBackend's
    docstring for why it needs a different check."""
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
    """The row-sum axiom holds by construction within a single explainer call
    (the diagonal is built from that call's own order-1 byproduct) — guards
    against a refactor breaking that construction."""
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    d = X.shape[1]
    n_classes = getattr(model, "n_classes_", 2)
    class_index = 1 if n_classes == 2 else 0

    import shapiq
    explainer = shapiq.TreeExplainer(
        model, mode="pathdependent", max_order=2, min_order=1, index="SII", class_index=class_index,
    )
    results = explainer.explain_X(X_eval.values, n_jobs=1)
    sv_byproduct = np.stack([np.asarray(iv.get_n_order_values(1)).ravel() for iv in results])

    interactions = ShapIQInteractionBackend(model, background).run_explainer(X_eval).to_numpy().reshape(len(X_eval), d, d)
    row_sums = interactions.sum(axis=2)
    np.testing.assert_allclose(row_sums, sv_byproduct, atol=1e-10)


def test_shapiq_interaction_matches_oracle(toy_rf):
    """shapiq's SII, after ShapIQInteractionBackend's /2 symmetric-split
    correction, computes the exact same quantity as shap's shap_interaction_values
    (the order-2 oracle) — not just a "closer" index, an exact match. See
    ShapIQInteractionBackend's docstring for the factor-of-2 root cause."""
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    shap_vals = ShapInteractionBackend(model, background).run_explainer(X_eval).to_numpy()
    shapiq_vals = ShapIQInteractionBackend(model, background).run_explainer(X_eval).to_numpy()
    np.testing.assert_allclose(shapiq_vals, shap_vals, atol=1e-6)


# --- GPU-gated backends: skip-path only (no CUDA on this machine) -----------

from backends.true_value.trees.woodelf_backend import (
    WoodelfGPUPathDependentBackend,
    WoodelfGPUInterventionalBackend,
)


@pytest.mark.parametrize("backend_cls", [
    WoodelfGPUPathDependentBackend,
    WoodelfGPUInterventionalBackend,
])
def test_woodelf_gpu_skips_without_cuda(toy_rf, backend_cls, monkeypatch):
    monkeypatch.setattr("backends.true_value.trees.woodelf_backend.cuda_available", lambda: False)
    model, X = toy_rf
    background = X.iloc[:10]
    X_eval = X.iloc[10:15]
    contrib = backend_cls(model, background).run_explainer(X_eval)
    assert contrib.shape == (5, 3)
    assert contrib.isna().all().all()
