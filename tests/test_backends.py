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
