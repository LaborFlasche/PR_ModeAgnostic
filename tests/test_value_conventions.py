"""Value-function and class-selection conventions the backends must agree on:
the shap oracle, marginal_predict, woodelf's sign-flip guard, and shapiq's
class_index must all explain the same scalar game per model type."""
# Must precede every other import, or a later xgboost/lightgbm .fit()
# segfaults — same load-order rule as run_benchmark.py.
import xgboost  # noqa: F401  isort:skip
import lightgbm  # noqa: F401  isort:skip

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from Benchmarking.backends.base_backend import marginal_predict
from Benchmarking.backends.tree_shapiq_backend import _class_index
from Benchmarking.backends.woodelf_backend import _woodelf_class_sign_is_flipped


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), y


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5,
                               n_classes=3, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), y


@pytest.fixture(scope="module")
def fitted(binary_data, multiclass_data):
    Xb, yb = binary_data
    Xm, ym = multiclass_data
    return {
        "rf_binary": RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0).fit(Xb, yb),
        "dt_binary": DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xb, yb),
        "hgb_binary": HistGradientBoostingClassifier(max_iter=15, random_state=0).fit(Xb, yb),
        "lgbm_binary": lightgbm.LGBMClassifier(
            n_estimators=15, max_depth=4, verbosity=-1, random_state=0).fit(Xb, yb),
        "xgb_binary": xgboost.XGBClassifier(
            n_estimators=15, max_depth=4, random_state=0, enable_categorical=False).fit(Xb, yb),
        "rf_multi": RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0).fit(Xm, ym),
        "hgb_multi": HistGradientBoostingClassifier(max_iter=15, random_state=0).fit(Xm, ym),
        "rf_regressor": RandomForestRegressor(
            n_estimators=10, max_depth=4, random_state=0).fit(Xb, yb.astype(float)),
    }


@pytest.mark.parametrize("model_key", ["lgbm_binary", "xgb_binary", "rf_binary", "hgb_binary"])
def test_oracle_is_additive_in_marginal_predict_space(model_key, fitted, binary_data):
    """The shap oracle's Shapley sums must satisfy local accuracy against the
    exact value function marginal_predict builds — this is what additivity_gap
    measures. Catches output-space mismatches like LGBMClassifier's margin
    leaves vs its probability predict_proba (each backend must hit the branch
    matching what shap.Explainer is additive in)."""
    X, _ = binary_data
    model = fitted[model_key]
    bg, xe = X.iloc[:100], X.iloc[100:110]
    sv = shap.Explainer(model, bg)(xe, check_additivity=False).values
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    f = marginal_predict(model, X.columns)
    baseline = float(np.mean(np.asarray(f(bg), dtype=float)))
    gap = np.abs(baseline + sv.sum(axis=1) - np.asarray(f(xe), dtype=float)).max()
    # A wrong output space (e.g. margin-space sums vs probability-space f) puts
    # the gap at O(1); 0.05 stays far below that while allowing shap's known
    # ~5e-3 traversal slack on HistGradientBoosting's binned trees (the reason
    # ShapTrueValueBackend passes check_additivity=False).
    assert gap < 0.05


def test_path_dependent_rows_are_additive_against_own_base_value(binary_data, tmp_path):
    """Path-dependent backends explain a different game than the marginal one;
    the runner must check their additivity against the base value they report
    (backend.baseline_), not the shared background mean — otherwise every
    path-dependent and order-2 row carries a fake constant gap."""
    from Benchmarking.runner import BenchmarkRunner
    from Benchmarking.backends import (
        ShapTrueValueBackend,
        ShapTreePathDependentBackend,
        ShapIQTreePathDependentBackend,
        ShapInteractionBackend,
    )

    X, y = binary_data
    model = lightgbm.LGBMClassifier(
        n_estimators=15, max_depth=4, verbosity=-1, random_state=0).fit(X, y)
    csv = str(tmp_path / "gaps.csv")
    runner = BenchmarkRunner(
        true_value_backends=[ShapTrueValueBackend, ShapTreePathDependentBackend,
                             ShapIQTreePathDependentBackend],
        approximation_specs=[], output_csv=csv, n_background=100, n_eval=8)
    runner.run(model, X, run_meta={"dataset": "d", "model": "m", "order": 1,
                                   "n_features": X.shape[1], "n_samples": len(X)})
    runner2 = BenchmarkRunner(
        true_value_backends=[ShapInteractionBackend],
        approximation_specs=[], output_csv=csv, n_background=100, n_eval=8)
    runner2.run(model, X, run_meta={"dataset": "d", "model": "m", "order": 2,
                                    "n_features": X.shape[1], "n_samples": len(X)})

    df = pd.read_csv(csv)
    assert (df["additivity_gap"] < 1e-4).all(), df[["backend", "additivity_gap"]]
    # The two games really have different base values (margin space here), so a
    # shared baseline could not have produced these gaps.
    oracle_base = df.loc[df["backend"] == "shap_true_value", "base_value"].iloc[0]
    pathdep_base = df.loc[df["backend"] == "shap_tree_path_dependent", "base_value"].iloc[0]
    assert abs(oracle_base - pathdep_base) > 1e-3


def test_woodelf_sign_flip_only_for_probability_leaves(fitted):
    """The flip corrects shap's per-class-column reshape of sklearn CART leaf
    values — it must not fire for margin-leaf models (HistGradientBoosting,
    xgboost, lightgbm), whose index 0 is already the class-1 direction."""
    expected = {"rf_binary": True, "dt_binary": True, "hgb_binary": False,
                "lgbm_binary": False, "xgb_binary": False, "rf_multi": False,
                "hgb_multi": False, "rf_regressor": False}
    got = {k: _woodelf_class_sign_is_flipped(m) for k, m in fitted.items()}
    assert got == expected


def test_shapiq_class_index_convention(fitted):
    """Binary -> class 1, multiclass -> class 0 (matching reduce_multiclass),
    derived from classes_ — n_classes_ is absent on HistGradientBoosting, and
    a default would silently treat multiclass as binary."""
    expected = {"rf_binary": 1, "dt_binary": 1, "hgb_binary": 1, "lgbm_binary": 1,
                "xgb_binary": 1, "rf_multi": 0, "hgb_multi": 0, "rf_regressor": 1}
    got = {k: _class_index(m) for k, m in fitted.items()}
    assert got == expected
