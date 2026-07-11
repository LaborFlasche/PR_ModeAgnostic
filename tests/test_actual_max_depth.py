"""actual_max_depth must report the realized depth of the fitted model, not
the configured cap, for every tree model type in the benchmark."""
# Must precede every other import (models.trainers pulls in torch), or a later
# xgboost/lightgbm .fit() segfaults — same load-order rule as run_benchmark.py.
import xgboost  # noqa: F401  isort:skip
import lightgbm  # noqa: F401  isort:skip

import pytest
from sklearn.datasets import make_regression

from datasets.load_datasets import Dataset
from models.model import Model, actual_max_depth

TREE_MODELS = [m for m in Model if m.is_tree]


@pytest.fixture(scope="module")
def data():
    X, y = make_regression(n_samples=300, n_features=8, random_state=0)
    return X, y


@pytest.mark.parametrize("model_enum", TREE_MODELS, ids=lambda m: m.value)
def test_binding_cap_is_respected(model_enum, data):
    X, y = data
    trainer = model_enum.get_model_with_params(
        Dataset.CALIFORNIA_HOUSING, {"max_depth": 3}, seed=0)
    trainer.fit(X, y, task="regression")
    depth = actual_max_depth(trainer.get_model())
    assert 0 < depth <= 3


@pytest.mark.parametrize("model_enum", TREE_MODELS, ids=lambda m: m.value)
def test_loose_cap_reports_realized_depth(model_enum, data):
    X, y = data
    # 300 samples can never grow a depth-1000 tree, so the reported depth must
    # be the realized one, strictly below the cap.
    trainer = model_enum.get_model_with_params(
        Dataset.CALIFORNIA_HOUSING, {"max_depth": 1000}, seed=0)
    trainer.fit(X, y, task="regression")
    depth = actual_max_depth(trainer.get_model())
    assert 0 < depth < 1000


def test_non_tree_model_raises():
    trainer = Model.LINEAR_BASELINE.get_model_with_params(
        Dataset.CALIFORNIA_HOUSING, {}, seed=0)
    X, y = make_regression(n_samples=50, n_features=4, random_state=0)
    trainer.fit(X, y, task="regression")
    with pytest.raises(TypeError):
        actual_max_depth(trainer.get_model())
