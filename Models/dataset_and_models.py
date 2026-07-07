from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge
)
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)
# xgboost/lightgbm are imported lazily below (not at module level): importing
# xgboost before shapiq segfaults shapiq's interventional TreeExplainer later —
# see tree_shapiq_backend.py.

# The Dataset enum (and its DatasetSpec-based loading) lives in one place —
# Datasets/load_datasets.py — and is re-exported here so existing imports
# (`from Models.dataset_and_models import Dataset, Model`) keep working.
from Datasets.load_datasets import Dataset  # noqa: F401  (re-export)

from Models.trainers import SklearnTrainer, PytorchTrainer
from enum import Enum

"""This file contains functions to load and train models on the different datasets."""


class Model(Enum):
    # 1) Lineare Baselines
    LINEAR_BASELINE = "linear_baseline"
    LINEAR_REGULARIZED = "linear_regularized"

    # 2) Einfache Bäume
    DECISION_TREE = "decision_tree"

    # 4) Baum-Ensembles
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    # 3) Neuronale Netzwerke (PyTorch)
    PYTORCH_NEURAL_NETWORK = "pytorch_neural_network"
    # Config-driven architectures (config-neural-networks-RQ3.yaml); values match
    # PytorchTrainer._ARCH_REGISTRY keys so architecture=self.value dispatches.
    MLP = "mlp"
    TRANSFORMER = "transformer"
    CNN_1D = "cnn_1d"

    @property
    def is_nn(self) -> bool:
        return self in {
            Model.PYTORCH_NEURAL_NETWORK,
            Model.MLP,
            Model.TRANSFORMER,
            Model.CNN_1D,
        }

    @property
    def is_tree(self) -> bool:
        return self in {
            Model.DECISION_TREE,
            Model.RANDOM_FOREST,
            Model.GRADIENT_BOOSTING,
            Model.XGBOOST,
            Model.LIGHTGBM,
        }

    def get_model_with_params(self, dataset: Dataset, params: dict, seed: int):
        is_reg = dataset.is_regression
        # random_state for every estimator comes from the benchmark seed (config.yaml ->
        # benchmark.seed); seed is required so it can never silently diverge from config.
        base = {'random_state': seed}

        if self == Model.LINEAR_BASELINE:
            if is_reg:
                model = LinearRegression()
            else:
                model = LogisticRegression(
                    **{**base, 'max_iter': 1000, **params})
            return SklearnTrainer(model)

        elif self == Model.LINEAR_REGULARIZED:
            if is_reg:
                # alpha is used by Ridge; C and other LogisticRegression-only params are
                # silently dropped by filtering against Ridge's accepted parameter names.
                valid = set(Ridge().get_params())
                filtered = {k: v for k, v in params.items() if k in valid}
                model = Ridge(**{**base, **filtered})
            else:
                # C is used by LogisticRegression; alpha and other Ridge-only params are
                # silently dropped by filtering against LogisticRegression's accepted parameter names.
                valid = set(LogisticRegression().get_params())
                filtered = {k: v for k, v in params.items() if k in valid}
                # C=0.1 is the built-in "regularized" default (distinguishes this
                # from LINEAR_BASELINE's C=1.0); config-supplied C overrides it.
                model = LogisticRegression(
                    **{**base, 'max_iter': 1000, 'C': 0.1, **filtered})
            return SklearnTrainer(model)

        elif self == Model.DECISION_TREE:
            if is_reg:
                model = DecisionTreeRegressor(**{**base, **params})
            else:
                model = DecisionTreeClassifier(**{**base, **params})
            return SklearnTrainer(model)

        elif self == Model.RANDOM_FOREST:
            if is_reg:
                model = RandomForestRegressor(**{**base, **params})
            else:
                model = RandomForestClassifier(**{**base, **params})
            return SklearnTrainer(model)

        elif self == Model.GRADIENT_BOOSTING:
            if is_reg:
                model = HistGradientBoostingRegressor(**{**base, **params})
            else:
                model = HistGradientBoostingClassifier(**{**base, **params})
            return SklearnTrainer(model)

        elif self == Model.XGBOOST:
            from xgboost import XGBRegressor, XGBClassifier
            base_xgb = {**base, 'enable_categorical': False}
            if is_reg:
                model = XGBRegressor(**{**base_xgb, **params})
            else:
                model = XGBClassifier(**{**base_xgb, **params})
            return SklearnTrainer(model)

        elif self == Model.LIGHTGBM:
            from lightgbm import LGBMRegressor, LGBMClassifier
            base_lgbm = {**base, 'verbosity': -1}
            if is_reg:
                model = LGBMRegressor(**{**base_lgbm, **params})
            else:
                model = LGBMClassifier(**{**base_lgbm, **params})
            return SklearnTrainer(model)

        elif self in (Model.MLP, Model.TRANSFORMER, Model.CNN_1D):
            # PytorchTrainer seeds torch itself (no sklearn random_state);
            # epochs/lr/batch_size/device are named kwargs, architecture-specific
            # keys (hidden_sizes, d_model, ...) flow through **arch_kwargs.
            return PytorchTrainer(architecture=self.value, seed=seed, **params)

        else:
            raise ValueError(
                f"Model {self.value} does not support hyperparameter config")

    @staticmethod
    def get_models(pytorch: bool = False) -> list:
        if pytorch:
            return [Model.PYTORCH_NEURAL_NETWORK]
        else:
            return [model for model in Model if not model.is_nn]
