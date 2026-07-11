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
# see benchmarking/backends/true_value/trees/shapiq_backend.py.
from datasets.load_datasets import Dataset

from models.trainers import SklearnTrainer, PytorchTrainer
from enum import Enum


def actual_max_depth(model) -> int:
    """Deepest root-to-leaf path actually grown in a fitted tree model/ensemble.

    The config's ``max_depth`` is only an upper bound the training may not
    reach (or may be null = unlimited); results CSVs report this realized
    depth instead. Dispatches on the fitted estimator via duck typing, so it
    never imports xgboost/lightgbm itself (see the import-order note above).
    """
    if hasattr(model, "tree_"):  # DecisionTree{Regressor,Classifier}
        return int(model.get_depth())
    if hasattr(model, "estimators_"):  # RandomForest{Regressor,Classifier}
        return max(int(e.get_depth()) for e in model.estimators_)
    if hasattr(model, "_predictors"):  # HistGradientBoosting{Regressor,Classifier}
        return max(int(p.get_max_depth())
                   for stage in model._predictors for p in stage)
    if hasattr(model, "get_booster"):  # XGB{Regressor,Classifier}
        import json

        def depth(node):
            if "children" not in node:
                return 0
            return 1 + max(depth(c) for c in node["children"])

        return max(depth(json.loads(t))
                   for t in model.get_booster().get_dump(dump_format="json"))
    if hasattr(model, "booster_"):  # LGBM{Regressor,Classifier}
        # node_depth is 1 at the root, so subtract 1 to count edges like the rest.
        trees = model.booster_.trees_to_dataframe()
        return int(trees["node_depth"].max()) - 1
    raise TypeError(
        f"actual_max_depth: unsupported model type {type(model).__name__}")


class Model(Enum):
    # 1) Linear baselines
    LINEAR_BASELINE = "linear_baseline"
    LINEAR_REGULARIZED = "linear_regularized"

    # 2) Single decision tree
    DECISION_TREE = "decision_tree"

    # 3) Tree ensembles
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    # 4) Neural networks (PyTorch)
    # Config-driven architectures (config-neural-networks.yaml); values match
    # PytorchTrainer._ARCH_REGISTRY keys so architecture=self.value dispatches.
    MLP = "mlp"
    TRANSFORMER = "transformer"
    CNN_1D = "cnn_1d"

    @property
    def is_nn(self) -> bool:
        return self in {
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
            return [model for model in Model if model.is_nn]
        else:
            return [model for model in Model if not model.is_nn]
