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

from Datasets.load_datasets import (
    load_california_housing,
    load_ames_housing,
    load_covertype,
    load_adult_census,
    load_gisette,
)

from Models.trainers import SklearnTrainer, PytorchTrainer
from enum import Enum

"""This file contains functions to load and train models on the different datasets."""

class Dataset(Enum):
    CALIFORNIA_HOUSING = "california_housing"
    AMES_HOUSING = "ames_housing"
    COVERTYPE = "covertype"
    ADULT_CENSUS = "adult_census"
    GISETTE = "gisette"

    def load_dataset(self, n_samples: int | None = None, n_features: int | None = None,
                     *, seed: int) -> dict:
        if self == Dataset.CALIFORNIA_HOUSING:
            return load_california_housing(n_samples=n_samples, n_features=n_features, seed=seed)
        elif self == Dataset.AMES_HOUSING:
            return load_ames_housing(n_samples=n_samples, n_features=n_features, seed=seed)
        elif self == Dataset.COVERTYPE:
            return load_covertype(n_samples=n_samples, n_features=n_features, seed=seed)
        elif self == Dataset.ADULT_CENSUS:
            return load_adult_census(n_samples=n_samples, n_features=n_features, seed=seed)
        elif self == Dataset.GISETTE:
            return load_gisette(n_samples=n_samples, n_features=n_features, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {self.value}")

    @property
    def is_regression(self) -> bool:
        return self in [Dataset.CALIFORNIA_HOUSING, Dataset.AMES_HOUSING]


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

    @property
    def is_tree(self) -> bool:
        return self in {
            Model.DECISION_TREE,
            Model.RANDOM_FOREST,
            Model.GRADIENT_BOOSTING,
            Model.XGBOOST,
            Model.LIGHTGBM,
        }

    def get_model(self, dataset: Dataset):
        is_reg = dataset.is_regression
        
        if self == Model.LINEAR_BASELINE:
            model = LinearRegression() if is_reg else LogisticRegression(max_iter=1000, random_state=42)
            return SklearnTrainer(model)
            
        elif self == Model.LINEAR_REGULARIZED:
            model = Ridge(random_state=42) if is_reg else LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            return SklearnTrainer(model)
            
        elif self == Model.DECISION_TREE:
            model = DecisionTreeRegressor(random_state=42) if is_reg else DecisionTreeClassifier(random_state=42)
            return SklearnTrainer(model)
            
        elif self == Model.RANDOM_FOREST:
            model = RandomForestRegressor(random_state=42) if is_reg else RandomForestClassifier(random_state=42)
            return SklearnTrainer(model)
            
        elif self == Model.GRADIENT_BOOSTING:
            model = HistGradientBoostingRegressor(random_state=42) if is_reg else HistGradientBoostingClassifier(random_state=42)
            return SklearnTrainer(model)

        elif self == Model.XGBOOST:
            from xgboost import XGBRegressor, XGBClassifier
            # enable_categorical defaults to True in xgboost>=3.0, which makes shap's
            # TreeExplainer reject the model outright.
            model = XGBRegressor(random_state=42, enable_categorical=False) if is_reg else XGBClassifier(random_state=42, enable_categorical=False)
            return SklearnTrainer(model)

        elif self == Model.LIGHTGBM:
            from lightgbm import LGBMRegressor, LGBMClassifier
            model = LGBMRegressor(random_state=42, verbosity=-1) if is_reg else LGBMClassifier(random_state=42, verbosity=-1)
            return SklearnTrainer(model)

        elif self == Model.PYTORCH_NEURAL_NETWORK:
            return PytorchTrainer()
            
        else:
            raise ValueError(f"Unknown model: {self.value}")

    def get_model_with_params(self, dataset: Dataset, params: dict, seed: int):
        is_reg = dataset.is_regression
        # random_state for every estimator comes from the benchmark seed (config.yaml ->
        # benchmark.seed); seed is required so it can never silently diverge from config.
        base = {'random_state': seed}

        if self == Model.LINEAR_BASELINE:
            if is_reg:
                model = LinearRegression()
            else:
                model = LogisticRegression(**{**base, 'max_iter': 1000, **params})
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
                model = LogisticRegression(**{**base, 'max_iter': 1000, **filtered})
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

        else:
            raise ValueError(f"Model {self.value} does not support hyperparameter config")

    @staticmethod
    def get_models(pytorch: bool = False) -> list:
        if pytorch:
            return [Model.PYTORCH_NEURAL_NETWORK]
        else:
            return [model for model in Model if model != Model.PYTORCH_NEURAL_NETWORK]