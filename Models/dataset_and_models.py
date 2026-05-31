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

    def load_dataset(self, n_samples: int | None = None, n_features: int | None = None) -> dict:
        if self == Dataset.CALIFORNIA_HOUSING:
            return load_california_housing(n_samples=n_samples, n_features=n_features)
        elif self == Dataset.AMES_HOUSING:
            return load_ames_housing(n_samples=n_samples, n_features=n_features)
        elif self == Dataset.COVERTYPE:
            return load_covertype(n_samples=n_samples, n_features=n_features)
        elif self == Dataset.ADULT_CENSUS:
            return load_adult_census(n_samples=n_samples, n_features=n_features)
        elif self == Dataset.GISETTE:
            return load_gisette(n_samples=n_samples, n_features=n_features)
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
    
    # 3) Neuronale Netzwerke (PyTorch)
    PYTORCH_NEURAL_NETWORK = "pytorch_neural_network"
    
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
            
        elif self == Model.PYTORCH_NEURAL_NETWORK:
            return PytorchTrainer()
            
        else:
            raise ValueError(f"Unknown model: {self.value}")

    def get_model_with_params(self, dataset: Dataset, params: dict):
        is_reg = dataset.is_regression
        base = {'random_state': 42}

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

        else:
            raise ValueError(f"Model {self.value} does not support hyperparameter config")

    @staticmethod
    def get_models(pytorch: bool = False) -> list:
        if pytorch:
            return [Model.PYTORCH_NEURAL_NETWORK]
        else:
            return [model for model in Model if model != Model.PYTORCH_NEURAL_NETWORK]