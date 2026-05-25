from sklearn.ensemble import RandomForestRegressor
from Datasets.load_datasets import load_california_housing, load_ames_housing, load_covertype
from enum import Enum

"""This file contains functions to load and train models on the different datasets."""
class Dataset(Enum):
    CALIFORNIA_HOUSING = "california_housing"
    AMES_HOUSING = "ames_housing"
    COVERTYPE = "covertype"
    def load_dataset(self) -> dict:
        if self == Dataset.CALIFORNIA_HOUSING:
            return load_california_housing()
        elif self == Dataset.AMES_HOUSING:
            return load_ames_housing()
        elif self == Dataset.COVERTYPE:
            return load_covertype()
        else:
            raise ValueError(f"Unknown dataset: {self.value}")
    
class Model(Enum):
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    PYTORCH_NEURAL_NETWORK = "pytorch_neural_network"
    
    def get_model(self):
        from Models.trainers import SklearnTrainer, PytorchTrainer
        if self == Model.RANDOM_FOREST_REGRESSOR:
            return SklearnTrainer(RandomForestRegressor(random_state=42))
        elif self == Model.PYTORCH_NEURAL_NETWORK:
            return PytorchTrainer()
        else:
            raise ValueError(f"Unknown model: {self.value}")
    def get_models(pytorch: bool = False) -> list:
        if pytorch:
            return [Model.PYTORCH_NEURAL_NETWORK]
        else:
            return [model for model in Model if model != Model.PYTORCH_NEURAL_NETWORK]
    