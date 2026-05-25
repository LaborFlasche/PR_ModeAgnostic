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
    
    def get_model(self):
        if self == Model.RANDOM_FOREST_REGRESSOR:
            return RandomForestRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown model: {self.value}")