from sklearn.ensemble import RandomForestRegressor
from Models.dataset_and_models import Dataset, Model

"""This file contains functions to load and train models on the different datasets."""

class TrainingConfig:
    """Configuration for training a model on a dataset.

    Attributes
    ----------
    dataset        : Dataset
    model          : Model
    """
    def __init__(self, dataset: Dataset, model: Model):
        self.dataset = dataset
        self.model = model
    
    def train(self, verbose: bool = False):
        """Load the dataset and train the model."""
        data = self.dataset.load_dataset()
        X, y = data["X"], data["y"]
        model = self.model.get_model()
        model.fit(X, y)
        if verbose:
            print(f"Trained {self.model.value} on {self.dataset.value}")
        return model
    
    @staticmethod
    def get_all_configs() -> list:
        # Get all combinations of datasets and models -> Return a list of TrainingConfig objects
        configs = []
        for dataset in Dataset:
            for model in Model:
                configs.append(TrainingConfig(dataset, model))
        return configs
    

