from Models.dataset_and_models import Dataset, Model


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

    def train(self, seed: int, verbose: bool = False):
        """Load the dataset and train the model.

        ``seed`` is the single source of randomness: it drives both the dataset
        subsampling and every estimator's ``random_state`` (config.yaml ->
        benchmark.seed), so data and model can never diverge.
        """
        data = self.dataset.load_dataset(seed=seed)
        trainer = self.model.get_model_with_params(self.dataset, {}, seed=seed)
        trainer.fit(data["X"], data["y"], task=data["task"])
        if verbose:
            print(f"Trained {self.model.value} on {self.dataset.value}")
        return trainer.get_model()

    @staticmethod
    def get_all_configs(pytorch: bool) -> list:
        """All (dataset, model) combinations, one TrainingConfig each."""
        configs = []
        for dataset in Dataset:
            for model in Model.get_models(pytorch):
                configs.append(TrainingConfig(dataset, model))
        return configs
