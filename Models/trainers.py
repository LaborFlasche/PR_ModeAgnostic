from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ModelTrainer(ABC):
    @abstractmethod
    def fit(self, X, y, task: str = "regression"): ...

    @abstractmethod
    def get_model(self): ...


class SklearnTrainer(ModelTrainer):
    "Wrapper to unify sklearn models with the same interface as PytorchTrainer."
    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, X, y, task: str = "regression"):
        self._estimator.fit(X, y)
        return self

    def get_model(self):
        return self._estimator


class PytorchTrainer(ModelTrainer):
    """Pytorch trainer to build and train a NN based on torch with flexible values"""
    _DEFAULT_EPOCHS = 20
    _LR = 1e-3
    _BATCH_SIZE = 64
    _DEFAULT_HIDDEN_DIMS = [64]


    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        epochs: int | None = None,
    ):
        self._hidden_dims = hidden_dims if hidden_dims is not None else self._DEFAULT_HIDDEN_DIMS
        self._epochs = epochs if epochs is not None else self._DEFAULT_EPOCHS
        self._model = None

    def fit(self, X, y, task: str = "regression"):
        X_np = X.values if hasattr(X, "values") else np.array(X)
        y_np = y.values if hasattr(y, "values") else np.array(y)

        in_features = X_np.shape[1]
        X_t = torch.tensor(X_np, dtype=torch.float32)

        if task == "classification":
            # Shift labels to 0-indexed (e.g. Covertype uses 1–7)
            y_np = y_np - y_np.min()
            out_features = int(y_np.max()) + 1
            y_t = torch.tensor(y_np, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
        else:
            out_features = 1
            y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
            loss_fn = nn.MSELoss()
        self.model = self.build_nn(in_features=in_features, out_features=out_features)

        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self._LR)
        self.loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self._BATCH_SIZE, shuffle=True
        )

        self.train_model()

        self._model.eval()
        return self
    
    def build_nn(self, in_features: int, out_features: int) -> nn.Sequential:
        layers = []
        prev = in_features
        for h in self._hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_features))
        return nn.Sequential(*layers)
    
    def train_model(self):
        if self._model is not None:
            self._model.train()
        for _ in range(self._epochs):
            for X_batch, y_batch in self.loader:
                self.optimizer.zero_grad()
                self.loss_fn(self._model(X_batch), y_batch).backward()
                self.optimizer.step()

        

    def get_model(self):
        return self._model
