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
    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, X, y, task: str = "regression"):
        if task == "classification":
            # xgboost requires labels in canonical 0..n_classes-1 form (e.g. gisette's
            # -1/1 raises error); other sklearn classifiers don't care. Sorted-order remap
            # preserves which class lands in column 1 downstream (base_backend.py).
            y = np.unique(y, return_inverse=True)[1]
        self._estimator.fit(X, y)
        return self

    def get_model(self):
        return self._estimator


class PytorchTrainer(ModelTrainer):
    _EPOCHS = 2       # full training: 20
    _LR = 1e-3
    _BATCH_SIZE = 64
    _HIDDEN_SIZE = 64

    def __init__(self):
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

        self._model = nn.Sequential(
            nn.Linear(in_features, self._HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(self._HIDDEN_SIZE, out_features),
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._LR)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self._BATCH_SIZE, shuffle=True
        )

        self._model.train()
        for _ in range(self._EPOCHS):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss_fn(self._model(X_batch), y_batch).backward()
                optimizer.step()

        return self

    def get_model(self):
        return self._model
