from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchPredictor(nn.Module):
    """Wraps a raw ``nn.Module`` with a unified scalar output for Shapley explanation.

    All methods (gradient-based and model-agnostic) explain the same scalar function:

    * Regression       → raw output (squeezed to 1-D).
    * Classification   → ``P(class=target_class)`` via softmax.
                         Binary: target_class=1, multiclass: target_class=0.
                         Matches ``marginal_predict`` and ``ShapIQTrueValueBackend``.

    This makes gradient-based attributions (GradientShap) and model-agnostic
    attributions (KernelShap, ShapIQ) directly comparable without needing a
    ``target`` neuron argument in captum.

    Parameters
    ----------
    model : nn.Module
        The trained PyTorch model (outputs raw logits).
    task : str
        ``"regression"`` or ``"classification"``.
    n_classes : int
        Number of output classes (1 for regression). Determines ``target_class``.
    """

    def __init__(self, model: nn.Module, task: str = "regression", n_classes: int = 1):
        super().__init__()
        self.model = model
        self.task = task
        self.n_classes = n_classes
        self._target = 1 if n_classes == 2 else 0

    # ------------------------------------------------------------------
    # nn.Module interface (gradient-based backends)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar per sample so gradient backends need no ``target``."""
        logits = self.model(x)
        if self.task == "classification":
            return torch.softmax(logits, dim=-1)[:, self._target]
        return logits.squeeze(-1)

    # ------------------------------------------------------------------
    # sklearn-compatible interface (model-agnostic backends)
    # ------------------------------------------------------------------

    def _to_tensor(self, X) -> torch.Tensor:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return X.float().to(device)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        """Return the scalar output as a 1-D numpy array."""
        self.model.eval()
        return self(self._to_tensor(X)).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """Return (n, 2) array of ``[1 - p, p]`` where ``p = forward(X)``.

        ``marginal_predict`` selects ``[:, 1]`` for ``shape[1] == 2``, so it
        receives ``P(target_class)`` — the same scalar as ``forward()``.
        """
        self.model.eval()
        p = self(self._to_tensor(X)).cpu().numpy()
        return np.stack([1 - p, p], axis=1)


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
    """Config-driven PyTorch trainer supporting multiple architectures.

    Parameters
    ----------
    architecture : str
        One of ``"mlp"``, ``"transformer"``, ``"cnn_1d"``.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam.
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed for reproducibility.
    **arch_kwargs
        Extra keyword arguments forwarded to the selected architecture.
    """

    # Mapping from architecture name to (class name, accepted kwargs)
    _ARCH_REGISTRY: dict[str, tuple[str, set[str]]] = {
        "mlp":         ("TabularMLP",         {"hidden_sizes"}),
        "transformer": ("TabularTransformer", {"d_model", "nhead", "num_layers"}),
        "cnn_1d":      ("TabularCNN1D",       {"n_filters", "kernel_size"}),
    }

    def __init__(
        self,
        architecture: str = "mlp",
        *,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
        device: str = "cpu",
        **arch_kwargs,
    ):
        if architecture not in self._ARCH_REGISTRY:
            raise ValueError(
                f"Unknown architecture {architecture!r}. "
                f"Choose from {list(self._ARCH_REGISTRY)}"
            )
        self._architecture = architecture
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._seed = seed
        self._device = torch.device(device)
        self._arch_kwargs = arch_kwargs
        self._model: TorchPredictor | None = None

    def _build_module(self, in_features: int, out_features: int) -> nn.Module:
        """Instantiate the architecture-specific ``nn.Module``."""
        from Models.architectures import TabularMLP, TabularTransformer, TabularCNN1D

        class_name, accepted = self._ARCH_REGISTRY[self._architecture]
        cls = {"TabularMLP": TabularMLP,
               "TabularTransformer": TabularTransformer,
               "TabularCNN1D": TabularCNN1D}[class_name]
        filtered = {k: v for k, v in self._arch_kwargs.items() if k in accepted}
        return cls(in_features, out_features, **filtered)

    def fit(self, X, y, task: str = "regression"):
        torch.manual_seed(self._seed)

        X_np = X.values if hasattr(X, "values") else np.array(X)
        y_np = y.values if hasattr(y, "values") else np.array(y)

        in_features = X_np.shape[1]
        X_t = torch.tensor(X_np, dtype=torch.float32)

        if task == "classification":
            # Remap labels to canonical 0..n_classes-1 like SklearnTrainer: a plain
            # min-shift breaks non-consecutive labels (gisette's -1/1 became 0/2 →
            # a 3-class head with a dead class, and TorchPredictor then explained
            # class 0 = the negative class). Sorted-order remap keeps class 1 =
            # the higher original label, matching predict_proba[:, 1] downstream.
            y_np = np.unique(y_np, return_inverse=True)[1]
            out_features = int(y_np.max()) + 1
            y_t = torch.tensor(y_np, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
        else:
            out_features = 1
            y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
            loss_fn = nn.MSELoss()

        module = self._build_module(in_features, out_features).to(self._device)
        X_t = X_t.to(self._device)
        y_t = y_t.to(self._device)

        optimizer = torch.optim.Adam(module.parameters(), lr=self._lr)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self._batch_size, shuffle=True
        )

        module.train()
        for _ in range(self._epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss_fn(module(X_batch), y_batch).backward()
                optimizer.step()

        self._model = TorchPredictor(module, task=task, n_classes=out_features)
        return self

    def get_model(self) -> TorchPredictor:
        return self._model
