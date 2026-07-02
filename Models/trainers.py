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

    Standardization lives *inside* this wrapper (``x_mean``/``x_std`` buffers,
    plus ``y_mean``/``y_std`` for regression): the wrapped module is trained on
    z-scored data (raw scales like adult's fnlwgt ~1e5 overflow the transformer
    attention to NaN), while every caller — gradient and model-agnostic alike —
    keeps operating in the original feature/output space.

    Parameters
    ----------
    model : nn.Module
        The trained PyTorch model (outputs raw logits, expects z-scored input).
    task : str
        ``"regression"`` or ``"classification"``.
    n_classes : int
        Number of output classes (1 for regression). Determines ``target_class``.
    x_mean, x_std : torch.Tensor | None
        Per-feature standardization applied before the module (identity if None).
    y_mean, y_std : float
        Target de-standardization applied to regression output.
    """

    # Rows per forward pass in predict()/predict_proba(): model-agnostic
    # explainers hand over 1e5-1e6 coalition-imputed rows per call, and a
    # single unchunked pass OOMs the cluster's 7.6 GB GPUs (transformer
    # attention is quadratic in tokens: 1024 rows x 4 heads x 256^2 ~ 1 GB).
    PREDICT_CHUNK = 1024

    def __init__(self, model: nn.Module, task: str = "regression", n_classes: int = 1,
                 x_mean: torch.Tensor | None = None, x_std: torch.Tensor | None = None,
                 y_mean: float = 0.0, y_std: float = 1.0):
        super().__init__()
        self.model = model
        self.task = task
        self.n_classes = n_classes
        self._target = 1 if n_classes == 2 else 0
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.y_mean = y_mean
        self.y_std = y_std

    # ------------------------------------------------------------------
    # nn.Module interface (gradient-based backends)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar per sample so gradient backends need no ``target``."""
        if self.x_mean is not None:
            x = (x - self.x_mean) / self.x_std
        logits = self.model(x)
        if self.task == "classification":
            return torch.softmax(logits, dim=-1)[:, self._target]
        return logits.squeeze(-1) * self.y_std + self.y_mean

    # ------------------------------------------------------------------
    # sklearn-compatible interface (model-agnostic backends)
    # ------------------------------------------------------------------

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @torch.no_grad()
    def _predict_scalar(self, X) -> np.ndarray:
        """Chunked scoring: keep the full batch on CPU, move PREDICT_CHUNK rows
        to the model device at a time (see PREDICT_CHUNK for why)."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.float()
        self.model.eval()
        device = self._device()
        outs = [
            self(X[i:i + self.PREDICT_CHUNK].to(device)).cpu()
            for i in range(0, len(X), self.PREDICT_CHUNK)
        ]
        return torch.cat(outs).numpy()

    def predict(self, X) -> np.ndarray:
        """Return the scalar output as a 1-D numpy array."""
        return self._predict_scalar(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return (n, 2) array of ``[1 - p, p]`` where ``p = forward(X)``.

        ``marginal_predict`` selects ``[:, 1]`` for ``shape[1] == 2``, so it
        receives ``P(target_class)`` — the same scalar as ``forward()``.
        """
        p = self._predict_scalar(X)
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

        # z-score features for training (raw scales like adult's fnlwgt ~1e5
        # overflow the transformer attention to NaN); the stats live in
        # TorchPredictor so callers keep the original feature space.
        x_mean = X_np.astype(np.float64).mean(axis=0)
        x_std = X_np.astype(np.float64).std(axis=0)
        x_std[x_std < 1e-12] = 1.0  # constant columns: identity, not div-by-zero
        X_t = torch.tensor((X_np - x_mean) / x_std, dtype=torch.float32)

        y_mean, y_std = 0.0, 1.0
        if task == "classification":
            # Remap labels to canonical 0..n_classes-1 (Covertype uses 1-7,
            # gisette -1/1 — a plain min-shift would leave gaps and inflate the
            # class count). Sorted-order remap, matching SklearnTrainer.
            y_np = np.unique(y_np, return_inverse=True)[1]
            out_features = int(y_np.max()) + 1
            y_t = torch.tensor(y_np, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
        else:
            # Standardize regression targets too (ames ~2e5) so MSE gradients
            # are well-conditioned; TorchPredictor de-standardizes the output.
            y_mean = float(y_np.mean())
            y_std = float(y_np.std()) or 1.0
            out_features = 1
            y_t = torch.tensor((y_np - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
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

        self._model = TorchPredictor(
            module, task=task, n_classes=out_features,
            x_mean=torch.tensor(x_mean, dtype=torch.float32, device=self._device),
            x_std=torch.tensor(x_std, dtype=torch.float32, device=self._device),
            y_mean=y_mean, y_std=y_std,
        )
        return self

    def get_model(self) -> TorchPredictor:
        return self._model
