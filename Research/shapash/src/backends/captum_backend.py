"""
Captum backend (PyTorch models only).

Captum is Pytorch's official interpretability library.  Unlike SHAP/ShapIQ it
requires a torch.nn.Module and works on tensors, not DataFrames.

Install: pip install captum torch
Docs:    https://captum.ai
"""

import numpy as np
import pandas as pd

from .base_backend import BaseBackend

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from captum.attr import IntegratedGradients, GradientShap, DeepLift, Saliency
    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False


# Maps method name strings to Captum classes
_METHOD_MAP = {
    "integrated_gradients": lambda m: IntegratedGradients(m),
    "gradient_shap": lambda m: GradientShap(m),
    "deep_lift": lambda m: DeepLift(m),
    "saliency": lambda m: Saliency(m),
}


class CaptumBackend(BaseBackend):
    """Attribution backend using Captum (PyTorch models).

    Computes attributions for tabular data via gradient-based methods.
    All methods return attributions of shape (n_samples, n_features) that
    can be compared directly with SHAP values.

    Supported methods
    -----------------
    - 'integrated_gradients' (default) — approximates Shapley values via path integrals
    - 'gradient_shap'  — combines gradients with SHAP (requires baseline distribution)
    - 'deep_lift'      — compares to a reference baseline
    - 'saliency'       — plain input gradients (fastest, least faithful to Shapley axioms)
    """

    name = "captum"

    def __init__(
        self,
        model,
        method: str = "integrated_gradients",
        baseline: pd.DataFrame | None = None,
        target: int | None = None,
        n_steps: int = 50,
        dtype=None,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch model.
        method : str, default 'integrated_gradients'
            Attribution method to use.
        baseline : pd.DataFrame, optional
            Reference input (zeros if None).  Used by IG, GradientShap, DeepLIFT.
        target : int, optional
            Output index (class index) to explain.  None → uses index 0.
        n_steps : int, default 50
            Integration steps for IntegratedGradients.
        dtype : torch.dtype, optional
            Tensor dtype, defaults to torch.float32.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is not installed. Run: pip install torch")
        if not _CAPTUM_AVAILABLE:
            raise ImportError("captum is not installed. Run: pip install captum")
        if method not in _METHOD_MAP:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(_METHOD_MAP)}")

        super().__init__(model)
        self.method = method
        self.baseline = baseline
        self.target = target
        self.n_steps = n_steps
        self.dtype = dtype or torch.float32
        self._attr = _METHOD_MAP[method](model)

    def _to_tensor(self, df: pd.DataFrame) -> "torch.Tensor":
        return torch.tensor(df.values, dtype=self.dtype)

    def _get_baseline(self, x: pd.DataFrame) -> "torch.Tensor":
        if self.baseline is not None:
            return self._to_tensor(self.baseline)
        # Default: zero baseline (same as IG convention)
        return torch.zeros(1, x.shape[1], dtype=self.dtype)

    def run_explainer(self, x: pd.DataFrame) -> dict:
        x_tensor = self._to_tensor(x)
        baseline_tensor = self._get_baseline(x)
        target = self.target  # None or int

        kwargs = {}
        if self.method == "integrated_gradients":
            kwargs["baselines"] = baseline_tensor
            kwargs["n_steps"] = self.n_steps
            kwargs["return_convergence_delta"] = False
        elif self.method == "gradient_shap":
            # GradientShap expects a baseline distribution (multiple rows)
            kwargs["baselines"] = baseline_tensor
            kwargs["n_samples"] = self.n_steps
        elif self.method == "deep_lift":
            kwargs["baselines"] = baseline_tensor

        if target is not None:
            kwargs["target"] = target

        self.model.eval()
        with torch.no_grad() if self.method == "saliency" else torch.enable_grad():
            if self.method == "saliency":
                # Saliency needs gradients enabled
                x_tensor.requires_grad_(True)
                attr = self._attr.attribute(x_tensor, target=target)
            else:
                attr = self._attr.attribute(x_tensor, **kwargs)

        values = attr.detach().cpu().numpy()
        contributions = pd.DataFrame(values, index=x.index, columns=x.columns)
        return {"contributions": contributions, "method": self.method}
