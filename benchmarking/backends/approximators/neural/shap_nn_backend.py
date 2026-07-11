import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from ...base_backend import BaseBackend, nan_result


def _unwrap_torch_model(model):
    from ...eval_counter import CountingModel
    if isinstance(model, CountingModel):
        return model._model
    return model


class _Unsqueeze(nn.Module):
    """Ensures model output is (batch, 1) for SHAP gradient explainers.

    TorchPredictor.forward() returns a 1-D scalar (batch,) so that captum
    needs no ``target``. SHAP's GradientExplainer/DeepExplainer internally
    index with ``outputs[:, idx]`` and require a 2-D output — this wrapper
    adds the trailing dimension without changing the gradient graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.unsqueeze(-1) if out.dim() == 1 else out


class ShapNNApproxBackend(BaseBackend):
    """SHAP's gradient-based explainers for PyTorch neural networks.

    Both methods use the background data as reference and compute attributions
    via backpropagation through the model, so they only work with
    torch.nn.Module models.

    config:
        approximator: "gradient" (GradientExplainer) or "deep" (DeepExplainer)
        target: output neuron index to explain (required for multi-output models)
    """

    name = "shap_nn_approx"
    library = "shap"
    computation_type = "approximation"

    SUPPORTED_APPROXIMATORS = ("gradient", "deep")

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        torch_model = _unwrap_torch_model(self.model)
        if not isinstance(torch_model, nn.Module):
            return nan_result(x)

        approximator = self.config.get("approximator", "gradient")
        if approximator not in self.SUPPORTED_APPROXIMATORS:
            raise ValueError(
                f"Unknown shap_nn approximator '{approximator}' "
                f"(use {self.SUPPORTED_APPROXIMATORS})"
            )

        torch_model.eval()

        device = next(torch_model.parameters()).device
        x_tensor = torch.tensor(x.values, dtype=torch.float32, device=device)
        baselines = torch.tensor(
            self.background.values, dtype=torch.float32, device=device,
        )

        # SHAP gradient explainers require (batch, n_outputs) — wrap scalar output
        shap_model = _Unsqueeze(torch_model)

        if approximator == "gradient":
            explainer = shap.GradientExplainer(shap_model, baselines)
            values = explainer.shap_values(x_tensor)
        else:
            try:
                explainer = shap.DeepExplainer(shap_model, baselines)
                values = explainer.shap_values(x_tensor)
            except Exception:
                return nan_result(x)

        # single output → SHAP returns list of length 1
        if isinstance(values, list):
            values = values[0]
        # _Unsqueeze adds a trailing output dim → attributions are (n, d, 1), squeeze to (n, d)
        if isinstance(values, np.ndarray) and values.ndim == 3 and values.shape[-1] == 1:
            values = values[..., 0]

        return pd.DataFrame(values, index=x.index, columns=x.columns)
