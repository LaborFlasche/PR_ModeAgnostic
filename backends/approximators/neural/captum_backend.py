import pandas as pd
import torch
import torch.nn as nn

from ...base_backend import BaseBackend, nan_result

try:
    from captum.attr import GradientShap, DeepLiftShap
    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False


def _unwrap_torch_model(model):
    from benchmarking.eval_counter import CountingModel
    if isinstance(model, CountingModel):
        return model._model
    return model


class CaptumApproxBackend(BaseBackend):
    """Captum-based Shapley value approximation for PyTorch neural networks.

    Both methods use the background data as a baseline distribution, matching the
    marginal / interventional value function used by other backends.

    config:
        approximator: "gradient_shap" (default) or "deep_lift_shap"
        n_eval: (GradientShap only) random samples per baseline, taken from
            benchmark.n_eval; falls back to 5 if n_eval isn't set
        stdevs: (GradientShap only) Gaussian noise std added to inputs, default 0.0
        target: output neuron index to explain (required for multi-output models)
    """

    name = "captum_approx"
    library = "captum"
    computation_type = "approximation"

    SUPPORTED_APPROXIMATORS = ("gradient_shap", "deep_lift_shap")

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        if not _CAPTUM_AVAILABLE:
            raise ImportError("captum is required: pip install captum")

        torch_model = _unwrap_torch_model(self.model)
        if not isinstance(torch_model, nn.Module):
            return nan_result(x)

        approximator = self.config.get("approximator", "gradient_shap")
        if approximator not in self.SUPPORTED_APPROXIMATORS:
            raise ValueError(
                f"Unknown captum approximator '{approximator}' "
                f"(use {self.SUPPORTED_APPROXIMATORS})"
            )

        target = self.config.get("target")
        torch_model.eval()

        device = next(torch_model.parameters()).device
        x_tensor = torch.tensor(x.values, dtype=torch.float32, device=device)
        baselines = torch.tensor(
            self.background.values, dtype=torch.float32, device=device,
        )

        if approximator == "gradient_shap":
            attr = self._gradient_shap(torch_model, x_tensor, baselines, target)
        else:
            try:
                attr = self._deep_lift_shap(torch_model, x_tensor, baselines, target)
            except Exception:
                return nan_result(x)

        values = attr.detach().cpu().numpy()
        return pd.DataFrame(values, index=x.index, columns=x.columns)

    def _gradient_shap(self, model, x, baselines, target):
        n_samples = self.config.get("n_eval", 5)
        stdevs = self.config.get("stdevs", 0.0)
        explainer = GradientShap(model)
        return explainer.attribute(
            x, baselines=baselines, n_samples=n_samples,
            stdevs=stdevs, target=target,
        )

    def _deep_lift_shap(self, model, x, baselines, target):
        explainer = DeepLiftShap(model)
        return explainer.attribute(x, baselines=baselines, target=target)
