import pandas as pd
import torch

from captum.attr import DeepLiftShap, GradientShap, KernelShap, ShapleyValueSampling

from .base_backend import BaseBackend, marginal_predict

_GRADIENT_METHODS = frozenset({"gradient_shap", "deep_lift_shap"})
_VALID_METHODS = frozenset({"shapley_value_sampling", "kernel_shap", "gradient_shap", "deep_lift_shap"})


class CaptumBackend(BaseBackend):
    """Captum attribution backends for model-agnostic and gradient-based SHAP variants.

    Model-agnostic methods wrap any sklearn-compatible model via ``marginal_predict``
    (same scalar output space as the oracle) and require no gradients:
        - 'shapley_value_sampling': random permutation sampling (Strumbelj & Kononenko).
          Absent features are replaced by the background mean.
        - 'kernel_shap': weighted-least-squares KernelSHAP over the background
          distribution. Absent features are replaced by samples drawn from ``background``.

    Gradient-based methods require the model to be a ``torch.nn.Module``:
        - 'gradient_shap': gradient × (input − baseline), averaged over randomly
          sampled baseline rows drawn from ``background``.
        - 'deep_lift_shap': DeepLIFT attributions averaged over all background rows.

    config keys
    -----------
    method : str
        One of the four methods above (default: 'shapley_value_sampling').
    budget : int
        Number of permutations / coalition samples / gradient samples (default: 200).
        Ignored by 'deep_lift_shap' (deterministic).
    target : int
        Output neuron index for ``torch.nn.Module`` models (default: 0).
        Ignored for sklearn wrappers — ``marginal_predict`` already returns a scalar.
    """

    name = "captum"
    library = "captum"
    computation_type = "approximation"

    def _sklearn_forward(self, columns):
        """Wrap marginal_predict as a no-grad torch callable."""
        f = marginal_predict(self.model, columns)

        def forward(x_t: torch.Tensor) -> torch.Tensor:
            out = f(x_t.detach().cpu().numpy())
            return torch.tensor(out, dtype=torch.float32)

        return forward

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        method = self.config.get("method", "shapley_value_sampling")
        budget = int(self.config.get("budget", 200))
        target = self.config.get("target", 0)

        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown captum method '{method}'. Choose from: {sorted(_VALID_METHODS)}"
            )

        is_torch_model = isinstance(self.model, torch.nn.Module)
        if method in _GRADIENT_METHODS and not is_torch_model:
            raise ValueError(
                f"Captum method '{method}' requires a torch.nn.Module; "
                f"got {type(self.model).__name__}. "
                "Use 'shapley_value_sampling' or 'kernel_shap' for sklearn models."
            )

        columns = list(x.columns)
        x_t = torch.tensor(x.values, dtype=torch.float32)
        bg_t = torch.tensor(self.background.values, dtype=torch.float32)

        # For torch models pass target; sklearn wrappers return a scalar already.
        if is_torch_model:
            forward_func = self.model
            kw_target = {"target": target}
        else:
            forward_func = self._sklearn_forward(columns)
            kw_target = {}

        if method == "shapley_value_sampling":
            # Single mean baseline: absent features replaced by background mean,
            # matching the "conditional mean" convention used across approximators.
            mean_baseline = bg_t.mean(dim=0, keepdim=True)
            explainer = ShapleyValueSampling(forward_func)
            attributions = explainer.attribute(
                x_t,
                baselines=mean_baseline,
                n_samples=budget,
                perturbations_per_eval=1,
                show_progress=False,
                **kw_target,
            )

        elif method == "kernel_shap":
            # Full background as baseline distribution; Captum averages attributions
            # over all baseline rows, matching the marginal SHAP convention.
            explainer = KernelShap(forward_func)
            attributions = explainer.attribute(
                x_t,
                baselines=bg_t,
                n_samples=budget,
                perturbations_per_eval=1,
                show_progress=False,
                **kw_target,
            )

        elif method == "gradient_shap":
            # Randomly samples `budget` baselines from bg_t, computes
            # gradient × (input − baseline) for each, then averages.
            self.model.eval()
            explainer = GradientShap(forward_func)
            attributions = explainer.attribute(
                x_t,
                baselines=bg_t,
                n_samples=budget,
                **kw_target,
            )

        else:  # deep_lift_shap
            # Deterministic: computes DeepLIFT attributions for every baseline row
            # in bg_t and averages them.
            self.model.eval()
            explainer = DeepLiftShap(forward_func)
            attributions = explainer.attribute(
                x_t,
                baselines=bg_t,
                **kw_target,
            )

        values = attributions.detach().cpu().numpy()
        return pd.DataFrame(values, index=x.index, columns=columns)
