import pandas as pd
from woodelf import WoodelfExplainer

from .base_backend import (
    BaseBackend,
    nan_result,
    nan_interaction_result,
    reduce_multiclass,
    flatten_interactions,
    cuda_available,
)


def _woodelf_gpu_ok() -> tuple[bool, str]:
    """(ok, skip_reason). woodelf's GPU=True needs both cupy and a CUDA device."""
    if not cuda_available():
        return False, "no CUDA device"
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False, "cupy not installed"
    return True, ""


class _WoodelfBackend(BaseBackend):
    """woodelf's tree-specific explainer. woodelf 0.4.3 doesn't support
    multiclass models (upstream limitation) — skips with an all-NaN frame."""

    library = "woodelf"
    computation_type = "true_value"
    interventional: bool
    gpu: bool = False

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        objective = str(getattr(self.model, "objective", "") or "")
        if "multi" in objective:
            print(f"  [SKIP] {self.name}: woodelf does not support multiclass models (upstream TODO)")
            return nan_result(x)

        if self.gpu:
            ok, reason = _woodelf_gpu_ok()
            if not ok:
                print(f"  [SKIP] {self.name}: {reason}")
                return nan_result(x)

        background = self.background if self.interventional else None
        try:
            explainer = WoodelfExplainer(self.model, background, GPU=self.gpu)
            values = explainer.shap_values(x)
        except Exception as e:
            print(f"  [BUG] {self.name} could not run on this model: {e.__class__.__name__}: {e}")
            return nan_result(x)

        return pd.DataFrame(reduce_multiclass(values), index=x.index, columns=x.columns)


class WoodelfTreePathDependentBackend(_WoodelfBackend):
    name = "woodelf_path_dependent"
    interventional = False


class WoodelfTreeInterventionalBackend(_WoodelfBackend):
    name = "woodelf_interventional"
    interventional = True


class WoodelfGPUPathDependentBackend(_WoodelfBackend):
    """GPU=True (cupy-backed) version of WoodelfTreePathDependentBackend.
    Unverified on real GPU hardware."""

    name = "woodelf_gpu_path_dependent"
    interventional = False
    gpu = True


class WoodelfGPUInterventionalBackend(_WoodelfBackend):
    """GPU=True version of WoodelfTreeInterventionalBackend. Same caveat."""

    name = "woodelf_gpu_interventional"
    interventional = True
    gpu = True


class WoodelfInteractionBackend(BaseBackend):
    """Pairwise (order-2) woodelf interactions, path-dependent only (no background)."""

    name = "woodelf_interaction"
    library = "woodelf"
    computation_type = "true_value"
    order = 2

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        objective = str(getattr(self.model, "objective", "") or "")
        if "multi" in objective:
            print(f"  [SKIP] {self.name}: woodelf does not support multiclass models (upstream TODO)")
            return nan_interaction_result(x)

        try:
            explainer = WoodelfExplainer(self.model, None, GPU=False)
            values = explainer.shap_interaction_values(x)
        except Exception as e:
            print(f"  [BUG] {self.name} could not run on this model: {e.__class__.__name__}: {e}")
            return nan_interaction_result(x)

        return flatten_interactions(reduce_multiclass(values, order=2), x)
