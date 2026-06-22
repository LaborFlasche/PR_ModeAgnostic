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
    """(ok, skip_reason). woodelf's GPU=True path needs cupy *and* an actual CUDA
    device — cupy import alone doesn't guarantee a device is visible, and a CUDA
    device alone is useless without cupy installed, so both are checked.
    """
    if not cuda_available():
        return False, "no CUDA device"
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False, "cupy not installed"
    return True, ""


class _WoodelfBackend(BaseBackend):
    """Shared logic for woodelf's tree-specific explainer.

    woodelf 0.4.3 does not support multiclass models (upstream TODO, per
    TreeSHAPBench's own guard in benchmark_utils.py:run_woodelf) — detected via the
    model's ``objective`` string rather than ``n_classes_``, which may not exist
    pre-fit. On that condition, or any model-load failure, return an all-NaN frame
    instead of raising so one incompatible (model, dataset) cell doesn't crash the
    whole sweep.
    """

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
    """Same as ``WoodelfTreePathDependentBackend`` but GPU=True (cupy-backed).
    Skips (NaN) when no CUDA device or cupy is unavailable — unverified on real
    GPU hardware, since this machine has neither.
    """

    name = "woodelf_gpu_path_dependent"
    interventional = False
    gpu = True


class WoodelfGPUInterventionalBackend(_WoodelfBackend):
    """Same as ``WoodelfTreeInterventionalBackend`` but GPU=True. Same caveats as
    ``WoodelfGPUPathDependentBackend``."""

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
