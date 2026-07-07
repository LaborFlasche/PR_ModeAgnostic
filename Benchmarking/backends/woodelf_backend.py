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


def _woodelf_multiclass_unsupported(model) -> bool:
    """True when woodelf's multiclass output is confirmed wrong for this model.

    woodelf never raises for a >2-class model — it silently always explains
    "class 0" (no error, no class-selection option), so this can't be caught by
    a try/except; it has to be predicted upfront. Empirically checked on toy
    3-class xgboost/lightgbm/sklearn models against every class of the shap
    oracle, both signs:
      - sklearn-native classifiers: EXACT match (mean_abs_diff 0.0) — woodelf's
        class-0 pick happens to equal ``reduce_multiclass``'s own "multiclass ->
        class 0" convention (the same one every other backend uses). Safe to run.
      - xgboost and lightgbm: best case (any class, either sign) is still
        >100% relative error vs. the oracle's own scale — a genuine upstream
        computation bug for boosting models specifically, not a fixable
        index/sign mismatch like ``_woodelf_class_sign_is_flipped`` below. Must
        skip both (not just xgboost — lightgbm is equally broken, despite the
        old ``.objective`` string check only ever having caught xgboost).

    Binary classifiers (<=2 classes) are unaffected by this function either way
    — see ``_woodelf_class_sign_is_flipped`` for their separate, fixable issue.
    """
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) <= 2:
        return False
    return not type(model).__module__.startswith("sklearn")


def _woodelf_class_sign_is_flipped(model) -> bool:
    """True for sklearn-native binary classifiers, where woodelf's values come out
    sign-flipped relative to every other backend's class-1 convention.

    Root cause (upstream, in the installed woodelf package, not this repo): woodelf
    parses trees by reusing shap's own loader (parse_models.py: "Use the shap
    package's Decision Tree loading. this is cheating, I know...") and then takes
    ``tree.values[index][0]`` as the leaf value (parse_models.py's
    ``load_decision_tree``). For sklearn-native classifiers, shap's ``SingleTree``
    reshapes sklearn's per-node ``(1, n_classes)`` value array into ``n_classes``
    columns (shap/explainers/_tree.py's ``SingleTree.__init__``), so for binary
    classification index 0 is class 0's probability — not class 1, the convention
    used everywhere else here (``ShapTrueValueBackend``, ``marginal_predict``,
    ``reduce_multiclass``). Since prob(class 0) = 1 - prob(class 1) and Shapley
    values are linear in the value function, this is an exact sign flip.
    Regressors are unaffected (a single output column, so index 0 is correct);
    xgboost/lightgbm are unaffected (native leaf values, not sklearn's
    ``tree.value``, so this reshape never applies to them).
    """
    return (
        type(model).__module__.startswith("sklearn")
        and hasattr(model, "classes_")
        and len(model.classes_) == 2
    )


class _WoodelfBackend(BaseBackend):
    """woodelf's tree-specific explainer. xgboost/lightgbm multiclass models are
    unsupported (confirmed upstream bug, not a fixable index/sign mismatch —
    see _woodelf_multiclass_unsupported) and skip to an all-NaN frame; sklearn-
    native multiclass classifiers work correctly and are not skipped."""

    library = "woodelf"
    computation_type = "true_value"
    interventional: bool
    gpu: bool = False

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        if _woodelf_multiclass_unsupported(self.model):
            print(f"  [SKIP] {self.name}: woodelf's xgboost/lightgbm multiclass output "
                  "doesn't match any class of the oracle (confirmed upstream bug)")
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

        reduced = reduce_multiclass(values)
        if _woodelf_class_sign_is_flipped(self.model):
            reduced = -reduced
        return pd.DataFrame(reduced, index=x.index, columns=x.columns)


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
        if _woodelf_multiclass_unsupported(self.model):
            print(f"  [SKIP] {self.name}: woodelf's xgboost/lightgbm multiclass output "
                  "doesn't match any class of the oracle (confirmed upstream bug)")
            return nan_interaction_result(x)

        try:
            explainer = WoodelfExplainer(self.model, None, GPU=False)
            values = explainer.shap_interaction_values(x)
        except Exception as e:
            print(f"  [BUG] {self.name} could not run on this model: {e.__class__.__name__}: {e}")
            return nan_interaction_result(x)

        reduced = reduce_multiclass(values, order=2)
        if _woodelf_class_sign_is_flipped(self.model):
            reduced = -reduced
        return flatten_interactions(reduced, x)
