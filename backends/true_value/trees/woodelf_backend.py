import pandas as pd
import shap
from woodelf import WoodelfExplainer

from ...base_backend import (
    BaseBackend,
    nan_result,
    nan_interaction_result,
    reduce_multiclass,
    flatten_interactions,
    cuda_available,
    select_base_value,
)


def _path_dependent_base_value(model) -> float:
    """Base value of the path-dependent game woodelf explains. woodelf never
    reports one, but the game is identical to shap's path-dependent
    TreeExplainer (values agree to ~1e-8), so shap's expected_value is exact.
    select_base_value's binary -> class 1 pick matches the sign-flip
    correction below."""
    return select_base_value(shap.TreeExplainer(model).expected_value)


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
    "class 0", so this must be predicted upfront, not caught via try/except.
    Empirically verified on toy 3-class models against every class of the shap
    oracle, both signs: sklearn-native classifiers match exactly (woodelf's
    class-0 pick equals ``reduce_multiclass``'s own convention, safe to run);
    xgboost and lightgbm are both a genuine upstream computation bug (>100%
    relative error vs. any class/sign) and must be skipped — not a fixable
    index/sign issue like ``_woodelf_class_sign_is_flipped`` below.

    Binary classifiers (<=2 classes) are unaffected either way.
    """
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) <= 2:
        return False
    return not type(model).__module__.startswith("sklearn")


def _woodelf_class_sign_is_flipped(model) -> bool:
    """True for sklearn binary classifiers with probability leaves, where
    woodelf's values come out sign-flipped relative to every other backend's
    class-1 convention.

    Root cause (upstream, in woodelf): it parses trees via shap's own loader
    (parse_models.py: "Use the shap package's Decision Tree loading. this is
    cheating, I know...") and takes ``tree.values[index][0]`` as the leaf
    value. For sklearn CART classifiers, shap's ``SingleTree`` reshapes the
    per-node value array into one column per class, so index 0 is class 0's
    probability — not class 1, the convention used everywhere else in this
    repo. Since prob(class 0) = 1 - prob(class 1), this is an exact sign flip.

    Only fires for models whose leaves store per-class columns: DecisionTree-
    Classifier and forest classifiers. HistGradientBoostingClassifier is
    sklearn too, but its leaves hold a single log-odds margin already in the
    class-1 direction — flipping it would corrupt correct values (verified:
    matches shap to ~1e-7 unflipped). Regressors and xgboost/lightgbm are
    unaffected (no such reshape).
    """
    if not (hasattr(model, "classes_") and len(model.classes_) == 2):
        return False
    if hasattr(model, "tree_"):  # DecisionTreeClassifier
        return True
    estimators = getattr(model, "estimators_", None)  # RandomForest/ExtraTrees
    return (
        isinstance(estimators, list)
        and len(estimators) > 0
        and hasattr(estimators[0], "tree_")
        and hasattr(estimators[0], "classes_")
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

        # Interventional woodelf explains the marginal game over the runner's
        # background, whose base value is the runner's own fallback — only the
        # path-dependent game needs its base value reported explicitly.
        if not self.interventional:
            self.baseline_ = _path_dependent_base_value(self.model)
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

        self.baseline_ = _path_dependent_base_value(self.model)
        reduced = reduce_multiclass(values, order=2)
        if _woodelf_class_sign_is_flipped(self.model):
            reduced = -reduced
        return flatten_interactions(reduced, x)
