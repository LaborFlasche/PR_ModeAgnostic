# Shapash — Backend Comparison Architecture

## How Shapash implements pluggable backends

### The core interface: `BaseBackend`

Shapash's entire XAI pipeline is decoupled from any specific attribution library through an abstract class `BaseBackend` (`shapash/backend/base_backend.py`).  The **only mandatory method** to implement is:

```python
def run_explainer(self, x: pd.DataFrame) -> dict:
    # Must return {'contributions': pd.DataFrame(n_samples, n_features)}
```

Everything else — preprocessing inversion, global importance, contribution ranking, the webapp, the HTML report — is handled by `SmartExplainer` on top of that single interface point.

### Two-phase execution in `SmartExplainer`

**Phase 1 — `SmartExplainer(model, backend='shap', ...)`**

The backend is stored by name or as a pre-instantiated object.  Nothing expensive happens yet.

**Phase 2 — `.compile(x=X_test)`**

```
compile(x)
  ├── backend.run_explainer(x)           → {'contributions': ...}
  ├── backend.get_local_contributions()  → tidy pd.DataFrame
  ├── backend.state                      → RegressionState / ClassificationState / MultiClassState
  └── state.rank_contributions()         → data['contrib_sorted'], data['var_dict'], data['x_sorted']
```

The state object (`SmartState` / `SmartMultiState`) handles everything that differs between regression, binary, and multi-class: ranking contributions per sample, masking, grouping.  Backends set `self.state` inside `run_explainer`.

### The built-in backends

| Backend | Class | Library | Auto-detects model type? |
|---|---|---|---|
| `'shap'` | `ShapBackend` | `shap` | Yes — Tree / Linear / Kernel |
| `'lime'` | `LimeBackend` | `lime` | No — always uses `LimeTabularExplainer` |

`ShapBackend.__init__` mirrors SHAP's own auto-detection:
1. `shap.explainers.Tree.supports_model_with_masker(model, None)` → `TreeExplainer`
2. `shap.explainers.Linear.supports_model_with_masker(model, masker)` → `LinearExplainer`
3. `hasattr(model, 'predict_proba')` → `Explainer(model.predict_proba, masker)`
4. fallback → `Explainer(model.predict, masker)` (KernelExplainer)

### What a backend must return

```python
{'contributions': pd.DataFrame}   # shape (n_samples, n_features)
                                   # index and columns must match x
```

For multi-class: `list[pd.DataFrame]`, one per class.

Contributions represent **signed feature attributions**: positive = pushed prediction up, negative = pushed prediction down.  This is the universal convention shared by SHAP, LIME, Integrated Gradients, and most other methods.

### How pre-computed contributions bypass the backend

If you already have contribution values (e.g. from FastTreeSHAP), pass them directly to `compile`:

```python
xpl.compile(x=X_test, contributions=my_dataframe)
```

Shapash calls `backend.format_and_aggregate_local_contributions(x, contributions)` instead of `run_explainer`.  The backend still handles preprocessing alignment; the library call is skipped.

### Global feature importance

Default implementation in `BaseBackend.get_global_features_importance`:

```python
state.compute_features_import(contributions, norm)
# = abs(contributions).mean(axis=0)   (L1-normalized)
```

Override this method in your subclass if your library provides a native global importance estimate.

---

## Our benchmarking framework (`src/`)

We extracted the essential backend pattern from Shapash and built a lightweight comparison harness on top of it.  The webapp, preprocessing inversion, and report generation are omitted — we only keep the contribution interface and add timing + cross-backend metrics.

### Architecture

```
src/
├── backends/
│   ├── base_backend.py     ← Shapash's BaseBackend, stripped to comparison essentials
│   ├── shap_backend.py     ← SHAP (mirrors Shapash's ShapBackend)
│   ├── shapiq_backend.py   ← ShapIQ (SV index = standard Shapley values)
│   └── captum_backend.py   ← Captum (gradient-based, PyTorch models only)
├── benchmarker.py          ← Runs all backends, collects results
└── metrics.py              ← Pairwise comparison metrics
```

### Comparison metrics

| Metric | What it measures |
|---|---|
| `sign_agreement` | % of (sample, feature) pairs where both methods agree on attribution direction |
| `mean_abs_diff` | Raw scale difference between attribution matrices |
| `mean_sample_rho` | Mean Spearman ρ per sample — do methods agree on within-sample feature ranking? |
| `global_rho` | Spearman ρ of global feature importance vectors — do methods agree on overall importance? |
| `top_k_overlap` | Fraction of top-k features shared across both rankings |

---

## Practical guide: adding a new backend

### Step 1 — Subclass `BaseBackend`

```python
# src/backends/my_backend.py
from .base_backend import BaseBackend
import pandas as pd

class MyBackend(BaseBackend):
    name = "mylib"                 # unique string key

    def __init__(self, model, **kwargs):
        super().__init__(model)
        # store any library-specific config here

    def run_explainer(self, x: pd.DataFrame) -> dict:
        # 1. Call your library
        raw_values = my_library.explain(self.model, x.values)  # shape (n_samples, n_features)

        # 2. Wrap in a DataFrame preserving index and columns
        contributions = pd.DataFrame(raw_values, index=x.index, columns=x.columns)

        # 3. Return the dict — this is the only required key
        return {'contributions': contributions}
```

Rules:
- `contributions` must be a `pd.DataFrame` with the **same index and columns** as `x`.
- For multi-class: return a **list** of DataFrames, one per class (class probabilities order).
- If your library is slow per-sample (like LIME), batch all samples in a single call.
- Store `self.explainer` if you want to expose native library objects later.

### Step 2 — Register in `__init__.py`

```python
# src/backends/__init__.py
from .my_backend import MyBackend
__all__ = [..., "MyBackend"]
```

### Step 3 — Add to a Benchmarker run

```python
from src.benchmarker import Benchmarker
from src.backends import ShapBackend, ShapIQBackend, MyBackend

bench = Benchmarker(
    model=clf,
    backends=[
        ShapBackend(clf, masker=X_train),
        ShapIQBackend(clf, data=X_train),
        MyBackend(clf),
    ],
)

results = bench.run(X_test)

print(results.summary())          # runtime + top-5 features per backend
print(results.pairwise_metrics()) # sign agreement, Spearman ρ, etc.
```

### Step 4 — Override optional methods if needed

| Method | When to override |
|---|---|
| `get_global_importance(contributions)` | Your library provides a native global importance estimate |
| `_to_dataframe(contributions, x)` | Your library returns an unusual object (not array/DataFrame) |

### Backend compatibility reference

| Backend | Model type | Install | Notes |
|---|---|---|---|
| `ShapBackend` | Any sklearn-compatible | `pip install shap` | Auto-detects TreeExplainer / KernelExplainer |
| `ShapIQBackend` | Any sklearn-compatible | `pip install shapiq` | Uses `index='SV'` for Shapley values comparable to SHAP |
| `CaptumBackend` | `torch.nn.Module` only | `pip install captum torch` | Gradient-based; choose method: `integrated_gradients`, `gradient_shap`, `deep_lift`, `saliency` |

### What Shapash adds on top (not in our benchmarker)

If you want to move from benchmarking to full explainability (webapp, report, postprocessing), wrap any `BaseBackend` subclass in `SmartExplainer`:

```python
from shapash import SmartExplainer

backend = MyBackend(clf)
xpl = SmartExplainer(model=clf, backend=backend)
xpl.compile(x=X_test)
xpl.plot.features_importance()   # full Shapash visualization stack
```

Our `BaseBackend` is a subset of Shapash's `BaseBackend`.  The only difference is that Shapash's version also handles preprocessing inversion (`_apply_preprocessing`) and sets `self.state` for the webapp's filter/mask machinery.  For benchmarking purposes these are not needed.
