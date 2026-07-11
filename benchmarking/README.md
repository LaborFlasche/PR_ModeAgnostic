# Benchmarking Framework

Compares Shapley-value XAI backends on runtime and accuracy across model × dataset sweeps. Results are written incrementally to CSV so runs can be interrupted and resumed.

---

## Add a New Backend

1. Create a new file, e.g. `benchmarking/backends/captum_backend.py`
2. Subclass `BaseBackend`:

```python
import pandas as pd
from benchmarking.backends.base_backend import BaseBackend, marginal_predict

class CaptumApproxBackend(BaseBackend):
    name = "captum_approx"    # unique string key, appears in CSV
    library = "captum"        # library name, used for reference resolution
    computation_type = "approximation"  # "true_value" or "approximation"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        # self.config holds the per-run knobs from the (backend, config) spec.
        approximator = self.config.get("approximator")   # e.g. "kernel" / "permutation"
        budget = self.config.get("budget")

        # marginal_predict gives a scalar prediction function aligned with the shap
        # exact reference. self.model is a CountingModel for approximation backends,
        # so every predict call it makes is counted automatically.
        f = marginal_predict(self.model, x.columns)

        attributions = ...  # run captum with f, self.background, approximator, budget

        # Return signed attributions, shape (n_samples, n_features), same index/columns
        # as x: positive = feature pushed the prediction up, negative = pushed it down.
        return pd.DataFrame(attributions, index=x.index, columns=x.columns)
```

3. Register in `benchmarking/backends/__init__.py`.

4. Add it to the runner's `approximation_specs` as `(CaptumApproxBackend, {"approximator": ..., "budget": ...})` (see below).

---


### On running the benchmarker

See `slurm/run_benchmark.py` for the driving implementation (one invocation per
(dataset, model) cell; `slurm/submit.sh` fans it out as a SLURM array job).

The runner takes **one exact oracle** (computed once per cell) plus a list of
**approximation specs** — `(backend_class, config)` pairs, where `config` carries the
per-run knobs `{"approximator": ..., "budget": ...}`. Each approximation is wrapped in a
predict-counter so the real number of model evaluations is recorded.

> **Why one oracle, not two?** Exact Shapley is exponential (`2^M`) only for a *black-box*
> model. Every model here is a tree or linear model, so `shap.Explainer` dispatches to a
> **polynomial** exact explainer (TreeSHAP / LinearSHAP) that is fast at any feature count.
> `shapiq`'s `budget = 2^M` brute force is *not* needed as a second oracle — `shapiq`
> appears only as an *approximation* backend. The polynomial oracle was validated once
> against brute-force exact enumeration.

> **Reproducibility & control via the config.** Two `benchmark` fields in each
> `configs/RQ*/*.yaml` are the single source of truth for the experiment, with no
> hardcoded fallbacks downstream:
> - `seed` — threaded into data subsampling, model training (every estimator's
>   `random_state`), and every stochastic approximator, so changing it reseeds the whole
>   experiment reproducibly.
> - `imputer` — the shared value function. The runner injects it into every backend, which
>   honors it (shapiq `imputer=`, shap `Independent` masker / `KernelExplainer` background,
>   lightshap `bg_X`, dalex background sampling) or raises if it cannot. Combined with
>   `marginal_predict` fixing the output space, this guarantees all libraries are compared
>   on the *same* value function. `marginal` is currently the only supported value.
>
> Both are recorded in every CSV row for provenance.

> **Note:** Results are appended to the CSV on every run. Delete `results.csv` before a clean sweep to avoid mixing old and new results.

### CSV output

One row per backend per run. Key columns:

| Column | Description |
|---|---|
| `dataset`, `model`, `n_features`, `n_samples` | Sweep parameters from `run_meta` |
| `backend`, `library`, `computation_type` | Backend identity |
| `approximator`, `budget` | Approximation config (NaN for the exact oracle) |
| `seed` | The single benchmark-wide RNG seed used for this run |
| `imputer` | The shared value function every library explains (e.g. `marginal`) |
| `n_eval` | Number of samples explained |
| `runtime_s` | Wall-clock time for that backend |
| `n_model_evals` | Actual rows scored by the model — the fair budget axis across libraries (NaN for the oracle) |
| `additivity_gap`, `relative_additivity_gap` | Local-accuracy violation vs the shared value function (see Metrics) — absolute, needs no reference backend |
| `mean_abs_diff`, `relative_mae`, `sign_agreement`, `mean_sample_rho` | Accuracy vs reference (NaN for the absolute reference) |
| `reference_backend` | Which backend was used as ground truth |

---

## Metrics

All metrics compare two attribution matrices of shape `(n_samples, n_features)`.

**`mean_abs_diff`** — Mean |a − b| across all samples and features. Measures the raw scale difference between two methods, in model-output units. Lower = closer agreement.

**`relative_mae`** — `mean_abs_diff` divided by the mean magnitude of the exact values. Dimensionless, so unlike `mean_abs_diff` it can be compared and averaged across datasets with different output scales (house prices vs. probabilities).

**`sign_agreement`** — Fraction of (sample, feature) pairs where both methods agree on attribution direction (positive vs. negative). Zero-valued attributions are excluded from both sides before comparing. Higher = better directional agreement.

**`mean_sample_rho`** — Mean Spearman rank correlation per sample. For each sample, ranks features by attribution magnitude and correlates the two rankings. Measures whether methods agree on *which features matter most*, independent of scale. Undefined (NaN) when `n_features = 1`.

### Additivity metrics (local accuracy / efficiency)

Shapley values satisfy the **efficiency axiom**: `E[f(X)] + Σ_j φ_ij = f(x_i)` for every
sample — the baseline plus all attributions must reproduce the model output exactly
(what the `shap` library calls *local accuracy* and asserts with `check_additivity`).
The runner verifies this against the shared value function: `baseline` is the mean of
`marginal_predict` over the shared background, and both metrics are recorded per backend
row, no reference backend involved.

**`additivity_gap`** — Mean |f(x_i) − (baseline + Σ_j φ_ij)| over the evaluation samples,
in model-output units. 0 = the property holds exactly. ~0 does not mean the values are
correct: exact methods and constraint-enforcing approximators (KernelSHAP) satisfy it by
construction even when per-feature values are off. A large gap proves a problem — real
signal for gradient-based methods, backends with a different internal baseline (e.g.
path-dependent TreeSHAP), and implementation bugs (wrong output space or background).

**`relative_additivity_gap`** — `additivity_gap` divided by the mean |f(x_i) − baseline|,
i.e. by the attribution mass the values are supposed to distribute. Dimensionless, 0 =
perfect, so comparable across datasets like `relative_mae`. NaN when every prediction
equals the baseline. Values ≳ 1 mean the violation is as large as the signal being
explained — the attributions do not decompose the prediction in any meaningful sense.

---

## Reference resolution

Accuracy metrics are always computed against a ground-truth reference:

| Backend type | Reference |
|---|---|
| `ShapTrueValueBackend` (shap, true_value) | None — this is the absolute ground truth (oracle) |
| Any other `true_value` backend | The shap oracle |
| Any `approximation` backend | The shap oracle |
