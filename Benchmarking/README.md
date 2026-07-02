# Benchmarking Framework

Compares Shapley-value XAI backends on runtime and accuracy across model × dataset sweeps. Results are written incrementally to CSV so runs can be interrupted and resumed.

---

## Add a New Backend

1. Create a new file, e.g. `Benchmarking/backends/captum_backend.py`
2. Subclass `BaseBackend`:

```python
import pandas as pd
from Benchmarking.backends.base_backend import BaseBackend, marginal_predict

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

3. Register in `Benchmarking/backends/__init__.py`.

4. Add it to the runner's `approximation_specs` as `(CaptumApproxBackend, {"approximator": ..., "budget": ...})` (see below).

---


### On running the benchmarker

See Libraries\library_merge.ipynb for an implementation.

The runner takes **one exact oracle** (computed once per cell) plus a list of
**approximation specs** — `(backend_class, config)` pairs, where `config` carries the
per-run knobs `{"approximator": ..., "budget": ...}`. Each approximation is wrapped in a
predict-counter so the real number of model evaluations is recorded.

> **Why one oracle, not two?** Exact Shapley is exponential (`2^M`) only for a *black-box*
> model. Every model here is a tree or linear model, so `shap.Explainer` dispatches to a
> **polynomial** exact explainer (TreeSHAP / LinearSHAP) that is fast at any feature count.
> `shapiq`'s `budget = 2^M` brute force is *not* needed as a second oracle — `shapiq`
> appears only as an *approximation* backend. The polynomial oracle is validated once
> against brute-force exact in the notebook's "Ground-truth validation" section.

> **Reproducibility & control via the config.** Two `benchmark` fields in
> `configs/config.yaml` are the single source of truth for the experiment, with no
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
| `shapley_values`, `shapley_n_eval`, `shapley_n_features` | The raw attribution matrix (JSON-flattened) and its shape |
| `pairwise_metrics` | JSON dict `{reference_backend: {metric: value}}` with the four pairwise accuracy metrics against every backend of the run |

---

## Metrics

Two kinds of metrics are recorded. The four **pairwise metrics** compare two attribution
matrices of shape `(n_samples, n_features)` and are computed for every ordered
(candidate, reference) backend pair into the `pairwise_metrics` JSON column. The two
**additivity metrics** need no reference backend: they check each backend's values against
the model itself.

All of them are only meaningful because every backend explains the *same* value function —
the marginal (interventional) game on the shared background, with `marginal_predict` fixing
the output space (raw prediction for regressors, margin/log-odds for margin-based
classifiers, probability of the target class otherwise). Comparing attributions produced in
different output spaces (probability vs. log-odds) is meaningless; that invariant is what
makes every number below interpretable.

### Pairwise metrics

**`mean_abs_diff`** — Mean |a − b| across all samples and features. Measures the raw
disagreement between two methods, in model-output units.
*Consider:* unit-bound — a value of 0.1 is huge for probabilities and negligible for house
prices, so it must never be averaged or compared across datasets/models with different
output scales. Use `relative_mae` for that.

**`relative_mae`** — `mean_abs_diff` divided by the mean magnitude of both matrices
(symmetric, so candidate and reference are interchangeable). Dimensionless, 0 = perfect.
*Consider:* comparable and averageable across datasets. NaN when both matrices are all
zero. Because the denominator is the *mean* magnitude, a method that is accurate on the few
large attributions but noisy on the many near-zero ones can still score well — pair it with
`sign_agreement`/`mean_sample_rho` before drawing conclusions.

**`sign_agreement`** — Fraction of (sample, feature) cells where both methods agree on the
attribution direction (positive vs. negative). Cells where either side is exactly zero are
excluded. Higher = better; 0.5 ≈ chance level.
*Consider:* completely magnitude-blind — two methods can agree on every sign while
disagreeing wildly on importance. The zero-exclusion means the score can rest on few cells
when attributions are sparse (NaN if none remain), so check the effective count on sparse
problems.

**`mean_sample_rho`** — Per-sample Spearman rank correlation of the signed attribution
vectors, averaged over samples. Measures whether two methods rank features the same way,
independent of scale (1 = identical ranking, −1 = reversed).
*Consider:* it ranks *signed* values, not absolute magnitudes — a large negative
attribution ranks below a small positive one, so it captures the full ordering rather than
"which features matter most" in the |value| sense. Unstable when attributions are
near-constant across features (rank ties) and undefined (NaN) for a single feature.

### Additivity metrics (local accuracy / efficiency)

Shapley values satisfy the **efficiency axiom**: `E[f(X)] + Σ_j φ_ij = f(x_i)` for every
sample — the baseline plus all attributions must reproduce the model output exactly
(what the `shap` library calls *local accuracy* and asserts with `check_additivity`).
The runner verifies this against the shared value function: `baseline` is the mean of
`marginal_predict` over the shared background, and both metrics are recorded per backend
row, no reference backend involved.

**`additivity_gap`** — Mean |f(x_i) − (baseline + Σ_j φ_ij)| over the evaluation samples,
in model-output units. 0 = the property holds exactly.
*Consider:*
- **~0 does not mean the values are correct.** Exact methods (TreeSHAP, LinearSHAP) satisfy
  it by construction, but so do constraint-enforcing approximators (KernelSHAP solves a
  regression with efficiency as a hard constraint; permutation SHAP telescopes to it) —
  even when their per-feature values are still far from the truth. A large gap proves a
  problem; a zero gap proves nothing about attribution quality.
- **Where it is a real signal:** gradient-based methods (captum GradientShap/DeepLiftShap
  satisfy completeness only approximately), sampling methods without the constraint, and
  implementation bugs (wrong output space, wrong background handling).
- **The gap conflates two errors:** attribution error and baseline mismatch. A backend that
  internally uses a different baseline than the shared background mean (e.g.
  path-dependent TreeSHAP, whose `expected_value` comes from tree cover statistics, or
  shap's binned traversal of `HistGradientBoosting` — the reason the oracle sets
  `check_additivity=False`) shows a nonzero gap even if its values are self-consistent.
  For this benchmark that is the intended reading: the gap measures deviation from the
  *common* value function all backends are supposed to target.
- For order-2 (interaction) backends the row-sum runs over the full flattened `d²` matrix;
  the property then requires the library's matrix convention to sum to `f(x) − baseline`
  (true for shap-style interaction matrices: main effects on the diagonal, split
  interactions off it).
- NaN for crashed backends (all-NaN rows propagate).

**`relative_additivity_gap`** — `additivity_gap` divided by the mean |f(x_i) − baseline|,
i.e. by the attribution mass the values are supposed to distribute. Dimensionless, 0 =
perfect, so comparable across datasets like `relative_mae`.
*Consider:* NaN when every prediction equals the baseline (a constant model has nothing to
attribute). Values ≳ 1 mean the violation is as large as the signal being explained — the
attributions do not decompose the prediction in any meaningful sense.

---

## Reference resolution

The pairwise metrics are computed for **every ordered (candidate, reference) pair** of the
run and stored in each row's `pairwise_metrics` JSON dict, keyed by reference backend name
(the self-comparison is included as an identity row). Analysis picks the reference at read
time — typically the shap oracle (`computation_type = "true_value"`) as ground truth, but
approximator-vs-approximator agreement is available from the same data.

The additivity metrics are independent of any reference: they compare each backend directly
against the model output and remain meaningful even in runs where no exact oracle could run.
