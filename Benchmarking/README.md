# Benchmarking Framework

Compares Shapley-value XAI backends on runtime and accuracy across model × dataset sweeps. Results are written incrementally to CSV so runs can be interrupted and resumed.

---

## Add a New Backend

1. Create a new file, e.g. `Benchmarking/backends/captum_backend.py`
2. Subclass `BaseBackend`:

```python
from Benchmarking.backends.base_backend import BaseBackend
import pandas as pd

class CaptumBackend(BaseBackend):
    name = "captum"           # unique string key, appears in CSV
    library = "captum"        # library name, used for reference resolution
    computation_type = "approximation"  # "true_value" or "approximation"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        # x: input samples, shape (n_samples, n_features)
        # self.model: trained model passed at construction
        # self.background: reference distribution passed at construction
        ...
        # Return: pd.DataFrame of signed attributions, shape (n_samples, n_features)
        #   - same index and columns as x
        #   - positive value = feature pushed prediction up
        #   - negative value = feature pushed prediction down
```

3. Register in `Benchmarking/backends/__init__.py`:

```python
from .captum_backend import CaptumBackend
```

4. Pass the class to `BenchmarkRunner` (see below)

---

## Use the Benchmarker

### Load the config

```python
from itertools import product
from sklearn.model_selection import ParameterGrid
from Models.config_parser import load_config, load_dataset_config
from Models.dataset_and_models import Dataset, Model

CONFIG = "../configs/config-test.yaml"

model_config   = load_config(CONFIG)
dataset_config = load_dataset_config(CONFIG)

model_runs = [
    (model_key, params)
    for model_key, param_grid in model_config.items()
    for params in ParameterGrid(param_grid)
]

dataset_runs = [
    (dataset_key, params)
    for dataset_key, param_grid in dataset_config.items()
    for params in ParameterGrid(param_grid)
]
```

### Run the benchmarker

```python
from Benchmarking import BenchmarkRunner
from Benchmarking.backends import ShapTrueValueBackend, ShapIQTrueValueBackend

runner = BenchmarkRunner(
    backends=[ShapTrueValueBackend, ShapIQTrueValueBackend],
    output_csv="../Benchmarking/results.csv",
    n_background=100,   # samples used as reference distribution (not explained)
    n_eval=None,        # None = explain all remaining samples; int = explain that many
)

tree_model_keys = ["random_forest", "decision_tree", "gradient_boosting"]

for dataset_key, dataset_params in dataset_runs:
    dataset_enum = Dataset[dataset_key.upper()]
    ds = dataset_enum.load_dataset(**dataset_params)
    X, y = ds["X"], ds["y"]

    tree_runs = [(mk, mp) for mk, mp in model_runs if mk in tree_model_keys]

    for model_key, model_params in tree_runs:
        trainer = Model[model_key.upper()].get_model_with_params(dataset_enum, model_params)
        trainer.fit(X, y, task=ds["task"])

        runner.run(
            model=trainer.get_model(),
            X=X,
            run_meta={
                "dataset": dataset_key,
                "model": model_key,
                "n_features": dataset_params.get("n_features"),
                "n_samples": dataset_params.get("n_samples"),
            },
        )
```

> **Note:** Results are appended to the CSV on every run. Delete `results.csv` before a clean sweep to avoid mixing old and new results.

### CSV output

One row per backend per run. Key columns:

| Column | Description |
|---|---|
| `dataset`, `model`, `n_features`, `n_samples` | Sweep parameters from `run_meta` |
| `backend`, `library`, `computation_type` | Backend identity |
| `n_eval` | Number of samples explained |
| `runtime_s` | Wall-clock time for that backend |
| `mean_abs_diff`, `sign_agreement`, `mean_sample_rho` | Accuracy vs reference (NaN for the absolute reference) |
| `reference_backend` | Which backend was used as ground truth |

---

## Metrics

All metrics compare two attribution matrices of shape `(n_samples, n_features)`.

**`mean_abs_diff`** — Mean |a − b| across all samples and features. Measures the raw scale difference between two methods. Lower = closer agreement.

**`sign_agreement`** — Fraction of (sample, feature) pairs where both methods agree on attribution direction (positive vs. negative). Zero-valued attributions are excluded from both sides before comparing. Higher = better directional agreement.

**`mean_sample_rho`** — Mean Spearman rank correlation per sample. For each sample, ranks features by attribution magnitude and correlates the two rankings. Measures whether methods agree on *which features matter most*, independent of scale. Undefined (NaN) when `n_features = 1`.

---

## Reference resolution

Accuracy metrics are always computed against a ground-truth reference:

| Backend type | Reference |
|---|---|
| `ShapTrueValueBackend` (shap, true_value) | None — this is the absolute ground truth |
| Any other `true_value` backend | `ShapTrueValueBackend` |
| Any `approximation` backend | The `true_value` backend from the same library |
