# ShapiqBench — Model-Agnostic XAI Benchmark

A benchmarking framework for comparing model-agnostic Shapley value-based XAI methods against [shapiq](https://github.com/mmschlk/shapiq) across correctness, interaction support, and runtime.

The goal is to evaluate whether `shapiq` is the best-in-class library for Shapley-based explainability — and to understand where competing tools offer complementary or superior capabilities.

## Libraries benchmarked

| Library | Focus | Interaction Indices | Notes |
|---|---|---|---|
| `shapiq` | **Benchmark baseline** | Any-order SII, STI, FSI, … | 20+ interaction indices; exact & approximate |
| `shap` | Standard SHAP baseline | Pairwise (trees only) | KernelSHAP, PermutationSHAP, and more |
| `shapr` | Conditional Shapley values | Feature groups | Targets correlated features; R-first, Python wrapper available |
| `lightshap` | Fast tabular attribution | None (interaction heuristic only) | Zero-dependency; Polars-native; standard errors |
| `captum` | PyTorch neural networks | Order 1 only | Gradient-based methods; KernelSHAP fully in Torch |
| `fastshap` | Amortized SHAP approximation | None | Learns a surrogate explainer; trades accuracy for speed |
| `alibi` | Production XAI toolbox | Via shap | Counterfactual & similarity explanations; shapiq for Shapley |
| `dalex` | Broad explainability | iBreakDown | Model-agnostic; R-origin, Python port |
| `shapleyflow` | Graph-based Shapley influence | High-order (graph paths) | Decomposes feature effects into direct vs. mediated paths |

Tree-specific only (not in the table above; see [Tree-specific benchmark](#tree-specific-benchmark)): `woodelf`, `fastTreeSHAP`, `GPUTreeSHAP`.

## Research questions

1. Does `shapiq` produce more faithful explanations than single-index competitors on tabular data?
2. How does runtime scale with number of features and coalition order across libraries?
3. Where do conditional methods (e.g. `shapr`) diverge from marginal methods (e.g. `shap`, `shapiq`) on correlated datasets?
4. Can gradient-based attribution (`captum`) match Shapley-based attribution for PyTorch models in quality and speed?
5. What does `shapleyflow`'s path decomposition reveal that standard interaction indices miss?

## Datasets

Stored in `Datasets/`. See `Datasets/dataset.md` for full details on feature encoding and the feature-selection strategy used for experiments.

| Dataset | Task | Samples | Features | Source |
|---|---|---|---|---|
| California Housing | Regression | 20,640 | 8 | `sklearn.datasets` |
| Ames Housing | Regression | 1,460 | ~79 | OpenML #42165 |
| Forest Covertype | Classification | 50,000* | 54 | `sklearn.datasets` |
| Adult Census | Classification | 48,842 | 14 | OpenML #1590 |
| Gisette | Classification | 7,000 | 5,000 | OpenML #41026 |

\* Stratified subsample of the full 581k-row dataset for faster experimentation.

## Benchmark configuration

Model hyperparameters and dataset sweep ranges are defined in `configs/config.yaml` (model-agnostic) and `configs/config-tree.yaml` (tree-specific — see below). Use `load_config` and `load_dataset_config` from `Models/config_parser.py` to expand either into all benchmark combinations:

```python
from itertools import product
from sklearn.model_selection import ParameterGrid
from Models.config_parser import load_config, load_dataset_config, load_seed
from Models.dataset_and_models import Dataset

CONFIG = "configs/config.yaml"

model_config   = load_config(CONFIG)        # {model_key: {param: [values]}}
dataset_config = load_dataset_config(CONFIG) # {dataset_key: {n_features: [...], n_samples: [...]}}
seed = load_seed(CONFIG)                     # benchmark-wide RNG seed (not a hyperparameter)

# All model × hyperparameter combinations
model_runs = [
    (model_key, params)
    for model_key, param_grid in model_config.items()
    for params in ParameterGrid(param_grid)
]

# All dataset × (n_features, n_samples) combinations
dataset_runs = [
    (dataset_key, params)
    for dataset_key, param_grid in dataset_config.items()
    for params in ParameterGrid(param_grid)
]

# Full cross-product: one entry per benchmark run
benchmark_runs = list(product(model_runs, dataset_runs))
print(f"{len(model_runs)} model configs × {len(dataset_runs)} dataset configs = {len(benchmark_runs)} total runs")

# Load a single dataset variant
dataset_key, dataset_params = dataset_runs[0]
ds = Dataset[dataset_key.upper()].load_dataset(**dataset_params, seed=seed)
X, y = ds["X"], ds["y"]
```

Features within each dataset are reduced by ranking on variance (`VarianceThreshold`) and keeping the top `n_features`; samples are drawn randomly with `random_state=42`.

## Tree-specific benchmark

Tree models (`xgboost`, `lightgbm`, and any sklearn tree model) can additionally be benchmarked against tree-specific exact/true-value SHAP backends, run via a separate config so the model-agnostic sweep above stays untouched:

```bash
uv run python slurm/run_benchmark.py --task-id 0 --config configs/config-tree.yaml
```

Or run `Libraries/tree_library_merge.ipynb` directly — it mirrors `run_benchmark.py`'s sweep against `configs/config-tree.yaml` for interactive use.

| Library | Modes | Order-2 interactions |
|---|---|---|
| `shap` (TreeSHAP) | path-dependent | yes — order-2 oracle |
| `shapiq` (TreeSHAP-IQ) | path-dependent | yes |
| `woodelf` | path-dependent, interventional | yes |
| `fasttreeshap` | path-dependent | no |

`fasttreeshap` requires `numpy<2`, incompatible with this project's main `numpy>=2` stack, so it runs out-of-process in a dedicated venv — provision it once with `bash scripts/setup_fasttreeshap_env.sh`. It also can't parse XGBoost 3.x's model format (an upstream limitation) and is skipped for XGBoost models specifically.

`shapiq`'s interventional `TreeExplainer` is excluded: it crashes unreliably in this environment (see `Benchmarking/backends/tree_shapiq_backend.py`).

GPU-backed variants (`woodelf` with `GPU=True`, and XGBoost's native GPU SHAP path under the name `gputreeshap`) exist in `Benchmarking/backends/` for future use but aren't currently wired into `configs/config-tree.yaml` or `slurm/run_benchmark.py` — no GPU hardware has been available to verify them yet.

`configs/config-tree.yaml`'s `tree_libraries`/`tree_modes`/`interaction_libraries` control which of the above run; `interaction_max_features` caps interaction sweeps since their output is quadratic in feature count.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager

### Install

```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `uv.lock` in one step.

### Run notebooks

```bash
Go to the notebook and run it using the venv from uv.
```

### Run scripts

```bash
uv run Models/load_and_train.py
```

### shapr (optional)

**shapr** requires R in addition to the Python wrapper:

```bash
# Install R >= 4.3 via https://cran.r-project.org, then:
Rscript -e "install.packages('shapr')"
uv add shaprpy rpy2
```

## Project structure

```
Libraries/
  shapiq.ipynb            # Benchmark baseline
  shap.ipynb              # Standard SHAP baseline
  shapr.ipynb             # Conditional Shapley values
  lightshap.ipynb         # Fast tabular attribution, zero-dependency
  captum.ipynb            # PyTorch attribution (IG, KernelSHAP, …)
  fastshap.ipynb          # Amortized SHAP approximation
  alibi.ipynb             # Production XAI — counterfactuals, anchors
  dalex.ipynb             # iBreakDown, PDP, variable importance
  shapleyflow.ipynb       # Graph-based path-decomposed Shapley values
  library_merge.ipynb     # Model-agnostic sweep, interactive equivalent of run_benchmark.py
  tree_library_merge.ipynb # Tree-specific sweep, interactive equivalent of run_benchmark.py --config configs/config-tree.yaml
Models/
  dataset_and_models.py   # Dataset and Model enums / definitions
  trainers.py             # ModelTrainer ABC; SklearnTrainer and PytorchTrainer implementations
  config_parser.py        # load_config / load_dataset_config — expand config.yaml into parameter lists
  load_and_train.py       # TrainingConfig — pairs a dataset with a model and exposes train()
Datasets/
  load_datasets.py        # Dataset download and caching helpers (support n_features / n_samples)
  dataset.md              # Dataset documentation incl. encoding strategy and feature-selection notes
Benchmarking/
  runner.py                # BenchmarkRunner — runs one oracle + backends/approximations per cell
  metrics.py                # mean_abs_diff, sign_agreement, mean_sample_rho, runtime
  backends/                 # one BaseBackend subclass per (library, mode); tree_*.py / woodelf_backend.py /
                             # fasttreeshap_backend.py / gputreeshap_backend.py are the tree-specific ones
configs/
  config.yaml               # Model-agnostic benchmark config (hyperparameters, dataset sweeps, approximators)
  config-tree.yaml          # Tree-specific config — see "Tree-specific benchmark" above
slurm/                      # SLURM array-job scripts; see SLURM.md
scripts/
  setup_fasttreeshap_env.sh # Provisions the dedicated venv fasttreeshap runs in (numpy<2)
tests/                      # pytest suite for backends, runner, and metrics
pyproject.toml              # Project metadata and dependencies
uv.lock                     # Locked dependency versions (commit this)
```
