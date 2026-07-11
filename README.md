# ShapiqBench — Model-Agnostic XAI Benchmark

A benchmarking framework for comparing model-agnostic Shapley value-based XAI methods against [shapiq](https://github.com/mmschlk/shapiq) across correctness, interaction support, and runtime.

The goal is to evaluate whether `shapiq` is the best-in-class library for Shapley-based explainability — and to understand where competing tools offer complementary or superior capabilities.

## Libraries benchmarked

| Library | Focus | Interaction Indices | Notes |
|---|---|---|---|
| `shapiq` | **Benchmark baseline** | Any-order SII, STI, FSI, … | 20+ interaction indices; exact & approximate |
| `shap` | Standard SHAP baseline | Pairwise (trees only) | KernelSHAP, PermutationSHAP, and more |
| `lightshap` | Fast tabular attribution | None (interaction heuristic only) | Zero-dependency; Polars-native; standard errors |
| `captum` | PyTorch neural networks | Order 1 only | Gradient-based methods; KernelSHAP fully in Torch |
| `dalex` | Broad explainability | iBreakDown | Model-agnostic; R-origin, Python port |

Wired into the automated `benchmarking/` pipeline (`slurm/run_benchmark.py` and `slurm/run_benchmark_nn.py`); tree-specific backends (`woodelf`, `fastTreeSHAP`) are covered separately below.

## Research questions

1. Does `shapiq` produce more faithful explanations than single-index competitors on tabular data?
2. How does runtime scale with number of features and coalition order across libraries?
3. Can gradient-based attribution (`captum`) match Shapley-based attribution for PyTorch models in quality and speed?

## Datasets

Stored in `datasets/`. See `datasets/README.md` for full details on feature encoding and the feature-selection strategy used for experiments.

| Dataset | Task | Samples | Features | Source |
|---|---|---|---|---|
| California Housing | Regression | 20,640 | 8 | `sklearn.datasets` |
| Ames Housing | Regression | 1,460 | ~79 | OpenML #42165 |
| Forest Covertype | Classification | 50,000* | 54 | `sklearn.datasets` |
| Adult Census | Classification | 48,842 | 14 | OpenML #1590 |
| Gisette | Classification | 7,000 | 5,000 | OpenML #41026 |

\* Stratified subsample of the full 581k-row dataset for faster experimentation.

## Benchmark configuration

Each research question has its own self-contained config under `configs/`, grouped by RQ:

| Config | Research question |
|---|---|
| `configs/RQ1-accuracy/config-accuracy.yaml` | Approximation accuracy vs. budget |
| `configs/RQ2-dimensionality/config-dimensionality.yaml` (+ `-extreme` variant) | Runtime/accuracy scaling with feature count |
| `configs/RQ3-neural-networks/config-neural-networks-{cpu,gpu}.yaml` | Gradient-based vs. Shapley-based attribution on PyTorch models |
| `configs/RQ4-tree/config-tree.yaml` (+ `-fasttreeshap` variant) | Tree-specific exact/true-value backends |
| `configs/RQ5-gpu/config-tree-gpu.yaml` | GPU-backed tree backends |

Each file defines its own `models:` and `datasets:` sections (hyperparameter and sweep ranges) plus a `benchmark:` section (seed, backends, timeouts, …). Use `load_config` and `load_dataset_config` from `models/config_parser.py` to expand any one of them into all benchmark combinations:

```python
from itertools import product
from sklearn.model_selection import ParameterGrid
from models.config_parser import load_config, load_dataset_config, load_seed
from models.dataset_and_models import Dataset

CONFIG = "configs/RQ1-accuracy/config-accuracy.yaml"

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
uv run python slurm/run_benchmark.py --task-id 0 --config configs/RQ4-tree/config-tree.yaml
```

| Library | Modes | Order-2 interactions |
|---|---|---|
| `shap` (TreeSHAP) | path-dependent | yes — order-2 oracle |
| `shapiq` (TreeSHAP-IQ) | path-dependent | yes |
| `woodelf` | path-dependent, interventional | yes |
| `fasttreeshap` | path-dependent | no |

`fasttreeshap` requires `numpy<2`, incompatible with this project's main `numpy>=2` stack, so it runs out-of-process in a dedicated venv — provision it once with `bash scripts/setup_fasttreeshap_env.sh`. It also can't parse XGBoost 3.x's model format (an upstream limitation) and is skipped for XGBoost models specifically.

`shapiq`'s interventional `TreeExplainer` is excluded: it crashes unreliably in this environment (see `benchmarking/backends/trees/shapiq_backend.py`).

GPU-backed `woodelf` (`GPU=True`) is wired into `slurm/run_benchmark.py`'s `TREE_TRUE_VALUE_MAP` under the `gpu_path_dependent`/`gpu_interventional` tree modes and exercised by `configs/RQ5-gpu/config-tree-gpu.yaml` (run via `slurm/bench_array_gpu.sh`, which requests a GPU node) — still unverified on real GPU hardware, so without a CUDA device + `cupy` it skips to all-NaN rows.

`configs/RQ4-tree/config-tree.yaml`'s `tree_libraries`/`tree_modes`/`interaction_libraries` control which of the above run; `interaction_max_features` caps interaction sweeps since their output is quadratic in feature count.

## Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager

### Install

```bash
uv sync
```

This creates a `.venv` and installs all dependencies from `uv.lock` in one step.

### Run scripts

```bash
uv run python slurm/run_benchmark.py --task-id 0 --config configs/RQ1-accuracy/config-accuracy.yaml
```

### Run tests

```bash
uv run pytest tests/
```

## Project structure

```
benchmarking/
  runner.py                 # BenchmarkRunner — runs one oracle + backends/approximations per cell
  metrics.py                # mean_abs_diff, sign_agreement, mean_sample_rho, additivity gaps
  eval_counter.py           # CountingModel — counts real model evaluations per backend
  timeout.py                # per-backend wall-clock budget
  backends/
    true_value/             # exact/reference backends (shap, shapiq, lightshap, dalex)
    approximators/          # approximate backends (shap, shapiq, lightshap, dalex, captum, shap_nn)
    trees/                  # tree-specific backends (shap, shapiq, woodelf, fasttreeshap)
models/
  dataset_and_models.py     # Dataset and Model enums / definitions
  trainers.py               # ModelTrainer ABC; SklearnTrainer and PytorchTrainer implementations
  architectures.py          # PyTorch architectures (MLP, TabularTransformer, CNN-1D)
  config_parser.py          # load_config / load_dataset_config — expand a config yaml into parameter lists
datasets/
  load_datasets.py          # Dataset download and caching helpers (support n_features / n_samples)
  README.md                 # Dataset documentation incl. encoding strategy and feature-selection notes
configs/
  RQ1-accuracy/ … RQ5-gpu/  # one self-contained config per research question — see "Benchmark configuration" above
slurm/                      # SLURM array-job scripts; see SLURM.md
scripts/
  setup_fasttreeshap_env.sh # Provisions the dedicated venv fasttreeshap runs in (numpy<2)
tests/                      # pytest suite for backends, runner, and metrics
pyproject.toml              # Project metadata and dependencies
uv.lock                     # Locked dependency versions (commit this)
```
