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

Excluded (tree-specific only): `woodelf`, `fastTreeSHAP`, `GPUTreeSHAP`.

## Research questions

1. Does `shapiq` produce more faithful explanations than single-index competitors on tabular data?
2. How does runtime scale with number of features and coalition order across libraries?
3. Where do conditional methods (e.g. `shapr`) diverge from marginal methods (e.g. `shap`, `shapiq`) on correlated datasets?
4. Can gradient-based attribution (`captum`) match Shapley-based attribution for PyTorch models in quality and speed?
5. What does `shapleyflow`'s path decomposition reveal that standard interaction indices miss?

## Datasets

Stored in `Datasets/`:

| Dataset | Task | Samples | Features | Correlation | Source |
|---|---|---|---|---|---|
| Census Income (Adult) | Classification | ~32k | 13 | Moderate | [UCI](https://archive.ics.uci.edu/dataset/20/census+income) |
| Superconductivity | Regression | ~21k | 81 | High | [UCI](https://archive.ics.uci.edu/dataset/464/superconductivty+data) |

The Superconductivity dataset is intentionally high-correlation — useful for stressing conditional vs. marginal methods (`shapr` vs. `shap`/`shapiq`).

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

**shapr** requires R in addition to the Python wrapper:
```bash
# Install R >= 4.3 via https://cran.r-project.org, then:
Rscript -e "install.packages('shapr')"
uv pip install shaprpy rpy2
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
Datasets/
  load_datasets.py        # Dataset download and caching helpers
  dataset.md              # Dataset documentation
requirements.txt
```
