# Datasets

A curated set of tabular datasets used to evaluate XAI libraries across a spectrum of feature-space sizes, tasks (regression / classification), and domains (real-estate, demographics, ecology, mobility, chemistry, finance, physics, high-dim NIPS-challenge). The choice mirrors Table 3 of the ProxySHAP paper so results are comparable to the datasets used there.

All loaders are registered in `DATASETS` and share one generic entry point:

```python
from Datasets.load_datasets import load, DATASETS, load_all

ds = load("ames", n_samples=1_000, n_features=32, seed=42)
```

Backwards-compatible `load_<name>()` wrappers are still exported for each dataset.

## Overview

| Key                 | Dataset              | Features    | Samples                | Task                     | Source       | Loader                      |
| ------------------- | -------------------- | ----------- | ---------------------- | ------------------------ | ------------ | --------------------------- |
| `california`        | California Housing   | 8           | 20 640                 | Regression               | sklearn      | `load_california_housing()` |
| `ames`              | Ames Housing         | ~79         | 1 460                  | Regression               | OpenML 42165 | `load_ames_housing()`       |
| `covertype`         | Forest Covertype     | 54          | 581 012 (50 k default) | Classification (7-class) | sklearn      | `load_covertype()`          |
| `adult_census`      | Adult Census         | 14          | 48 842                 | Classification (binary)  | OpenML 1590  | `load_adult_census()`       |
| `bike`              | Bike Sharing         | 12          | 17 379                 | Regression               | OpenML 42712 | `load_bike()`               |
| `qsar_biodeg`       | QSAR Biodegradation  | 41          | 1 055                  | Classification (binary)  | OpenML 46952 | `load_qsar_biodeg()`        |
| `bankruptcy`        | Taiwanese Bankruptcy | 64 (95 raw) | 6 819                  | Classification (binary)  | OpenML 46950 | `load_bankruptcy()`         |
| `superconductivity` | Superconductivity    | 81          | 21 263                 | Regression               | OpenML 46961 | `load_superconductivity()`  |
| `gisette`           | Gisette              | 5 000       | 7 000                  | Classification (binary)  | OpenML 41026 | `load_gisette()`            |

`load_all(seed=…)` returns every dataset **except** `gisette` (excluded so the smoke path stays cheap; call `load_gisette()` or `load("gisette", …)` explicitly).

---

## California Housing

**Source:** `sklearn.datasets.fetch_california_housing`

| Property | Value                             |
| -------- | --------------------------------- |
| Features | 8                                 |
| Samples  | 20 640                            |
| Task     | Regression                        |
| Target   | Median house value (100 000s USD) |

All features are numeric. No missing values. A well-established regression benchmark with a small, interpretable feature set — serves as the baseline for XAI comparisons.

**Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

---

## Ames Housing

**Source:** `sklearn.datasets.fetch_openml(data_id=42165)`

| Property | Value           |
| -------- | --------------- |
| Features | ~79             |
| Samples  | 1 460           |
| Task     | Regression      |
| Target   | SalePrice (USD) |

Rich mix of numeric and categorical features describing residential properties in Ames, Iowa. Same prediction task as California Housing (house prices) but roughly 10× more features, enabling evaluation of how XAI methods scale with feature count. Categorical columns are label-encoded and missing values are median/mode imputed by the loader.

**Feature groups:** Lot characteristics, building type & style, quality/condition ratings, room counts, garage & basement details, sale conditions.

---

## Forest Covertype

**Source:** `sklearn.datasets.fetch_covtype`

| Property | Value                                           |
| -------- | ----------------------------------------------- |
| Features | 54                                              |
| Samples  | 581 012 (loader returns stratified 50 k subset) |
| Task     | Multi-class classification (7 classes)          |
| Target   | Forest cover type (1–7)                         |

Cartographic features (elevation, slope, distances, soil type indicators) for 30×30 m forest patches in the Roosevelt National Forest, Colorado. All features are integer-valued. The large sample count and purely numeric features make this a good stress-test for computational cost of XAI methods. The loader subsamples to 50 000 rows by default (via `stratified_default_n`) to keep notebooks responsive.

**Feature groups:** Elevation, Aspect, Slope, Horizontal/Vertical distances to hydrology & roadways, Hillshade indices (×3), Wilderness area (×4 binary), Soil type (×40 binary).

---

## Adult Census

**Source:** `sklearn.datasets.fetch_openml(data_id=1590)`

| Property | Value                        |
| -------- | ---------------------------- |
| Features | 14                           |
| Samples  | 48 842                       |
| Task     | Binary classification        |
| Target   | Income bracket (≤50K / >50K) |

Mix of numeric and categorical features describing individuals from the 1994 US Census. Categorical columns are label-encoded and missing values are imputed with the column mode (categorical) or median (numeric). A standard fairness and classification benchmark — complements Covertype by adding a real-world tabular classification task with a smaller feature set.

**Features:** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

---

## Bike Sharing

**Source:** `sklearn.datasets.fetch_openml(data_id=42712)`

| Property | Value                       |
| -------- | --------------------------- |
| Features | 12                          |
| Samples  | 17 379                      |
| Task     | Regression                  |
| Target   | count (hourly rental count) |

Hourly bike rental counts for Capital Bikeshare (Washington DC, 2011–2012) with weather and calendar covariates. Referenced as a TabPFN-friendly small-feature regression benchmark in Table 3 of the reference paper. Adds a second small (n ≤ 15) regression benchmark alongside California Housing but with a mix of categorical (season, weather, working-day) and continuous (temp, humidity, wind) features that survive label-encoding.

**Features:** season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed.

---

## QSAR Biodegradation

**Source:** `sklearn.datasets.fetch_openml(data_id=46952)`

| Property | Value                    |
| -------- | ------------------------ |
| Features | 41                       |
| Samples  | 1 055                    |
| Task     | Binary classification    |
| Target   | Biodegradable (RB / NRB) |

Molecular-descriptor features (SpMax, SM6, F04, etc.) predicting whether a chemical compound is ready-biodegradable. Referenced as the TabArena QSAR Biodeg benchmark. Introduces a chemistry-domain classification task and fills the mid-size (30–50 features) region between Adult Census (14) and Bankruptcy (64) — a region that was previously unrepresented in the registry.

---

## Bankruptcy

**Source:** `sklearn.datasets.fetch_openml(data_id=46950)`

| Property | Value                                                                  |
| -------- | ---------------------------------------------------------------------- |
| Features | 64 (of 95 raw columns retained after preprocessing / variance ranking) |
| Samples  | 6 819                                                                  |
| Task     | Binary classification                                                  |
| Target   | company_bankrupt                                                       |

Financial-ratio features (profitability, leverage, cash-flow ratios) for Taiwanese companies over 1999–2009 predicting bankruptcy. Referenced as the TabArena Bankruptcy benchmark. Adds a real-world class-imbalanced financial classification task at the mid-large feature-count tier (~64 features), between QSAR Biodeg (41) and Superconductivity (81).

---

## Superconductivity

**Source:** `sklearn.datasets.fetch_openml(data_id=46961)`

| Property | Value             |
| -------- | ----------------- |
| Features | 81                |
| Samples  | 21 263            |
| Task     | Regression        |
| Target   | critical_temp (K) |

Physicochemical descriptors of superconducting materials predicting the critical temperature. Referenced as the TabArena Superconductivity benchmark. Provides a **high-dimensional regression** benchmark — previously the registry had no regression task with more than ~79 features (Ames), so this closes that gap and makes regression comparable to classification along the feature-count axis.

---

## Gisette

**Source:** `sklearn.datasets.fetch_openml(data_id=41026)`

| Property | Value                                            |
| -------- | ------------------------------------------------ |
| Features | 5 000                                            |
| Samples  | 7 000 (train + validation split of 13 500 total) |
| Task     | Binary classification                            |
| Target   | Digit class (−1 = digit 4, +1 = digit 9)         |

High-dimensional dataset from the NIPS 2003 feature selection challenge. Pixel-level features for handwritten digit images distinguishing 4 from 9. All features are integer-valued. Provides a high-dimensional stress-test for XAI methods — 5 000 features far exceeds the other datasets and exposes scalability limits of explanation algorithms. Loaded from sparse ARFF (`dense_from_sparse=True`) since OpenML cannot return sparse data as a DataFrame directly.

---

## Feature encoding across datasets

Datasets differ in how categorical and binary features are represented, which affects XAI explanations. The generic loader applies one uniform recipe (median-impute numeric, mode-impute + `.cat.codes` for non-numeric) so encoding is consistent across all datasets:

| Dataset             | Encoding       | Notes                                                                                                |
| ------------------- | -------------- | ---------------------------------------------------------------------------------------------------- |
| California Housing  | —              | All features continuous                                                                              |
| Ames Housing        | Label encoding | Nominal categoricals mapped to ordinal ints via `.cat.codes`; missing values imputed before encoding |
| Forest Covertype    | Native one-hot | 4 `Wilderness_Area_*` and 40 `Soil_Type_*` binary indicator columns in the raw data                  |
| Adult Census        | Label encoding | Same recipe as Ames                                                                                  |
| Bike Sharing        | Label encoding | Categorical calendar/weather columns → codes; continuous weather columns kept as-is                  |
| QSAR Biodegradation | —              | All features are pre-computed numeric molecular descriptors                                          |
| Bankruptcy          | —              | Pre-computed financial ratios; numeric                                                               |
| Superconductivity   | —              | Numeric physicochemical descriptors                                                                  |
| Gisette             | —              | Pixel-derived integer features                                                                       |

**Implication for feature selection:** Variance-based feature reduction (`VarianceThreshold` ranking) is used to control `n_features` across experiments. Because binary one-hot columns have variance bounded by `p·(1−p) ≤ 0.25` while continuous features can have variance in the thousands or millions, Covertype's continuous features are always ranked above its binary indicator columns. This is an accepted trade-off: the selection method is held fixed across experiments as a preprocessing constant, not as an optimised choice. The goal is to study XAI library behaviour as a function of feature count, not to maximise model performance.

---

## API

The public surface is small and regular:

```python
from Datasets.load_datasets import (
    DATASETS,       # dict[str, DatasetSpec] – single source of truth
    DatasetSpec,    # dataclass describing a dataset
    load,           # generic loader: load(key, n_samples=..., n_features=..., seed=...)
    load_all,       # bulk loader (skips gisette by default)
    # backwards-compatible per-dataset wrappers
    load_california_housing, load_ames_housing, load_covertype, load_adult_census,
    load_bike, load_qsar_biodeg, load_bankruptcy, load_superconductivity, load_gisette,
)

# generic path
ds = load("bike", n_samples=1_000, n_features=8, seed=42)
X, y = ds["X"], ds["y"]
print(ds["name"], ds["task"], X.shape)

# bulk load (california, ames, covertype, adult_census, bike, qsar_biodeg, bankruptcy, superconductivity)
datasets = load_all(seed=42)
for key, ds in datasets.items():
    print(key, ds["X"].shape, ds["task"])
```

Each loader returns a `dict` with keys: `name`, `task`, `X` (DataFrame), `y` (Series), `feature_names`, `target_name`.

## Adding a new dataset

Add one entry to the `DATASETS` registry:

```python
"my_new_dataset": DatasetSpec(
    name="My New Dataset",
    task="classification",         # or "regression"
    fetch=_openml(<id>),           # or any zero-arg callable returning an sklearn Bunch
    target="my_target_col",        # optional; defaults to bunch.target_names[0]
    drop_columns=("Id",),          # optional
    target_transform=_binary_from_positive_label("positive"),  # optional
    stratified_default_n=None,     # optional; auto-subsample when caller passes n_samples=None
    dense_from_sparse=False,       # set True only for sparse ARFF payloads
),
```

The generic `load()` handles the rest (drop → target → impute+encode → subsample → variance-based feature reduction). No new code path is required for typical OpenML datasets.

> **Note:** `load_gisette()` is not included in `load_all()` and must be called individually.
