# Datasets

Three tabular datasets are used to evaluate XAI libraries across different feature-space sizes.

## Overview

| Dataset | Features | Samples | Task | Loader |
|---|---|---|---|---|
| California Housing | 8 | 20 640 | Regression | `load_california_housing()` |
| Ames Housing | ~79 | 1 460 | Regression | `load_ames_housing()` |
| Forest Covertype | 54 | 581 012 (50 k subset) | Classification | `load_covertype()` |
| Adult Census | 14 | 48 842 | Classification | `load_adult_census()` |
| Gisette | 5 000 | 7 000 | Classification | `load_gisette()` |

---

## California Housing

**Source:** `sklearn.datasets.fetch_california_housing`

| Property | Value |
|---|---|
| Features | 8 |
| Samples | 20 640 |
| Task | Regression |
| Target | Median house value (100 000s USD) |

All features are numeric. No missing values. A well-established regression benchmark with a small, interpretable feature set — serves as the baseline for XAI comparisons.

**Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

---

## Ames Housing

**Source:** `sklearn.datasets.fetch_openml("house_prices", version=3)`

| Property | Value |
|---|---|
| Features | ~79 |
| Samples | 1 460 |
| Task | Regression |
| Target | SalePrice (USD) |

Rich mix of numeric and categorical features describing residential properties in Ames, Iowa. Same prediction task as California Housing (house prices) but roughly 10× more features, enabling evaluation of how XAI methods scale with feature count.  Categorical columns are label-encoded and missing values are median/mode imputed by the loader.

**Feature groups:** Lot characteristics, building type & style, quality/condition ratings, room counts, garage & basement details, sale conditions.

---

## Forest Covertype

**Source:** `sklearn.datasets.fetch_covtype`

| Property | Value |
|---|---|
| Features | 54 |
| Samples | 581 012 (loader returns stratified 50 k subset) |
| Task | Multi-class classification (7 classes) |
| Target | Forest cover type (1–7) |

Cartographic features (elevation, slope, distances, soil type indicators) for 30×30 m forest patches in the Roosevelt National Forest, Colorado. All features are integer-valued. The large sample count and purely numeric features make this a good stress-test for computational cost of XAI methods.  The loader subsamples to 50 000 rows by default to keep notebooks responsive.

**Feature groups:** Elevation, Aspect, Slope, Horizontal/Vertical distances to hydrology & roadways, Hillshade indices (×3), Wilderness area (×4 binary), Soil type (×40 binary).

---

## Adult Census

**Source:** `sklearn.datasets.fetch_openml(data_id=1590)`

| Property | Value |
|---|---|
| Features | 14 |
| Samples | 48 842 |
| Task | Binary classification |
| Target | Income bracket (≤50K / >50K) |

Mix of numeric and categorical features describing individuals from the 1994 US Census. Categorical columns are label-encoded and missing values are imputed with the column mode (categorical) or median (numeric). A standard fairness and classification benchmark — complements Covertype by adding a real-world tabular classification task with a smaller feature set.

**Features:** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

---

## Gisette

**Source:** `sklearn.datasets.fetch_openml(data_id=41026)`

| Property | Value |
|---|---|
| Features | 5 000 |
| Samples | 7 000 (train + validation split of 13 500 total) |
| Task | Binary classification |
| Target | Digit class (−1 = digit 4, +1 = digit 9) |

High-dimensional dataset from the NIPS 2003 feature selection challenge. Pixel-level features for handwritten digit images distinguishing 4 from 9. All features are integer-valued. Provides a high-dimensional stress-test for XAI methods — 5 000 features far exceeds the other datasets and exposes scalability limits of explanation algorithms.

**Feature groups:** 5 000 integer-valued pixel and derived features (no named subgroups).

---

## Feature encoding across datasets

Datasets differ in how categorical and binary features are represented, which affects XAI explanations:

| Dataset | Encoding | Notes |
|---|---|---|
| California Housing | — | All features are continuous; no categoricals |
| Ames Housing | Label encoding | Nominal categoricals mapped to ordinal integers via `.cat.codes`; missing values imputed before encoding |
| Forest Covertype | Native one-hot | 4 `Wilderness_Area_*` and 40 `Soil_Type_*` binary indicator columns are part of the raw dataset |
| Adult Census | Label encoding | Same approach as Ames Housing |
| Gisette | — | All features are integer-valued pixel derivatives; no categoricals |

**Implication for feature selection:** Variance-based feature reduction (`VarianceThreshold` ranking) is used to control `n_features` across experiments. Because binary one-hot columns have variance bounded by `p*(1−p) ≤ 0.25` while continuous features can have variance in the thousands or millions, Covertype's continuous features are always ranked above its binary indicator columns. This is an accepted trade-off: the selection method is held fixed across experiments as a preprocessing constant, not as an optimised choice. The goal is to study XAI library behaviour as a function of feature count, not to maximise model performance.

---

## Usage

```python
from Datasets.load_datasets import (
    load_california_housing,
    load_ames_housing,
    load_covertype,
    load_adult_census,
    load_gisette,
    load_all,
)

ds = load_california_housing()
X, y = ds["X"], ds["y"]
print(ds["name"], ds["task"], X.shape)

# Or load all at once (california, ames, covertype, adult_census)
datasets = load_all()
for key, ds in datasets.items():
    print(key, ds["X"].shape, ds["task"])
```

Each loader returns a `dict` with keys: `name`, `task`, `X` (DataFrame), `y` (Series), `feature_names`, `target_name`.

> **Note:** `load_gisette()` is not included in `load_all()` and must be called individually.
