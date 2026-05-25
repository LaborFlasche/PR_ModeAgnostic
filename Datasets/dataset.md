# Datasets

Three tabular datasets are used to evaluate XAI libraries across different feature-space sizes.

## Overview

| Dataset | Features | Samples | Task | Loader |
|---|---|---|---|---|
| California Housing | 8 | 20 640 | Regression | `load_california_housing()` |
| Ames Housing | ~79 | 1 460 | Regression | `load_ames_housing()` |
| Forest Covertype | 54 | 581 012 (50 k subset) | Classification | `load_covertype()` |

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

## Usage

```python
from Datasets.load_datasets import (
    load_california_housing,
    load_ames_housing,
    load_covertype,
    load_all,
)

ds = load_california_housing()
X, y = ds["X"], ds["y"]
print(ds["name"], ds["task"], X.shape)

# Or load all at once
datasets = load_all()
for key, ds in datasets.items():
    print(key, ds["X"].shape, ds["task"])
```

Each loader returns a `dict` with keys: `name`, `task`, `X` (DataFrame), `y` (Series), `feature_names`, `target_name`.
