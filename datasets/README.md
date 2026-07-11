# Datasets

A curated set of tabular datasets used to evaluate XAI libraries. They are chosen to spread across **three axes simultaneously** — feature count, feature types (numeric / categorical / mixed), and domain — so that library behaviour can be studied as those axes vary independently rather than as a single confounded "size" knob.

## Design

The whole module is one enum. Each `Dataset` member _is_ a supported dataset, and its value is a `DatasetSpec` describing how to fetch and preprocess it. There is **no** parallel string registry and **no** per-dataset `load_x()` helper — the previous design had both and it was redundant.

```python
from datasets.load_datasets import Dataset, load_dataset, load_all_datasets

# load one, via the enum member …
ds = Dataset.ADULT_CENSUS.load_dataset(n_samples=1_000, n_features=8, seed=42)

# … or via its name (config key), whichever is terser at the call site
ds = load_dataset("adult_census", seed=42)

# load several (defaults to every member)
all_data = load_all_datasets(seed=42)                         # dict[Dataset, dict]
subset   = load_all_datasets([Dataset.BIKE, Dataset.DIABETES_130], seed=42)
```

Every load returns a `dict` with keys: `name`, `task`, `X` (DataFrame), `y` (Series), `feature_names`, `target_name`.

## Overview

| Member               | Config key           | Features | Samples                | Task                     | Domain    | Types            | Source       |
| -------------------- | -------------------- | -------- | ---------------------- | ------------------------ | --------- | ---------------- | ------------ |
| `CALIFORNIA_HOUSING` | `california_housing` | 8        | 20 640                 | Regression               | housing   | numeric          | sklearn      |
| `BIKE`               | `bike`               | 12       | 17 379                 | Regression               | mobility  | mixed            | OpenML 42712 |
| `ADULT_CENSUS`       | `adult_census`       | 14       | 48 842                 | Classification           | social    | mixed            | OpenML 1590  |
| `QSAR_BIODEG`        | `qsar_biodeg`        | 41       | 1 055                  | Classification           | chemistry | numeric          | OpenML 46952 |
| `DIABETES_130`       | `diabetes_130`       | 47       | 101 766                | Classification           | medical   | mixed            | OpenML 46922 |
| `COVERTYPE`          | `covertype`          | 54       | 581 012 (50 k default) | Classification (7-class) | ecology   | numeric + binary | sklearn      |
| `BANKRUPTCY`         | `bankruptcy`         | 64       | 6 819                  | Classification           | finance   | numeric          | OpenML 46950 |
| `AMES_HOUSING`       | `ames_housing`       | ~79      | 1 460                  | Regression               | housing   | mixed            | OpenML 42165 |
| `GISETTE`            | `gisette`            | 5 000    | 7 000                  | Classification           | image     | numeric          | OpenML 41026 |

### Spread at a glance

- **Feature count:** roughly log-spaced from 8 to 5 000.
- **Task:** 3 regression (housing ×2, mobility) and 6 classification.
- **Feature types:** 4 mixed (categorical + numeric), 5 purely numeric — both the low and high feature-count ends contain each type.
- **Domain:** housing, mobility, social, chemistry, medical, ecology, finance, image — no single domain dominates a feature-count band.

> **Note on housing datasets.** California Housing (8, numeric) and Ames Housing (~79, mixed) are both housing regression tasks but sit at opposite ends of the feature-count and feature-type axes, so they are _not_ redundant — one is the small all-numeric baseline, the other the large mixed-type case. (An earlier draft also included Superconductivity as an ~81-feature regression task; it was dropped because it overlapped Ames on both feature count and task while adding little on the type/domain axes.)

---

## California Housing

**Source:** `sklearn.datasets.fetch_california_housing` · **Target:** median house value (100 000s USD)

8 numeric features, no missing values. The small, all-numeric regression baseline.

**Features:** MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

---

## Bike Sharing

**Source:** `fetch_openml(data_id=42712)` · **Target:** `count` (hourly rentals)

Hourly Capital Bikeshare counts (Washington DC, 2011–2012). A small **mixed-type** regression task: categorical calendar/weather columns (season, weathersit, workingday) alongside continuous ones (temp, humidity, windspeed). Complements California Housing at the low feature-count end but with categoricals.

**Features:** season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed.

---

## Adult Census

**Source:** `fetch_openml(data_id=1590)` · **Target:** income bracket (≤50K / >50K, binarized to 1 = >50K)

1994 US Census records; the canonical small mixed-type classification benchmark. Categorical columns are label-encoded, missing values mode/median imputed.

**Features:** age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

---

## QSAR Biodegradation

**Source:** `fetch_openml(data_id=46952)` · **Target:** `Biodegradable` (RB / NRB)

Molecular-descriptor features predicting ready-biodegradability of chemical compounds. A **chemistry-domain, all-numeric** classification task filling the mid feature-count band (~41).

---

## Diabetes 130-US

**Source:** `fetch_openml(data_id=46922)` · **Target:** early hospital readmission

Ten years (1999–2008) of US hospital encounter records for diabetic patients. A **medical-domain, mixed-type** classification task (~47 features: demographics, diagnoses, medications, prior-visit counts). Adds the medical domain and a second mixed-type classification case in the mid feature-count band.

---

## Forest Covertype

**Source:** `sklearn.datasets.fetch_covtype` · **Target:** `Cover_Type` (7 classes)

Cartographic features for 30×30 m forest patches (Roosevelt National Forest, Colorado). All integer-valued, with 4 `Wilderness_Area_*` and 40 `Soil_Type_*` native binary indicator columns. The only multi-class task and the large-sample stress case; the loader stratified-subsamples to 50 000 rows by default (`stratified_default_n`).

**Feature groups:** Elevation, Aspect, Slope, distances to hydrology & roadways, Hillshade (×3), Wilderness area (×4 binary), Soil type (×40 binary).

---

## Bankruptcy

**Source:** `fetch_openml(data_id=46950)` · **Target:** `company_bankrupt`

Financial-ratio features for Taiwanese companies (1999–2009). A **finance-domain, all-numeric**, class-imbalanced classification task at the mid-large feature-count tier (~64).

---

## Ames Housing

**Source:** `fetch_openml(data_id=42165)` · **Target:** SalePrice (USD)

Residential property sales in Ames, Iowa. A **large mixed-type** regression task (~79 features): same prediction domain as California Housing but ~10× more features and a rich categorical/numeric mix. Categorical columns are label-encoded, missing values median/mode imputed.

**Feature groups:** lot characteristics, building type & style, quality/condition ratings, room counts, garage & basement details, sale conditions.

---

## Gisette

**Source:** `fetch_openml(data_id=41026)` · **Target:** digit class (−1 = 4, +1 = 9)

NIPS 2003 feature-selection challenge: 5 000 pixel-derived integer features distinguishing handwritten 4s from 9s. The high-dimensional stress-test that exposes scalability limits. Loaded from sparse ARFF (`dense_from_sparse=True`) since OpenML cannot return sparse data as a DataFrame.

---

## Preprocessing pipeline

`Dataset.load_dataset()` (implemented by the shared `_load_spec`) applies, in order:

1. **fetch** the raw `Bunch` (`DatasetSpec.fetch`),
2. **drop** id columns (`drop_columns`, e.g. Ames' `Id`),
3. **target** extraction + optional transform (`target`, `target_transform`); default coercion is float for regression and integer codes for classification,
4. **impute + label-encode** features — numeric columns median-imputed, non-numeric mode-imputed then mapped to `.cat.codes` (one uniform recipe for all datasets),
5. optional **(stratified) subsample** to `n_samples` (stratified for classification; `stratified_default_n` auto-applies when `n_samples=None`, used by Covertype),
6. optional **variance-based feature reduction** to `n_features` (top-N highest-variance columns).

### Feature encoding note

Because label-encoded / one-hot binary columns have variance bounded by `p·(1−p) ≤ 0.25` while continuous features can have variance in the thousands, the variance ranking used for `n_features` always keeps continuous features ahead of binary indicators (e.g. Covertype's soil-type flags). This is an accepted, fixed preprocessing constant — the goal is to study XAI behaviour as a function of feature count, not to maximise model accuracy.

---

## Adding a dataset

Add one member to the `Dataset` enum:

```python
MY_DATASET = DatasetSpec(
    name="My Dataset",
    task="classification",              # or "regression"
    domain="medical",                   # for spread bookkeeping
    feature_types="mixed",              # "numeric" | "categorical" | "mixed"
    fetch=_openml(<id>),                # or any zero-arg callable -> sklearn Bunch
    target="my_target_col",             # optional; defaults to bunch.target_names[0]
    drop_columns=("Id",),               # optional
    target_transform=_binary_from_positive_label("positive"),  # optional
    stratified_default_n=None,          # optional; auto-subsample when n_samples=None
    dense_from_sparse=False,            # True only for sparse ARFF payloads
)
```

The member name (lower-cased) becomes the config key. No new code path is needed for typical OpenML datasets — the generic pipeline handles the rest, and every consumer (`Models`, `slurm/run_benchmark.py`) picks it up automatically via `Dataset[key.upper()]`.
