"""
Dataset loading utilities for XAI library comparison.

The module exposes a small registry (``DATASETS``) mapping a short key to a
``DatasetSpec`` and a single generic ``load()`` function that performs all
shared preprocessing (drop id columns, impute + label-encode, optional
variance-based feature selection, optional subsampling). Backward-compatible
``load_<name>()`` wrappers are kept so existing call sites in
``Models/dataset_and_models.py`` and elsewhere don't have to change.

Every loader returns a consistent dict with:
    X              : pd.DataFrame  – features
    y              : pd.Series     – target
    feature_names  : list[str]
    target_name    : str
    task           : "regression" | "classification"
    name           : str
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml
from sklearn.feature_selection import VarianceThreshold


# --------------------------------------------------------------------------- #
# Shared preprocessing helpers                                                #
# --------------------------------------------------------------------------- #

def _select_features_by_variance(X: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """Keep the top-n features ranked by variance (highest first)."""
    if n_features >= X.shape[1]:
        return X
    selector = VarianceThreshold()
    selector.fit(X)
    top_idx = np.argsort(selector.variances_)[::-1][:n_features]
    return X.iloc[:, sorted(top_idx)]


def _subsample(X: pd.DataFrame, y: pd.Series, n_samples: int,
               seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Random subsample without replacement, reproducible via the benchmark seed."""
    if n_samples >= len(X):
        return X, y
    idx = X.sample(n=n_samples, random_state=seed).index
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def _stratified_subsample(X: pd.DataFrame, y: pd.Series, n_samples: int,
                          seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample keyed on ``y``; used for very large imbalanced sets."""
    if n_samples >= len(X):
        return X, y
    frac = n_samples / len(y)
    idx = (
        y.groupby(y)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .index.get_level_values(1)
    )
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def _impute_and_encode(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns, mode-impute + label-encode categoricals."""
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            mode = X[col].mode()
            fill = mode.iloc[0] if len(mode) else ""
            X[col] = X[col].fillna(fill)
            X[col] = X[col].astype("category").cat.codes
        else:
            X[col] = X[col].fillna(X[col].median())
    return X


def _binary_from_positive_label(positive_label: str) -> Callable[[pd.Series], pd.Series]:
    """Build a target-transform that maps ``positive_label`` -> 1, else 0.

    Handles stringly-typed OpenML labels (e.g. ``">50K."`` with a trailing period)
    by stripping whitespace and periods before comparison.
    """
    def _tx(y: pd.Series) -> pd.Series:
        s = y.astype(str).str.strip().str.replace(".", "", regex=False)
        return (s == positive_label).astype(int)
    return _tx


# --------------------------------------------------------------------------- #
# Dataset spec + registry                                                     #
# --------------------------------------------------------------------------- #

TargetTransform = Callable[[pd.Series], pd.Series]


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of a dataset.

    Attributes:
        name: Human-readable label recorded in the result dict.
        task: "regression" or "classification".
        fetch: Zero-arg callable returning an sklearn ``Bunch``.
        target: Target column name; ``None`` uses ``bunch.target_names[0]``.
        drop_columns: Columns to drop before feature/target split (e.g. "Id").
        target_transform: Optional post-processing for the raw target column
            (e.g. binarize ``">50K"``). If ``None``, the target is coerced to
            float for regression and to integer codes for classification.
        stratified_default_n: If set, when the caller passes ``n_samples=None``
            the loader stratified-samples to this size (used to keep covtype
            responsive by default).
        dense_from_sparse: If ``True``, fetch with ``as_frame=False`` and
            densify a sparse ARFF payload (gisette).
    """
    name: str
    task: Literal["regression", "classification"]
    fetch: Callable[[], Any]
    target: Optional[str] = None
    drop_columns: tuple[str, ...] = ()
    target_transform: Optional[TargetTransform] = None
    stratified_default_n: Optional[int] = None
    dense_from_sparse: bool = False


def _openml(data_id: int, *, as_frame: bool = True):
    """Small convenience wrapper for OpenML fetchers used inside the registry."""
    return lambda: fetch_openml(data_id=data_id, as_frame=as_frame, parser="auto")


DATASETS: dict[str, DatasetSpec] = {
    # ---- original set --------------------------------------------------- #
    "california": DatasetSpec(
        name="California Housing",
        task="regression",
        fetch=lambda: fetch_california_housing(as_frame=True),
    ),
    "ames": DatasetSpec(
        name="Ames Housing",
        task="regression",
        fetch=_openml(42165),
        drop_columns=("Id",),
    ),
    "covertype": DatasetSpec(
        name="Forest Covertype",
        task="classification",
        fetch=lambda: fetch_covtype(as_frame=True),
        target="Cover_Type",
        stratified_default_n=50_000,
    ),
    "adult_census": DatasetSpec(
        name="Adult Census",
        task="classification",
        fetch=_openml(1590),
        target_transform=_binary_from_positive_label(">50K"),
    ),
    "gisette": DatasetSpec(
        name="Gisette",
        task="classification",
        fetch=lambda: fetch_openml(
            data_id=41026, as_frame=False, parser="auto"),
        dense_from_sparse=True,
    ),

    # ---- additions (Table 3) ------------------------------------------- #
    # Small regression, TabPFN-tier — complements ames on the low end.
    "bike": DatasetSpec(
        name="Bike Sharing",
        task="regression",
        fetch=_openml(42712),
    ),
    # Chemical binary classification, ~41 features — new domain and fills the
    # gap between adult_census (14) and bankruptcy (64).
    "qsar_biodeg": DatasetSpec(
        name="QSAR Biodegradation",
        task="classification",
        fetch=_openml(46952),
    ),
    # Financial binary classification, ~64 features — mid-large tabular.
    "bankruptcy": DatasetSpec(
        name="Bankruptcy",
        task="classification",
        fetch=_openml(46950),
    ),
    # Physics regression, ~81 features — the only high-dim regression benchmark
    # in the registry (previously nothing sat between ames and gisette on the
    # regression side).
    "superconductivity": DatasetSpec(
        name="Superconductivity",
        task="regression",
        fetch=_openml(46961),
    ),
}


# --------------------------------------------------------------------------- #
# Generic loader                                                              #
# --------------------------------------------------------------------------- #

def load(key: str, n_samples: int | None = None, n_features: int | None = None,
         *, seed: int) -> dict:
    """Load a registered dataset by short key.

    Applies (in order): drop id columns, target extraction + transform,
    impute + label-encode features, optional (stratified) subsample, optional
    variance-based feature selection.
    """
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset key '{key}'. Available: {sorted(DATASETS)}"
        )
    spec = DATASETS[key]
    bunch = spec.fetch()

    # ---- extract X, y ---------------------------------------------------- #
    if spec.dense_from_sparse:
        # Sparse ARFF (gisette) can't be returned as a DataFrame — densify manually.
        import scipy.sparse as sp

        data = bunch.data
        data = data.toarray() if sp.issparse(data) else np.asarray(data)
        feature_names = (
            list(bunch.feature_names)
            if getattr(bunch, "feature_names", None) is not None
            else [f"f{i}" for i in range(data.shape[1])]
        )
        X = pd.DataFrame(data, columns=feature_names)
        y_raw = pd.Series(np.asarray(bunch.target))
        target_name = (
            bunch.target_names[0]
            if getattr(bunch, "target_names", None)
            else "target"
        )
    else:
        df: pd.DataFrame = bunch.frame.copy()
        if spec.drop_columns:
            df = df.drop(columns=list(spec.drop_columns), errors="ignore")
        target_name = spec.target or bunch.target_names[0]
        feature_cols = [c for c in df.columns if c != target_name]
        X = df[feature_cols].copy()
        y_raw = df[target_name]

    # ---- target ---------------------------------------------------------- #
    if spec.target_transform is not None:
        y = spec.target_transform(y_raw)
    elif spec.task == "regression":
        y = y_raw.astype(float)
    else:
        # classification default: keep numeric ints, otherwise label-encode
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.astype(int)
        else:
            y = y_raw.astype("category").cat.codes.astype(int)
    y = pd.Series(y).reset_index(drop=True)

    # ---- features -------------------------------------------------------- #
    if not spec.dense_from_sparse:
        X = _impute_and_encode(X)
    X = X.reset_index(drop=True)

    # ---- (stratified) subsample ----------------------------------------- #
    if n_samples is None and spec.stratified_default_n is not None:
        X, y = _stratified_subsample(X, y, spec.stratified_default_n, seed)
    elif n_samples is not None:
        # Stratify for classification when possible; fall back to random.
        if spec.task == "classification" and y.nunique() > 1:
            try:
                X, y = _stratified_subsample(X, y, n_samples, seed)
            except Exception:
                X, y = _subsample(X, y, n_samples, seed)
        else:
            X, y = _subsample(X, y, n_samples, seed)

    # ---- feature reduction ---------------------------------------------- #
    if n_features is not None:
        X = _select_features_by_variance(X, n_features)

    return {
        "name": spec.name,
        "task": spec.task,
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": target_name,
    }


# --------------------------------------------------------------------------- #
# Backwards-compatible wrappers                                               #
# --------------------------------------------------------------------------- #
# These preserve the historical API used across the codebase (Models,
# tests, notebooks). Every wrapper is a one-liner over ``load()``.

def _wrapper(key: str):
    def _load(n_samples: int | None = None, n_features: int | None = None,
              *, seed: int) -> dict:
        return load(key, n_samples=n_samples, n_features=n_features, seed=seed)
    _load.__name__ = f"load_{key}"
    _load.__doc__ = f"Convenience wrapper around ``load('{key}', ...)``."
    return _load


load_california_housing = _wrapper("california")
load_ames_housing = _wrapper("ames")
load_covertype = _wrapper("covertype")
load_adult_census = _wrapper("adult_census")
load_gisette = _wrapper("gisette")
load_bike = _wrapper("bike")
load_qsar_biodeg = _wrapper("qsar_biodeg")
load_bankruptcy = _wrapper("bankruptcy")
load_superconductivity = _wrapper("superconductivity")


# Datasets included in the default ``load_all`` sweep. Very large or very
# high-dimensional datasets (covertype's full 581k rows, gisette's 5k feats)
# are excluded so smoke-testing stays cheap; call the wrappers explicitly if
# you need them.
DEFAULT_LOAD_ALL: tuple[str, ...] = (
    "california",
    "ames",
    "covertype",
    "adult_census",
    "bike",
    "qsar_biodeg",
    "bankruptcy",
    "superconductivity",
)


def load_all(*, seed: int, keys: tuple[str, ...] | None = None) -> dict[str, dict]:
    """Load a curated set of datasets keyed by short name.

    Args:
        seed: Benchmark seed threaded into every subsampler.
        keys: Optional subset of ``DATASETS`` keys. Defaults to
            :data:`DEFAULT_LOAD_ALL` (skips gisette).
    """
    selected = keys if keys is not None else DEFAULT_LOAD_ALL
    return {k: load(k, seed=seed) for k in selected}


if __name__ == "__main__":
    datasets = load_all(seed=42)
    for name, data in datasets.items():
        print(f"{name}: {data['X'].shape[0]} samples, "
              f"{data['X'].shape[1]} features ({data['task']})")
