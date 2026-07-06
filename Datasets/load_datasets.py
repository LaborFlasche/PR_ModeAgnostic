"""
Dataset loading utilities for XAI library comparison.

Public surface (deliberately tiny):

    Dataset            – an ``Enum`` whose members *are* the supported datasets.
    Dataset.<X>.load() – load one dataset.
    load(dataset)      – load one dataset (accepts a ``Dataset`` or its name str).
    load_all(...)      – load several datasets at once.

Every ``load`` returns a consistent dict:

    X              : pd.DataFrame  – features
    y              : pd.Series     – target
    feature_names  : list[str]
    target_name    : str
    task           : "regression" | "classification"
    name           : str

The datasets are chosen to spread across three axes simultaneously — feature
count, feature types (numeric / categorical / mixed), and domain — so that XAI
library behaviour can be studied as those axes vary rather than as a single
confounded "size" knob:

    key            feats  task            domain      types
    california       8    regression      housing     numeric
    bike            12    regression      mobility    mixed
    adult_census    14    classification  social      mixed
    qsar_biodeg     41    classification  chemistry   numeric
    diabetes_130    47    classification  medical     mixed
    covertype       54    classification  ecology     numeric + binary
    bankruptcy      64    classification  finance     numeric
    ames            79    regression      housing     mixed
    gisette       5000    classification  image       numeric
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


def _openml(data_id: int, *, as_frame: bool = True) -> Callable[[], Any]:
    """Zero-arg fetcher for an OpenML dataset id (used inside the registry)."""
    return lambda: fetch_openml(data_id=data_id, as_frame=as_frame, parser="auto")


# --------------------------------------------------------------------------- #
# Dataset spec                                                                #
# --------------------------------------------------------------------------- #

TargetTransform = Callable[[pd.Series], pd.Series]


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of a dataset — the value carried by each ``Dataset``.

    Attributes:
        name: Human-readable label recorded in the result dict.
        task: "regression" or "classification".
        domain: Coarse subject area (housing, medical, ...), for spread bookkeeping.
        feature_types: "numeric", "categorical", or "mixed" — documents the
            categorical/continuous makeup so the benchmark set can be checked for
            balance at a glance.
        fetch: Zero-arg callable returning an sklearn ``Bunch``.
        target: Target column name; ``None`` uses ``bunch.target_names[0]``.
        drop_columns: Columns to drop before feature/target split (e.g. "Id").
        target_transform: Optional post-processing for the raw target column
            (e.g. binarize ``">50K"``). If ``None``, the target is coerced to
            float for regression and to integer codes for classification.
        stratified_default_n: If set, when the caller passes ``n_samples=None``
            the loader stratified-samples to this size (keeps covertype responsive).
        dense_from_sparse: If ``True``, fetch with ``as_frame=False`` and densify
            a sparse ARFF payload (gisette).
    """
    name: str
    task: Literal["regression", "classification"]
    domain: str
    feature_types: Literal["numeric", "categorical", "mixed"]
    fetch: Callable[[], Any]
    target: Optional[str] = None
    drop_columns: tuple[str, ...] = ()
    target_transform: Optional[TargetTransform] = None
    stratified_default_n: Optional[int] = None
    dense_from_sparse: bool = False


def _load_spec(spec: DatasetSpec, n_samples: int | None, n_features: int | None,
               *, seed: int) -> dict:
    """Run the shared pipeline for a single spec.

    Order: fetch -> drop id cols -> target extraction/transform ->
    impute + label-encode features -> optional (stratified) subsample ->
    optional variance-based feature reduction.
    """
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
# The registry: one enum member per supported dataset                         #
# --------------------------------------------------------------------------- #

class Dataset(Enum):
    """Supported datasets. Each member's value is its :class:`DatasetSpec`.

    Members are the single source of truth — there is no parallel string
    registry or per-dataset ``load_x`` helper. Look up by name with
    ``Dataset["ADULT_CENSUS"]`` (member names are the config keys, upper-cased).
    """

    CALIFORNIA_HOUSING = DatasetSpec(
        name="California Housing",
        task="regression",
        domain="housing",
        feature_types="numeric",
        fetch=lambda: fetch_california_housing(as_frame=True),
    )
    BIKE = DatasetSpec(
        name="Bike Sharing",
        task="regression",
        domain="mobility",
        feature_types="mixed",
        fetch=_openml(42712),
    )
    ADULT_CENSUS = DatasetSpec(
        name="Adult Census",
        task="classification",
        domain="social",
        feature_types="mixed",
        fetch=_openml(1590),
        target_transform=_binary_from_positive_label(">50K"),
    )
    QSAR_BIODEG = DatasetSpec(
        name="QSAR Biodegradation",
        task="classification",
        domain="chemistry",
        feature_types="numeric",
        fetch=_openml(46952),
    )
    DIABETES_130 = DatasetSpec(
        name="Diabetes 130-US",
        task="classification",
        domain="medical",
        feature_types="mixed",
        fetch=_openml(46922),
    )
    COVERTYPE = DatasetSpec(
        name="Forest Covertype",
        task="classification",
        domain="ecology",
        feature_types="numeric",  # numeric + native binary indicator columns
        fetch=lambda: fetch_covtype(as_frame=True),
        target="Cover_Type",
        stratified_default_n=50_000,
    )
    BANKRUPTCY = DatasetSpec(
        name="Bankruptcy",
        task="classification",
        domain="finance",
        feature_types="numeric",
        fetch=_openml(46950),
    )
    AMES_HOUSING = DatasetSpec(
        name="Ames Housing",
        task="regression",
        domain="housing",
        feature_types="mixed",
        fetch=_openml(42165),
        drop_columns=("Id",),
    )
    GISETTE = DatasetSpec(
        name="Gisette",
        task="classification",
        domain="image",
        feature_types="numeric",
        fetch=lambda: fetch_openml(
            data_id=41026, as_frame=False, parser="auto"),
        dense_from_sparse=True,
    )

    # -- convenience API ---------------------------------------------------- #
    @property
    def spec(self) -> DatasetSpec:
        return self.value

    @property
    def is_regression(self) -> bool:
        return self.spec.task == "regression"

    def load(self, n_samples: int | None = None, n_features: int | None = None,
             *, seed: int) -> dict:
        """Load this dataset. See module docstring for the returned dict shape."""
        return _load_spec(self.spec, n_samples, n_features, seed=seed)

    def __str__(self) -> str:  # nicer prints / f-strings
        return self.spec.name


# --------------------------------------------------------------------------- #
# Module-level convenience functions                                          #
# --------------------------------------------------------------------------- #

def load(dataset: "Dataset | str", n_samples: int | None = None,
         n_features: int | None = None, *, seed: int) -> dict:
    """Load a single dataset, accepting a :class:`Dataset` or its name string.

    ``load("adult_census", seed=42)`` and ``load(Dataset.ADULT_CENSUS, seed=42)``
    are equivalent.
    """
    if isinstance(dataset, str):
        try:
            dataset = Dataset[dataset.upper()]
        except KeyError:
            raise KeyError(
                f"Unknown dataset '{dataset}'. Available: "
                f"{[d.name.lower() for d in Dataset]}"
            ) from None
    return dataset.load(n_samples, n_features, seed=seed)


def load_all(datasets: "list[Dataset] | None" = None, *, seed: int,
             ) -> dict["Dataset", dict]:
    """Load several datasets at once, keyed by :class:`Dataset` member.

    Args:
        datasets: Which datasets to load. Defaults to *every* member of
            :class:`Dataset`. Pass a subset list for a cheaper sweep, e.g.
            ``load_all([Dataset.BIKE, Dataset.ADULT_CENSUS], seed=42)``.
        seed: Benchmark seed threaded into every subsampler.
    """
    selected = list(datasets) if datasets is not None else list(Dataset)
    return {d: d.load(seed=seed) for d in selected}


if __name__ == "__main__":
    # Light smoke set (skips the 581k-row covertype and 5000-feature gisette).
    demo = [d for d in Dataset if d not in (
        Dataset.COVERTYPE, Dataset.GISETTE)]
    for d, data in load_all(demo, seed=42).items():
        s = d.spec
        print(f"{d.name.lower():18s} {data['X'].shape[0]:>6} samples "
              f"{data['X'].shape[1]:>4} feats  {s.task:<14} "
              f"{s.domain:<10} {s.feature_types}")
