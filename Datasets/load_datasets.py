"""
Dataset loading utilities for XAI library comparison.

Each loader returns a consistent dict with:
    X              : pd.DataFrame  – features
    y              : pd.Series     – target
    feature_names  : list[str]
    target_name    : str
    task           : "regression" | "classification"
    name           : str
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml
from sklearn.feature_selection import VarianceThreshold


def _select_features_by_variance(X: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """Keep the top-n features ranked by variance (highest first)."""
    if n_features >= X.shape[1]:
        return X
    selector = VarianceThreshold()
    selector.fit(X)
    top_idx = np.argsort(selector.variances_)[::-1][:n_features]
    return X.iloc[:, sorted(top_idx)]


def _subsample(X: pd.DataFrame, y: pd.Series, n_samples: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Random subsample without replacement, reproducible via the benchmark seed."""
    if n_samples >= len(X):
        return X, y
    idx = X.sample(n=n_samples, random_state=seed).index
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def load_california_housing(n_samples: int | None = None, n_features: int | None = None,
                            *, seed: int) -> dict:
    """California Housing – 8 features, 20 640 samples, regression."""
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.frame[bunch.feature_names]
    y = bunch.frame[bunch.target_names[0]]
    if n_features is not None:
        X = _select_features_by_variance(X, n_features)
    if n_samples is not None:
        X, y = _subsample(X, y, n_samples, seed)
    return {
        "name": "California Housing",
        "task": "regression",
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": bunch.target_names[0],
    }


def load_ames_housing(n_samples: int | None = None, n_features: int | None = None,
                      *, seed: int) -> dict:
    """Ames Housing – ~79 features, 1 460 samples, regression.

    Categorical columns are label-encoded so tree and linear models can consume
    the data without extra preprocessing.  Missing values are imputed with the
    column median (numeric) or mode (categorical).
    """
    bunch = fetch_openml(data_id=42165, as_frame=True, parser="auto")
    df: pd.DataFrame = bunch.frame.copy()

    # Drop the id column if present
    df = df.drop(columns=["Id"], errors="ignore")

    target_col = bunch.target_names[0]
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].astype(float)

    # Impute and encode
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mode().iloc[0])
            X[col] = X[col].astype("category").cat.codes
        else:
            X[col] = X[col].fillna(X[col].median())

    if n_features is not None:
        X = _select_features_by_variance(X, n_features)
    if n_samples is not None:
        X, y = _subsample(X, y, n_samples, seed)
    return {
        "name": "Ames Housing",
        "task": "regression",
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": target_col,
    }


def load_covertype(n_samples: int | None = None, n_features: int | None = None,
                   *, seed: int) -> dict:
    """Forest Covertype – 54 features, 581 012 samples, classification (7 classes).

    For faster experimentation a stratified subset of 50 000 samples is returned
    by default.  Pass subset=None to get the full dataset.
    """
    bunch = fetch_covtype(as_frame=True)
    X: pd.DataFrame = bunch.frame[bunch.feature_names]
    y: pd.Series = bunch.frame["Cover_Type"].astype(int)

    if n_samples is not None:
        X, y = _subsample(X, y, n_samples, seed)
    else:
        # Default: stratified 50 k subsample so notebooks stay responsive
        sample_idx = (
            y.groupby(y)
            .apply(lambda g: g.sample(frac=50_000 / len(y), random_state=seed))
            .index.get_level_values(1)
        )
        X = X.loc[sample_idx].reset_index(drop=True)
        y = y.loc[sample_idx].reset_index(drop=True)

    if n_features is not None:
        X = _select_features_by_variance(X, n_features)

    return {
        "name": "Forest Covertype",
        "task": "classification",
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": "Cover_Type",
    }

def load_adult_census(n_samples: int | None = None, n_features: int | None = None,
                      *, seed: int) -> dict:
    bunch = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    df: pd.DataFrame = bunch.frame.copy()

    # Drop the id column if present
    df = df.drop(columns=["Id"], errors="ignore")

    target_col = bunch.target_names[0]
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    # Target is the binary income bracket ("<=50K" / ">50K", sometimes with a trailing
    # period from the openml ARFF) — a string/categorical that cannot be cast to float
    # directly. Map the high-income class to 1 so downstream predict_proba[:, 1] reads
    # as P(>50K).
    y = (
        df[target_col].astype(str).str.strip().str.replace(".", "", regex=False) == ">50K"
    ).astype(int)

    # Impute and encode
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mode().iloc[0])
            X[col] = X[col].astype("category").cat.codes
        else:
            X[col] = X[col].fillna(X[col].median())

    if n_features is not None:
        X = _select_features_by_variance(X, n_features)
    if n_samples is not None:
        X, y = _subsample(X, y, n_samples, seed)
    return {
        "name": "Adult Census",
        "task": "classification",
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": target_col,
    }

def load_gisette(n_samples: int | None = None, n_features: int | None = None,
                 *, seed: int) -> dict:
    """Gisette – 5000 features, 7000 samples, binary classification (digits 4 vs 9).

    High-dimensional dataset from the NIPS 2003 feature selection challenge.
    Labels are -1 and 1. Good test case for feature selection and regularization.
    OpenML id: 41026 (train + validation split, 7k of 13.5k total rows).
    """
    # Gisette is stored as a sparse ARFF on openml, which cannot be returned as a
    # DataFrame (as_frame=True raises "Sparse ARFF datasets cannot be loaded with
    # as_frame=True"). Fetch raw arrays instead and densify — the data is only nominally
    # sparse (pixel-derived features) and the downstream pipeline (variance selection,
    # sklearn models, shap) expects a dense DataFrame.
    import scipy.sparse as sp

    bunch = fetch_openml(data_id=41026, as_frame=False, parser="auto")
    data = bunch.data
    data = data.toarray() if sp.issparse(data) else np.asarray(data)
    feature_names = (
        list(bunch.feature_names)
        if getattr(bunch, "feature_names", None) is not None
        else [f"f{i}" for i in range(data.shape[1])]
    )
    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(np.asarray(bunch.target)).astype(int)

    if n_features is not None:
        X = _select_features_by_variance(X, n_features)
    if n_samples is not None:
        X, y = _subsample(X, y, n_samples, seed)
    return {
        "name": "Gisette",
        "task": "classification",
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": bunch.target_names[0],
    }



def load_all(*, seed: int) -> dict[str, dict]:
    """Load all three datasets keyed by short name."""
    return {
        "california": load_california_housing(seed=seed),
        "ames": load_ames_housing(seed=seed),
        "covertype": load_covertype(seed=seed),
        "adult_census": load_adult_census(seed=seed),
    }


if __name__ == "__main__":
    datasets = load_all(seed=42)
    for name, data in datasets.items():
        print(f"{name}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")