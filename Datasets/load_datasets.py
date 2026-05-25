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

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml


def load_california_housing() -> dict:
    """California Housing – 8 features, 20 640 samples, regression."""
    bunch = fetch_california_housing(as_frame=True)
    return {
        "name": "California Housing",
        "task": "regression",
        "X": bunch.frame[bunch.feature_names],
        "y": bunch.frame[bunch.target_names[0]],
        "feature_names": bunch.feature_names,
        "target_name": bunch.target_names[0],
    }


def load_ames_housing() -> dict:
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

    return {
        "name": "Ames Housing",
        "task": "regression",
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "target_name": target_col,
    }


def load_covertype() -> dict:
    """Forest Covertype – 54 features, 581 012 samples, classification (7 classes).

    For faster experimentation a stratified subset of 50 000 samples is returned
    by default.  Pass subset=None to get the full dataset.
    """
    bunch = fetch_covtype(as_frame=True)
    X: pd.DataFrame = bunch.frame[bunch.feature_names]
    y: pd.Series = bunch.frame["Cover_Type"].astype(int)

    # Stratified subsample so notebooks stay responsive
    sample_idx = (
        y.groupby(y)
        .apply(lambda g: g.sample(frac=50_000 / len(y), random_state=42))
        .index.get_level_values(1)
    )
    X = X.loc[sample_idx].reset_index(drop=True)
    y = y.loc[sample_idx].reset_index(drop=True)

    return {
        "name": "Forest Covertype",
        "task": "classification",
        "X": X,
        "y": y,
        "feature_names": bunch.feature_names,
        "target_name": "Cover_Type",
    }


def load_all() -> dict[str, dict]:
    """Load all three datasets keyed by short name."""
    return {
        "california": load_california_housing(),
        "ames": load_ames_housing(),
        "covertype": load_covertype(),
    }


if __name__ == "__main__":
    datasets = load_all()
    for name, data in datasets.items():
        print(f"{name}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")