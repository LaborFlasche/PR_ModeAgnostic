import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def mean_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float((a - b).abs().mean().mean())


def relative_mae(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Scale-free accuracy: MAE normalized by the mean magnitude of both values.

    ``mean_abs_diff`` is in the units of the model output (house prices vs.
    probabilities), so it cannot be averaged across datasets. Dividing by
    the average magnitude of both DataFrames yields a dimensionless error
    (0 = perfect) that is comparable across datasets and models, and is symmetric.
    Returns NaN when both DataFrames are zero.
    """
    denom = float((a.abs().values + b.abs().values).mean()) / 2.0
    if denom == 0:
        return float("nan")
    return mean_abs_diff(a, b) / denom


def sign_agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    # NaN != 0 is True, so all-NaN frames (timed-out/skipped backends) would pass
    # the mask and read as 0.0 (total disagreement) instead of NaN (missing).
    mask = (a != 0) & (b != 0) & a.notna() & b.notna()
    total = mask.sum().sum()
    if total == 0:
        return float("nan")
    return float((np.sign(a[mask]) == np.sign(b[mask])).sum().sum() / total)


def mean_sample_rho(a: pd.DataFrame, b: pd.DataFrame) -> float:
    rhos = [
        spearmanr(a.iloc[i].values, b.iloc[i].values).statistic
        for i in range(len(a))
    ]
    return float(np.mean(rhos))
