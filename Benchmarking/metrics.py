import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def mean_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float((a - b).abs().mean().mean())


def sign_agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    mask = (a != 0) & (b != 0)
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
