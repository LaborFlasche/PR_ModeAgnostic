"""
Comparison metrics for attribution benchmarking.

All functions accept pd.DataFrames of shape (n_samples, n_features) and
return scalar values or pd.Series/DataFrames for pairwise comparison.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Pairwise sample-level metrics
# ---------------------------------------------------------------------------

def sign_agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Fraction of (sample, feature) pairs where both attributions have the same sign.

    Zero-valued attributions are excluded from both sides before comparison.
    """
    mask = (a != 0) & (b != 0)
    if mask.sum().sum() == 0:
        return float("nan")
    return float((np.sign(a[mask]) == np.sign(b[mask])).sum().sum() / mask.sum().sum())


def mean_absolute_difference(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean |a_ij - b_ij| across all samples and features."""
    return float((a - b).abs().mean().mean())


def rank_correlation_per_sample(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Per-sample Spearman rank correlation of feature attributions.

    Returns a Series of length n_samples.
    """
    correlations = []
    for i in range(len(a)):
        rho, _ = spearmanr(a.iloc[i].values, b.iloc[i].values)
        correlations.append(rho)
    return pd.Series(correlations, index=a.index, name="spearman_rho")


# ---------------------------------------------------------------------------
# Global importance metrics
# ---------------------------------------------------------------------------

def global_importance_rank_correlation(imp_a: pd.Series, imp_b: pd.Series) -> float:
    """Spearman rank correlation between two global feature importance vectors.

    Higher = the two backends agree on which features matter most overall.
    """
    shared = imp_a.index.intersection(imp_b.index)
    rho, _ = spearmanr(imp_a[shared].values, imp_b[shared].values)
    return float(rho)


def top_k_overlap(imp_a: pd.Series, imp_b: pd.Series, k: int = 5) -> float:
    """Fraction of top-k features shared between two global importance rankings."""
    top_a = set(imp_a.nlargest(k).index)
    top_b = set(imp_b.nlargest(k).index)
    return len(top_a & top_b) / k


# ---------------------------------------------------------------------------
# Pairwise summary table
# ---------------------------------------------------------------------------

def pairwise_summary(
    contributions: dict[str, pd.DataFrame],
    importances: dict[str, pd.Series],
    top_k: int = 5,
) -> pd.DataFrame:
    """Build a pairwise comparison table for all backend pairs.

    Parameters
    ----------
    contributions : dict[backend_name -> pd.DataFrame(n_samples, n_features)]
    importances   : dict[backend_name -> pd.Series(n_features)]
    top_k         : number of top features for overlap metric

    Returns
    -------
    pd.DataFrame with MultiIndex columns (metric, backend_b) and backend_a as index.
    """
    names = list(contributions.keys())
    records = []

    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i >= j:
                continue
            a_cont = contributions[na]
            b_cont = contributions[nb]
            a_imp = importances[na]
            b_imp = importances[nb]

            records.append(
                {
                    "backend_a": na,
                    "backend_b": nb,
                    "sign_agreement": sign_agreement(a_cont, b_cont),
                    "mean_abs_diff": mean_absolute_difference(a_cont, b_cont),
                    "mean_sample_rho": rank_correlation_per_sample(a_cont, b_cont).mean(),
                    "global_rho": global_importance_rank_correlation(a_imp, b_imp),
                    f"top{top_k}_overlap": top_k_overlap(a_imp, b_imp, k=top_k),
                }
            )

    return pd.DataFrame(records).set_index(["backend_a", "backend_b"])
