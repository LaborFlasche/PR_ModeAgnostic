"""
BaseBackend — mirrors Shapash's shapash/backend/base_backend.py interface.

The only required method is `run_explainer`.  Everything else has working defaults
so a new backend only needs to override what differs from the standard behaviour.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseBackend(ABC):
    """Abstract base for all attribution backends.

    To add a new backend subclass this class and implement `run_explainer`.
    The returned dict must contain a 'contributions' key whose value is a
    pd.DataFrame of shape (n_samples, n_features) with the same index and
    columns as the input `x`.

    For multi-class models return a list of such DataFrames, one per class.
    """

    # String identifier used to look up backends by name.
    name: str = "base"

    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def run_explainer(self, x: pd.DataFrame) -> dict:
        """Compute local feature attributions.

        Parameters
        ----------
        x : pd.DataFrame
            Input samples, shape (n_samples, n_features).

        Returns
        -------
        dict
            Must contain:
            - 'contributions': pd.DataFrame (n_samples, n_features)  *or*
              list[pd.DataFrame] for multi-class.
            May optionally contain any extra keys (e.g. 'runtime_s').
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience helpers (same role as Shapash's BaseBackend helpers)
    # ------------------------------------------------------------------

    def get_contributions(self, x: pd.DataFrame) -> pd.DataFrame | list[pd.DataFrame]:
        """Run the explainer and return a tidy contributions DataFrame."""
        explain_data = self.run_explainer(x)
        contributions = explain_data["contributions"]
        return self._to_dataframe(contributions, x)

    def get_global_importance(
        self,
        contributions: pd.DataFrame | list[pd.DataFrame],
    ) -> pd.Series | list[pd.Series]:
        """Global importance = mean(|contribution|) across samples.

        Matches Shapash's default `get_global_features_importance` logic.
        """
        if isinstance(contributions, list):
            return [c.abs().mean(axis=0).sort_values(ascending=False) for c in contributions]
        return contributions.abs().mean(axis=0).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_dataframe(
        self,
        contributions: np.ndarray | pd.DataFrame | list,
        x: pd.DataFrame,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Coerce raw array or list of arrays into a tidy DataFrame."""
        if isinstance(contributions, list):
            return [self._to_dataframe(c, x) for c in contributions]
        if isinstance(contributions, np.ndarray):
            return pd.DataFrame(contributions, index=x.index, columns=x.columns)
        return contributions
