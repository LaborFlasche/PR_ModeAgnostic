from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd


class BaseBackend(ABC):
    name: str
    library: str
    computation_type: Literal["true_value", "approximation"]
    chosen_method: str | None = None

    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.background = background

    @abstractmethod
    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return contributions as a DataFrame of shape (n_samples, n_features)."""
