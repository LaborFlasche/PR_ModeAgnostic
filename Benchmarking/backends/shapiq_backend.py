import pandas as pd
import shapiq

from .base_backend import BaseBackend


class ShapIQModeAgnosticBackend(BaseBackend):
    name = "shapiq_mode_agnostic"
    library = "shapiq"
    computation_type = "approximation"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        background_np = self.background.values.astype(float)
        x_np = x.values.astype(float)

        explainer = shapiq.Explainer(
            model=self.model,
            data=background_np,
            index="SV",
            max_order=1,
        )
        self.chosen_method = type(explainer).__name__

        rows = [explainer.explain(x_np[i]).get_n_order_values(1) for i in range(len(x_np))]
        return pd.DataFrame(rows, index=x.index, columns=x.columns)
