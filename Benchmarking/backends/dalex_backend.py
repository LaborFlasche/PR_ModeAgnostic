import dalex as dx
import pandas as pd

from .base_backend import BaseBackend


class DalexBackend(BaseBackend):
    name = "dalex_shap"
    library = "dalex"
    computation_type = "approximation"
    B: int = 25

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        y = self.model.predict(self.background)
        exp = dx.Explainer(self.model, self.background, y, verbose=False)
        self.chosen_method = f"shap_random_path_B{self.B}"

        columns = list(x.columns)
        rows = []
        for i in range(len(x)):
            pp = exp.predict_parts(new_observation=x.iloc[[i]], type="shap", B=self.B)
            avg = pp.result[pp.result["B"] == 0].set_index("variable_name")
            rows.append([float(avg.loc[col, "contribution"]) if col in avg.index else float("nan") for col in columns])
        return pd.DataFrame(rows, index=x.index, columns=columns)
