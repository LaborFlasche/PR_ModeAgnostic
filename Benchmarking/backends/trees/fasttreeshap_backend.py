import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from ..base_backend import BaseBackend, nan_result, nan_interaction_result

_RUNNER_SCRIPT = Path(__file__).parent / "_fasttreeshap_runner.py"
_DEFAULT_VENV_PYTHON = str(Path.home() / ".cache" / "pr-modeagnostic" / ".venv-fasttreeshap" / "bin" / "python")


class _FastTreeShapBackend(BaseBackend):
    """Shared subprocess plumbing for fasttreeshap's out-of-process TreeExplainer
    calls. Requires numpy<2 (this project requires numpy>=2), so it runs out-of-
    process via subprocess in a dedicated venv (see scripts/setup_fasttreeshap_env.sh),
    never imported directly. XGBoost is skipped (see below); any other failure
    (missing venv, model-load error) logs a skip and returns an all-NaN frame."""

    library = "fasttreeshap"
    computation_type = "true_value"
    interactions: bool = False

    def _nan(self, x: pd.DataFrame) -> pd.DataFrame:
        return nan_interaction_result(x) if self.interactions else nan_result(x)

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        if type(self.model).__module__.startswith("xgboost"):
            # fasttreeshap 0.1.6 (unmaintained since 2022) cannot parse XGBoost
            # 3.x's model format at all (UnicodeDecodeError) — not fixable by
            # version-pinning. LightGBM/sklearn models are unaffected.
            print(f"  [SKIP] {self.name}: fasttreeshap 0.1.6 cannot parse XGBoost "
                  "3.x's model format (confirmed upstream incompatibility)")
            return self._nan(x)

        venv_python = os.environ.get("FASTTREESHAP_VENV_PYTHON", _DEFAULT_VENV_PYTHON)
        if not Path(venv_python).exists():
            print(f"  [SKIP] {self.name}: no fasttreeshap venv found at {venv_python} "
                  "(see scripts/setup_fasttreeshap_env.sh)")
            return self._nan(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            x_path = Path(tmpdir) / "x.csv"
            output_path = Path(tmpdir) / "out.csv"

            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            x.to_csv(x_path)

            cmd = [venv_python, str(_RUNNER_SCRIPT),
                   "--model", str(model_path), "--x", str(x_path), "--output", str(output_path)]
            if self.interactions:
                cmd.append("--interactions")

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [BUG] {self.name} subprocess failed: {result.stderr.strip()[-500:]}", file=sys.stderr)
                return self._nan(x)

            # Sidecar written by the runner script: base value of the
            # path-dependent game fasttreeshap explains (see _fasttreeshap_runner.py).
            baseline_path = Path(str(output_path) + ".baseline")
            if baseline_path.exists():
                self.baseline_ = float(baseline_path.read_text())

            return pd.read_csv(output_path, index_col=0)


class FastTreeShapBackend(_FastTreeShapBackend):
    """fasttreeshap's TreeExplainer, path-dependent only (first-order Shapley values)."""

    name = "fasttreeshap_path_dependent"
    interactions = False


class FastTreeShapInteractionBackend(_FastTreeShapBackend):
    """fasttreeshap's TreeExplainer, pairwise (order-2) interactions. Same
    xgboost/missing-venv skip conditions as FastTreeShapBackend. Requests
    algorithm="v1" (via --interactions in the runner script): fasttreeshap's faster
    "v2" algorithm doesn't support shap_interaction_values."""

    name = "fasttreeshap_interaction"
    order = 2
    interactions = True
