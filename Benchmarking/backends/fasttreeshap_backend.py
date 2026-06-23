import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from .base_backend import BaseBackend, nan_result

_RUNNER_SCRIPT = Path(__file__).parent / "_fasttreeshap_runner.py"
_DEFAULT_VENV_PYTHON = str(Path.home() / ".cache" / "pr-modeagnostic" / ".venv-fasttreeshap" / "bin" / "python")


class FastTreeShapBackend(BaseBackend):
    """fasttreeshap's TreeExplainer, path-dependent only. Requires numpy<2 (this
    project requires numpy>=2), so it runs out-of-process via subprocess in a
    dedicated venv (see scripts/setup_fasttreeshap_env.sh), never imported
    directly. XGBoost is skipped (see below); any other failure (missing venv,
    model-load error) logs a skip and returns an all-NaN frame."""

    name = "fasttreeshap_path_dependent"
    library = "fasttreeshap"
    computation_type = "true_value"

    def run_explainer(self, x: pd.DataFrame) -> pd.DataFrame:
        if type(self.model).__module__.startswith("xgboost"):
            # fasttreeshap 0.1.6 (unmaintained since 2022) cannot parse XGBoost
            # 3.x's model format at all (UnicodeDecodeError) — not fixable by
            # version-pinning. LightGBM/sklearn models are unaffected.
            print(f"  [SKIP] {self.name}: fasttreeshap 0.1.6 cannot parse XGBoost "
                  "3.x's model format (confirmed upstream incompatibility)")
            return nan_result(x)

        venv_python = os.environ.get("FASTTREESHAP_VENV_PYTHON", _DEFAULT_VENV_PYTHON)
        if not Path(venv_python).exists():
            print(f"  [SKIP] {self.name}: no fasttreeshap venv found at {venv_python} "
                  "(see scripts/setup_fasttreeshap_env.sh)")
            return nan_result(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            x_path = Path(tmpdir) / "x.csv"
            output_path = Path(tmpdir) / "out.csv"

            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            x.to_csv(x_path)

            result = subprocess.run(
                [venv_python, str(_RUNNER_SCRIPT),
                 "--model", str(model_path), "--x", str(x_path), "--output", str(output_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  [BUG] {self.name} subprocess failed: {result.stderr.strip()[-500:]}", file=sys.stderr)
                return nan_result(x)

            return pd.read_csv(output_path, index_col=0)
