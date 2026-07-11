import json

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from Benchmarking.runner import BenchmarkRunner
from Benchmarking.backends.true_value.shap_backend import ShapTrueValueBackend
from Benchmarking.backends.true_value.shapiq_backend import ShapIQTrueValueBackend


@pytest.fixture
def toy_rf_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((120, 4)), columns=["f0", "f1", "f2", "f3"])
    y = X["f0"] + 2 * X["f1"] + rng.normal(0, 0.1, 120)
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X


@pytest.fixture
def runner(tmp_path, toy_rf_data):
    model, X = toy_rf_data
    return BenchmarkRunner(
        true_value_backends=[ShapTrueValueBackend, ShapIQTrueValueBackend],
        approximation_specs=[],
        output_csv=str(tmp_path / "results.csv"),
        n_background=100,
        n_eval=None,
        seed=0,
    ), model, X


def test_csv_created_after_run(runner):
    bench, model, X = runner
    bench.run(model, X, run_meta={"dataset": "test_ds", "model": "rf", "n_features": 4, "n_samples": 120})
    assert Path(bench.output_csv).exists()


def test_csv_has_correct_columns(runner):
    bench, model, X = runner
    bench.run(model, X, run_meta={"dataset": "test_ds", "model": "rf", "n_features": 4, "n_samples": 120})
    df = pd.read_csv(bench.output_csv)
    expected = {"dataset", "model", "n_features", "n_samples", "backend", "library",
                "computation_type", "n_eval", "runtime_s", "n_model_evals",
                "additivity_gap", "relative_additivity_gap", "shapley_values",
                "shapley_n_eval", "shapley_n_features", "pairwise_metrics"}
    assert expected.issubset(set(df.columns))


def test_csv_has_one_row_per_backend(runner):
    bench, model, X = runner
    bench.run(model, X, run_meta={"dataset": "test_ds", "model": "rf", "n_features": 4, "n_samples": 120})
    df = pd.read_csv(bench.output_csv)
    assert len(df) == 2  # ShapTrueValueBackend + ShapIQTrueValueBackend


def test_csv_appends_on_second_run(runner):
    bench, model, X = runner
    meta = {"dataset": "test_ds", "model": "rf", "n_features": 4, "n_samples": 120}
    bench.run(model, X, run_meta=meta)
    bench.run(model, X, run_meta=meta)
    df = pd.read_csv(bench.output_csv)
    assert len(df) == 4  # 2 backends × 2 runs


def test_pairwise_self_comparison_is_exact(runner):
    bench, model, X = runner
    bench.run(model, X, run_meta={"dataset": "test_ds", "model": "rf", "n_features": 4, "n_samples": 120})
    df = pd.read_csv(bench.output_csv)
    shap_row = df[df["backend"] == "shap_true_value"].iloc[0]
    self_metrics = json.loads(shap_row["pairwise_metrics"])["shap_true_value"]
    assert self_metrics["mean_abs_diff"] == 0.0
    assert self_metrics["relative_mae"] == 0.0
    assert self_metrics["mean_sample_rho"] == 1.0


def test_pairwise_metrics_cover_every_backend(runner):
    bench, model, X = runner
    bench.run(model, X, run_meta={"dataset": "test_ds", "model": "rf", "n_features": 4, "n_samples": 120})
    df = pd.read_csv(bench.output_csv)
    shapiq_row = df[df["backend"] == "shapiq_true_value"].iloc[0]
    pairwise = json.loads(shapiq_row["pairwise_metrics"])
    assert set(pairwise) == {"shap_true_value", "shapiq_true_value"}
    vs_shap = pairwise["shap_true_value"]
    assert np.isfinite(vs_shap["mean_abs_diff"])
    assert np.isfinite(vs_shap["mean_sample_rho"])


def test_n_eval_limits_explained_samples(tmp_path, toy_rf_data):
    model, X = toy_rf_data
    bench = BenchmarkRunner(
        true_value_backends=[ShapTrueValueBackend],
        approximation_specs=[],
        output_csv=str(tmp_path / "results.csv"),
        n_background=100,
        n_eval=10,
    )
    bench.run(model, X, run_meta={"dataset": "ds", "model": "rf", "n_features": 4, "n_samples": 120})
    df = pd.read_csv(bench.output_csv)
    assert df.iloc[0]["n_eval"] == 10


def test_raises_when_x_too_small(tmp_path, toy_rf_data):
    model, X = toy_rf_data
    bench = BenchmarkRunner(
        true_value_backends=[ShapTrueValueBackend],
        approximation_specs=[],
        output_csv=str(tmp_path / "results.csv"),
        n_background=100,
        n_eval=None,
    )
    X_small = X.iloc[:50]  # only 50 rows, less than n_background=100
    with pytest.raises(ValueError, match="no evaluation rows remain"):
        bench.run(model, X_small, run_meta={"dataset": "ds", "model": "rf", "n_features": 4, "n_samples": 50})
