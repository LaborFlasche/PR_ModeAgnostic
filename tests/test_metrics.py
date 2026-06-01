import numpy as np
import pandas as pd
import pytest
from Benchmarking.metrics import mean_abs_diff, sign_agreement, mean_sample_rho


def _df(arr):
    return pd.DataFrame(arr, columns=["f0", "f1", "f2"])


def test_mean_abs_diff_identical():
    a = _df(np.ones((4, 3)))
    assert mean_abs_diff(a, a) == pytest.approx(0.0)


def test_mean_abs_diff_known():
    a = _df(np.zeros((4, 3)))
    b = _df(np.ones((4, 3)))
    assert mean_abs_diff(a, b) == pytest.approx(1.0)


def test_sign_agreement_identical():
    a = _df(np.array([[1, -1, 2], [-1, 2, -3]], dtype=float))
    assert sign_agreement(a, a) == pytest.approx(1.0)


def test_sign_agreement_opposite():
    a = _df(np.array([[1.0, 1.0, 1.0]]))
    b = _df(np.array([[-1.0, -1.0, -1.0]]))
    assert sign_agreement(a, b) == pytest.approx(0.0)


def test_sign_agreement_excludes_zeros():
    a = _df(np.array([[0.0, 1.0, -1.0]]))
    b = _df(np.array([[0.0, 1.0, 1.0]]))
    # only the two non-zero pairs in a are counted; b's zero slot in a is excluded
    assert sign_agreement(a, b) == pytest.approx(0.5)


def test_mean_sample_rho_identical():
    rng = np.random.default_rng(0)
    a = _df(rng.random((5, 3)))
    assert mean_sample_rho(a, a) == pytest.approx(1.0)


def test_mean_sample_rho_reversed():
    a = _df(np.array([[1.0, 2.0, 3.0]]))
    b = _df(np.array([[3.0, 2.0, 1.0]]))
    assert mean_sample_rho(a, b) == pytest.approx(-1.0)
