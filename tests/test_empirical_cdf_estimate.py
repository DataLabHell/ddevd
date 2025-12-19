import numpy as np

from ddevd.distributions import EmpiricalCDFEstimate


def test_empirical_bandwidth_positive():
    rng = np.random.default_rng(42)
    data = [rng.normal(size=50).tolist(), rng.normal(loc=2, scale=1.5, size=60).tolist()]

    estimator = EmpiricalCDFEstimate(data)

    h_vec = estimator.bandwidths
    assert len(h_vec) == len(data)
    assert np.all(h_vec > 0)


def test_empirical_cdf_monotone_and_bounded():
    rng = np.random.default_rng(123)
    data = [rng.normal(loc=0.5, scale=1.2, size=80).tolist()]

    estimator = EmpiricalCDFEstimate(data)

    xs = np.linspace(-5, 5, 25)
    cdf_vals = estimator.cdf(xs)

    assert np.all(np.diff(cdf_vals) >= -1e-6)
    assert np.all(cdf_vals >= -1e-6)
    assert np.all(cdf_vals <= 1 + 1e-6)
