import numpy as np
from ddevd.distributions import WeibullDistributionML


def test_weibull_fit_converges():
    rng = np.random.default_rng(0)
    shape_true = 2.5
    scale_true = 3.0
    x = (rng.weibull(shape_true, size=2000)) * scale_true
    k_hat, lam_hat = WeibullDistributionML.fit(x)
    # allow modest tolerance due to custom iterative method
    assert np.isfinite(k_hat) and np.isfinite(lam_hat)
    assert 1.5 < k_hat < 3.5
    assert 2.0 < lam_hat < 4.5


def test_weibull_fit_handles_small_sample():
    x = np.array([1.0, 2.0, 3.0])
    k_hat, lam_hat = WeibullDistributionML.fit(x, iters=50)
    assert np.isfinite(k_hat)
    assert np.isfinite(lam_hat)
