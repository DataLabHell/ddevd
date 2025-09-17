import numpy as np
import pytest
from ddevd.ddevd import DDEVD


def _make_data(seed=0, m=6):
    rng = np.random.default_rng(seed)
    return [rng.normal(loc=i*0.2, scale=1.0 + 0.1*i, size=rng.integers(25, 45)) for i in range(m)]


def test_quantile_optimization_position():
    data = _make_data()
    dist = DDEVD(data, h_opt_position="quantile_0.9")
    # Ensure cdf callable works
    x = np.linspace(-2, 5, 30)
    F = dist.cdf(x)
    assert np.all(np.diff(F) >= -1e-6)


def test_numeric_optimization_position():
    data = _make_data(seed=1)
    dist = DDEVD(data, h_opt_position=0.0)
    x = np.linspace(-1, 4, 20)
    F = dist.cdf(x)
    assert F[0] >= 0 and F[-1] <= 1


def test_invalid_quantile_format():
    data = _make_data(seed=2)
    with pytest.raises(ValueError):
        DDEVD(data, h_opt_position="quantile_notnumber")
