import numpy as np
import pytest
from ddevd.ddevd import DDEVD


def test_empty_data_rejected():
    with pytest.raises(ValueError):
        DDEVD([])


def test_mean_measurement_too_large():
    # create data with one very large measurement length to push mean > 500
    big = list(np.random.default_rng(0).normal(size=600))
    small = [1.0]
    data = [big, small]
    with pytest.raises(ValueError):
        DDEVD(data)


def test_invalid_h_opt_position():
    rng = np.random.default_rng(1)
    data = [rng.normal(size=20) for _ in range(5)]
    with pytest.raises(ValueError):
        DDEVD(data, h_opt_position="not_valid")
