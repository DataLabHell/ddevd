import numpy as np
import pytest
from ddevd.ddevd import DDEVD


def test_empty_data_rejected():
    with pytest.raises(ValueError):
        DDEVD([])

def test_invalid_h_opt_position():
    rng = np.random.default_rng(1)
    data = [rng.normal(size=20) for _ in range(5)]
    with pytest.raises(ValueError):
        DDEVD(data, h_opt_position="not_valid")
