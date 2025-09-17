import numpy as np
import pytest

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(1234)

@pytest.fixture
def small_block_data(rng):
    # produce 8 blocks with varying sizes 15-40
    return [rng.weibull(a=2.0, size=rng.integers(15, 40)) * 5 for _ in range(8)]

@pytest.fixture
def medium_block_data(rng):
    return [rng.normal(loc=0, scale=3, size=rng.integers(30, 60)) for _ in range(15)]
