import numpy as np
import pytest

from ddevd.ddevd import DDEVD


def test_ddevd_cdf_monotonic(small_block_data):
    dist = DDEVD(small_block_data, h_opt_position="global")
    x = np.linspace(0, np.max([d.max() for d in small_block_data]) * 1.2, 50)
    F = dist.cdf(x)
    # monotonic non-decreasing
    assert np.all(np.diff(F) >= -1e-6)
    # bounds
    assert F[0] >= 0 - 1e-9
    assert F[-1] <= 1 + 1e-9


def test_ddevd_return_levels_shape(small_block_data):
    dist = DDEVD(small_block_data, h_opt_position="global")
    periods = [2, 5, 10]
    rls = dist.return_levels(periods)
    assert len(rls) == len(periods)
    # return levels should be increasing with period
    assert all(rls[i] <= rls[i+1] for i in range(len(rls)-1))


def test_ddevd_bootstrap_shape(small_block_data):
    dist = DDEVD(small_block_data, h_opt_position="global")
    samples = dist.bootstrap_return_levels([5, 10], n_resample=20)
    assert samples.shape == (20, 2)
