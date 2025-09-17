import numpy as np
from ddevd.ddevd import DDEVD


def test_quantile_inverse_property(small_block_data):
    dist = DDEVD(small_block_data, h_opt_position="global")
    qs = [0.1, 0.5, 0.9]
    for q in qs:
        x = dist.quantile(q)
        F = dist.cdf(np.array([x]))[0]
        assert abs(F - q) < 0.05  # loose tolerance due to smoothing / finite sample


def test_return_levels_equivalence(small_block_data):
    dist = DDEVD(small_block_data)
    periods = [2, 5, 10]
    rls = dist.return_levels(periods)
    manual = [dist.quantile(1 - 1/p) for p in periods]
    for a, b in zip(rls, manual):
        assert abs(a - b) / (abs(b) + 1e-9) < 0.05
