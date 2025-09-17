import numpy as np
from ddevd.mev import MEV


def test_mev_cdf_properties(small_block_data):
    mev = MEV(small_block_data)
    x = np.linspace(0, np.max([d.max() for d in small_block_data]) * 1.1, 40)
    F = [mev.cdf(val) for val in x]
    assert all(0 <= v <= 1 for v in F)
    # monotonic
    assert all(F[i] <= F[i+1] for i in range(len(F)-1))
