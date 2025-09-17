import numpy as np
from scipy.stats import norm
from ddevd.evd import DistributionEVD


def test_distribution_evd_matches_formula():
    sample_sizes = [3, 7, 12]
    evd = DistributionEVD(norm, sample_sizes)
    xs = np.linspace(-2, 2, 25)
    manual = np.mean([norm.cdf(xs) ** n for n in sample_sizes], axis=0)
    impl = np.array([evd.cdf(x) for x in xs])
    assert np.allclose(manual, impl, rtol=1e-10, atol=1e-12)
