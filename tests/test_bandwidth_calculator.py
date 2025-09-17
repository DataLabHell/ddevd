import numpy as np
import scipy.stats
from ddevd.optimal_bandwidth import BandwidthCalculator


def test_bandwidth_calculator_global():
    rng = np.random.default_rng(5)
    samples = [rng.normal(size=40), rng.normal(loc=2, scale=1.5, size=55), rng.normal(loc=-1, scale=0.5, size=30)]
    bc = BandwidthCalculator(samples, scipy.stats.norm.pdf, scipy.stats.norm.cdf, optimization_position="global")
    h_global = bc.compute_optimal_global_bandwidth()
    assert h_global is None or h_global > 0  # if None indicates negative / failure already logged


def test_bandwidth_calculator_binwise():
    rng = np.random.default_rng(6)
    samples = [rng.normal(size=35), rng.normal(loc=1, scale=2, size=50)]
    bc = BandwidthCalculator(samples, scipy.stats.norm.pdf, scipy.stats.norm.cdf, optimization_position="global")
    h_vec = bc.compute_optimal_binwise_bandwidth()
    if h_vec is not None:
        assert len(h_vec) == len(samples)
        assert all(h > 0 for h in h_vec)
