import numpy as np
from ddevd.helpers import ddevd_weibull_kernel


def test_weibull_kernel_basic(small_block_data):
    pdf, cdf = ddevd_weibull_kernel(small_block_data)
    xs = np.linspace(-2, 15, 200)
    cdf_vals = cdf(xs)
    # monotonic & bounded
    assert np.all(np.diff(cdf_vals) >= -1e-6)
    assert 0 <= cdf_vals[0] <= 1
    assert 0.7 < cdf_vals[-1] <= 1
    pdf_vals = pdf(xs)
    assert pdf_vals.shape == xs.shape
    # non-negativity (allow tiny negative numerical noise)
    assert np.min(pdf_vals) > -1e-6
