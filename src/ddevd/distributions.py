"""Various statistical distributions used in DDEVD.

This file contains implementations of different statistical distributions that can be used in the DDEVD.
"""

import numpy as np
import scipy.stats
from scipy.stats import weibull_min
import logging

from ddevd.optimal_bandwidth import BandwidthCalculator

logger = logging.getLogger(__name__)
# add the loggername to the log messages
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

class WeibullDistributionML:
    def __init__(self):
        pass

    @staticmethod
    def fit(x, iters=500, eps=1e-6):
        """Fits a 2-parameter Weibull distribution to the given data using maximum-likelihood estimation.

        :param x: 1d-ndarray of samples from an (unknown) distribution. Each value must satisfy x > 0.
        :param iters: Maximum number of iterations
        :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
        :return: Tuple (Shape, Scale) which can be (NaN, NaN) if a fit is impossible.
            Impossible fits may be due to 0-values in x.
        """
        x = np.asarray(x, dtype=float)
        # if any value is less than or equal to zero, return NaN
        if np.any(x <= 0):
            # if less than 5% of the values are non-positive, log a warning and drop them
            if np.sum(x <= 0) / len(x) > 0.05:
                logger.warning(f"Weibull fit failed due to non-positive values in input data ({np.sum(x <= 0)} / {len(x)}). Returning NaN values.")
                return np.nan, np.nan
            x = x[x > 0]
        # fit k via MLE
        ln_x = np.log(x)
        k = 1.
        k_t_1 = k

        logger.debug(f"Starting Weibull fit with initial guess k={k}. Performing up to {iters} iterations.")

        for i in range(iters):
            x_k = x ** k
            x_k_ln_x = x_k * ln_x
            ff = np.sum(x_k_ln_x)
            fg = np.sum(x_k)
            f = ff / fg - np.mean(ln_x) - (1. / k)

            # Calculate second derivative d^2f/dk^2
            ff_prime = np.sum(x_k_ln_x * ln_x)
            fg_prime = ff
            f_prime = (ff_prime/fg - (ff/fg * fg_prime/fg)) + (1. / (k*k))

            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            k -= f/f_prime

            if np.isnan(f):
                logger.warning("Weibull fit failed. Returning NaN values.")
                return np.nan, np.nan
            if abs(k - k_t_1) < eps:
                logger.debug(f"Weibull fit converged after {i+1} iterations. k={k}.")
                break
            k_t_1 = k
        if i == iters - 1:
            logger.warning(f"Weibull fit did not converge after {iters} iterations. Last k={k}.")
        lam = np.mean(x ** k) ** (1.0 / k)
        return k, lam

    @staticmethod
    def cdf(x, shape, scale):
        return weibull_min.cdf(x, shape, scale=scale)
    
    @staticmethod
    def pdf(x, shape, scale):
        return weibull_min.pdf(x, shape, scale=scale)

class EmpiricalCDFEstimate:
    """Two-stage plug-in empirical CDF estimator with data-driven bandwidths.

    Steps (as described in the request):
      1) Compute pilot KDEs using Silverman's rule of thumb per block to obtain
         \hat f_{pilot}, \hat F_{pilot}, and \hat f'_{pilot}.
      2) Plug these pilot functionals into the alpha/beta definitions and into the
         integrals for c and Q, yielding \hat h_opt = -1/2 Q^{-1} c.

    Args:
        data: list of measurement blocks, each a list/array of floats.
        kernel_functions: tuple(kernel_pdf, kernel_cdf). Defaults to Gaussian kernel.
        optimization_position: passed through to the bandwidth optimization logic
            ("global" or a specific evaluation point / quantile_x).
        kernel_pdf_prime: optional derivative of kernel PDF; if omitted, an analytic
            Gaussian derivative or numerical derivative is used.
    """

    def __init__(
        self,
        data: list[list[float]],
        kernel_functions: tuple = None,
        optimization_position: str | int | float = "global",
        kernel_pdf_prime=None,
    ) -> None:
        if len(data) < 1:
            raise ValueError("The number of samples should be at least 1.")
        if any(len(d) < 1 for d in data):
            raise ValueError("Each sample should contain at least 1 measurement.")

        self.data = [np.asarray(d, dtype=float) for d in data]
        self.m = len(self.data)

        if kernel_functions is None:
            kernel_pdf, kernel_cdf = scipy.stats.norm.pdf, scipy.stats.norm.cdf
        else:
            kernel_pdf, kernel_cdf = kernel_functions

        self.bandwidth_calculator = BandwidthCalculator(
            self.data,
            kernel_pdf,
            kernel_cdf,
            optimization_position=optimization_position,
            kernel_pdf_prime_func=kernel_pdf_prime,
        )

        self.h_bin_estimates = self.bandwidth_calculator.compute_optimal_binwise_bandwidth()
        self.h_global_estimate = self.bandwidth_calculator.compute_optimal_global_bandwidth()

        if self.h_bin_estimates is None:
            # Fallback to global estimate if binwise failed
            if self.h_global_estimate is None:
                raise ValueError("Unable to estimate bandwidths from data.")
            self.h_bin_estimates = np.array([self.h_global_estimate for _ in range(self.m)])

        if self.h_global_estimate is None:
            # derive a conservative global estimate from binwise values
            self.h_global_estimate = float(np.mean(self.h_bin_estimates))

        self.kernel_cdf = kernel_cdf

    def _cdf_estimate(self, y: np.ndarray, measurement: np.ndarray, h: np.ndarray):
        y = np.atleast_1d(y)
        measurement = np.atleast_1d(measurement)
        diff = (y[:, None] - measurement[None, :]) / h
        return np.mean(self.kernel_cdf(diff), axis=1)

    def cdf(self, y: np.ndarray, mode: str = "binwise"):
        if np.isscalar(y):
            scalar_input = True
        else:
            scalar_input = False
        y = np.atleast_1d(y)

        match mode:
            case "binwise":
                h_vec = self.h_bin_estimates
            case "global":
                h_vec = [self.h_global_estimate for _ in range(self.m)]
            case _:
                raise ValueError("mode should be either 'binwise' or 'global'.")

        cdf_values = np.array(
            [self._cdf_estimate(y, meas, h) ** len(meas) for meas, h in zip(self.data, h_vec, strict=True)]
        )

        result = np.mean(cdf_values, axis=0)
        if scalar_input:
            return result.item()
        return result

    @property
    def bandwidths(self) -> np.ndarray:
        """Return the data-driven plug-in bandwidth vector."""
        return np.asarray(self.h_bin_estimates, dtype=float)


