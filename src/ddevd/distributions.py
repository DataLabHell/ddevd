"""Various statistical distributions used in DDEVD.

This file contains implementations of different statistical distributions that can be used in the DDEVD.
"""

import numpy as np
from scipy.stats import weibull_min
import logging

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
            logger.warning("Weibull fit failed due to non-positive values in input data. Returning NaN values.")
            return np.nan, np.nan
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

