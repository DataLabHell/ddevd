
"""Base class for extreme value distributions.

This file contains a base class that can be extended to implement different extreme value distributions.
It is initialized with a list of lists of measured values. The inner lists correspond to the observations
in a measurement, while the outer list corresponds to different measurements of the same sample.
The class has methods that return the distribution function, probability density function,
quantile function, and return levels for given values.
"""

import scipy.optimize as opt
import numpy as np

class ExtremeValueDistribution:
    """Base class for extreme value distributions."""

    def __init__(self, data: list[list[float]]) -> None:
        """Initialize the extreme value distribution with a list of lists of measured values.

        Args:
            data (list[list[float]]): A list of lists of measured values.
               The inner lists correspond to the observations in a measurement,
               while the outer list corresponds to different measurements of the same sample.
        """
        self.data = data
        self.m = len(data)
        self.n = [len(d) for d in data]

    def cdf(self, x: float) -> float:
        """Return the distribution function of the extreme value distribution for a given value.

        Args:
           x (float): The value for which to return the distribution function.

        Returns:
          float: The distribution function of the extreme value distribution for the given value.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def pdf(self, x: float) -> float:
        """Return the probability density function of the extreme value distribution for a given value.

        The probability density function is the derivative of the distribution function.

        Args:
          x (float): The value for which to return the probability density function.

        Returns:
          float: The probability density function of the extreme value distribution for the given value.
        """
        dx = 1e-6
        return (self.cdf(x + dx) - self.cdf(x - dx)) / (2 * dx)

    def quantile(self, q: float, **kwargs) -> float:
        """The quantile function for the extreme value distribution."""
        return opt.bisect(lambda y: self.cdf(y, **kwargs) - q, -1e6, 1e6)

    def return_levels(self, return_periods: list[float], **kwargs) -> list[float]:
        """Compute the return levels for a given set of return periods."""

        rp = np.asarray(return_periods, dtype=float)

        if np.any(rp <= 1):
            raise ValueError("Return periods must be > 1.")
        quantile_arguments = 1.0 - 1.0 / rp
        return [self.quantile(float(q), **kwargs) for q in quantile_arguments]


class DistributionEVD(ExtremeValueDistribution):
    """Class for the Metastatistical Extreme Value Distribution directly from distribution.

    This class implements the MEV distribution, but with a known base distribution.
    It is initialized with the distribution (scipy.stats interface) and a list of sample sizes.
    It implements the EVD interface, but the data is generated from the distribution, not from the data.
    """

    def __init__(self, distribution, sample_sizes: list[int]) -> None:
        """Initialize the EVD distribution with a list of lists of measured values.

        Args:
            data (list[list[float]]): A list of lists of measured values.
               The inner lists correspond to the observations in a measurement,
               while the outer list corresponds to different measurements of the same sample.
            distribution (scipy.stats.rv_continuous): The distribution from which to generate the data.
            sample_sizes (list[int]): A list of sample sizes for each measurement.
        """
        self.distribution = distribution
        data = []
        for s in sample_sizes:
            data.append(distribution.rvs(size=s))
        super().__init__(data)

    def cdf(self, x: float) -> float:
        """Return the distribution function of the EVD distribution for a given value.

        Args:
           x (float): The value for which to return the distribution function.

        Returns:
          float: The distribution function of the EVD distribution for the given value.
        """
        return 1 / self.m * sum([self.distribution.cdf(x) ** self.n[i] for i in range(self.m)])
