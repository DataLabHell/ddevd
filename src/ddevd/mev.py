"""The Metastatistical Extreme Value Distribution.

This file contains a class that implements the MEV distribution.
"""

import numpy as np

from scipy.stats import weibull_min

from ddevd.distributions import WeibullDistributionML
from ddevd.evd import ExtremeValueDistribution

class MEV(ExtremeValueDistribution):
    """The Metastatistical Extreme Value Distribution (MEV).

    This class implements the MEV distribution function.
    It is initialized with a list of lists of measured values. The inner lists
    correspond to the observations in a measurement, while the outer list corresponds to different
    measurements of the same sample. The class has a method that returns the distribution function
    of the MEV distribution for a given value.
    """

    def __init__(self, data: list[list[float]]) -> None:
        """Initialize the MEV distribution with a list of lists of measured values.

        Args:
            data (list[list[float]]): A list of lists of measured values.
               The inner lists correspond to the observations in a measurement,
               while the outer list corresponds to different measurements of the same sample.
        """
        super().__init__(data)
        # fit a weibull distribution to each measurement
        self.weibull_dist = []
        for i in range(self.m):
            self.weibull_dist.append(WeibullDistributionML.fit(self.data[i]))
            print(f"Weibull parameters for measurement {i}: {self.weibull_dist[-1]}")

    def cdf(self, x: float) -> float:
        """Return the distribution function of the MEV distribution for a given value.

        Args:
           x (float): The value for which to return the distribution function.

        Returns:
          float: The distribution function of the MEV distribution for the given value.
        """
        return 1 / self.m * sum([weibull_min.cdf(x, self.weibull_dist[i][0], scale=self.weibull_dist[i][1]) ** self.n[i] for i in range(self.m)])