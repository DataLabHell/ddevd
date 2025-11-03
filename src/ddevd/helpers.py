"""Helper functions for the DDEVD package.

This module provides various helper functions used throughout the DDEVD package.
"""

from typing import Callable

import numpy as np
from scipy.stats import weibull_min
from scipy.special import gamma

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

from ddevd.distributions import WeibullDistributionML

def heaviside(x: float, x0: float = 0.0) -> float:
    """Return the Heaviside step function for a given value.

    Args:
        x (float): The value for which to return the Heaviside step function.
        x0 (float): The value at which the Heaviside step function jumps from 0 to 1.

    Returns:
        float: The value of the Heaviside step function for the given value.
    """
    if x < x0:
        return 0.0
    if x >= x0:
        return 1.0
    raise ValueError("The value of x must be a float.")


def sigmoid(x: float, x0: float = 0.0, k: float | None = 1.0) -> float:
    """Return the sigmoid function for a given value.

    Args:
       x (float): The value for which to return the sigmoid function.
       x0 (float): The value at which the sigmoid function is 0.5.
       k (float): The steepness of the sigmoid function.

    Returns:
       float: The value of the sigmoid function for the given value.
    """
    if k is None:
        return heaviside(x, x0)
    if k <= 0:
        raise ValueError("The steepness parameter k must be non-negative.")

    return 1 / (1 + np.exp(-k * (x - x0)))


def get_empirical_return_periods(extreme_values):
    """Calculates empirical return periods for a given set of extreme values.

    Parameters:
    extreme_values (array-like): A list or NumPy array of extreme event magnitudes.
                                 These are typically annual maxima or similar.

    Returns:
    tuple: (sorted_magnitudes, return_periods)
        - sorted_magnitudes (np.array): The extreme values sorted in descending order.
        - return_periods (np.array): The corresponding empirical return periods.
    """
    sorted_magnitudes = np.sort(extreme_values)[::-1]
    len_sorted = len(sorted_magnitudes)

    ranks = np.arange(1, len_sorted + 1)

    return_periods = (len_sorted + 1) / ranks

    return sorted_magnitudes, return_periods

def ddevd_weibull_kernel(data: list[list[float]] | None = None, shape: float | None = None) -> tuple[Callable, Callable]:
    """Return the kernel functions for the Weibull distribution.

    Args:
      data (list[list[float]]): The data for which to compute the kernel functions.

    Returns:
      tuple[Callable, Callable]: The kernel functions for the Weibull distribution.
    """
    if data is None and shape is None:
        raise ValueError("Either data or shape must be provided.")
    if shape is not None:
        logger.info(f"Using provided shape parameter: {shape}")
        return (
            lambda x: weibull_min.pdf(x + gamma(1 + 1 / shape), shape),
            lambda x: weibull_min.cdf(x + gamma(1 + 1 / shape), shape),
        )
    full_data = np.concatenate(data)
    shape_param, _ = WeibullDistributionML.fit(full_data)
    logger.info(f"Shape parameter: {shape_param}")
    return (
        lambda x: weibull_min.pdf(x + gamma(1 + 1 / shape_param), shape_param),
        lambda x: weibull_min.cdf(x + gamma(1 + 1 / shape_param), shape_param),
    )
