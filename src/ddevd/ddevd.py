"""The Data Driven Extreme Value Distribution.

This file contains a class that implements the DDEVD.
It is initialized with a list of lists of measured values. The inner lists
correspond to the observations in a measurement, while the outer list corresponds to different
measurements of the same sample. The class has a method that returns the distribution function
of the DDEVD for a given value.
"""

import logging
from typing import Callable

import numpy as np
import scipy.optimize as opt
import scipy.special
from scipy.stats import norm, weibull_min
from tqdm import tqdm

from ddevd.evd import ExtremeValueDistribution
from ddevd.optimal_bandwidth import BandwidthCalculator

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

class DDEVD(ExtremeValueDistribution):
    """Class for the data driven extreme value distribution (DDEVD)."""

    def __init__(
        self,
        data: list[list[float]],
        h_opt_position: str | int | float = "global",
        kernel_functions: tuple[Callable, Callable] = None,
        target_distribution = scipy.stats.norm,
        max_bins: int | None = None,
        use_scaling: bool = False,
    ) -> None:
        """Initialize the DDEVD class.

        The computations to get the parameters of the DDEVD are done in the constructor.
        See the paper for details.

        Args:
          data (list[list[float]]): The data for which to initialize the DDEVD class.
          h_opt_position (str | int | float): The position at which of the optimal bandwidth is computed.
            Can be either 'global', 'quantile_x', a float or an int.
            If 'global', the Mean integrated square error (MISE) is computed.
            If 'quantile_x', the quantile mean integrated square error (q-MISE) at the given quantile x of the data is used,
              That is the integration will be done from the quantile to infinity.
              Note that x must be between 0 and 1.
            If a float, the MSE at the given value is used.
            If an int, the MSE at the given value is used.
          kernel_functions (tuple[Callable, Callable]): The kernel functions to use for the DDEVD.
            The first function is the kernel-PDF, the second the kernel-CDF.
            If None, the standard normal kernel is used.
        """
        if len(data) < 1:
            raise ValueError("The number of samples should be at least 1.")
        if any(len(d) < 1 for d in data):
            raise ValueError("Each sample should contain at least 1 measurement.")
        
        data = [np.array(d, dtype=float) for d in data if len(d) >= 10]
        if len(data) != len(data):
            logger.warning("Removed %d samples because they contained less than 10 measurements.", len(data) - len(data))

        super().__init__(data)

        self.kernel_pdf = kernel_functions[0] if kernel_functions is not None else norm.pdf
        self.kernel_cdf = kernel_functions[1] if kernel_functions is not None else norm.cdf

        if h_opt_position == "global":
            opt_pos = h_opt_position
        elif isinstance(h_opt_position, (int, float)):
            opt_pos = h_opt_position
        elif isinstance(h_opt_position, str) and h_opt_position.startswith("quantile"):
            opt_pos = h_opt_position
        else:
            raise ValueError("The h_opt_position should be either 'global', 'quantile_x', a float or an int.")
        logger.info("Starting optimal bandwidth calculation.")

        self.bw_calcs = []
        if max_bins is not None and len(data) > max_bins:
            number_of_bins = len(data) // max_bins + 1
            elements_per_bin = len(data) // number_of_bins
            num_bins_revised = len(data) // elements_per_bin + 1
            if elements_per_bin < 1:
                raise ValueError(f"Too many bins ({number_of_bins}) for the given data ({len(data)}).")
            logger.info(f"Data will be split into {number_of_bins} bins with {elements_per_bin} elements each.")
            data_split = [
                data[i * elements_per_bin : (i + 1) * elements_per_bin]
                for i in range(num_bins_revised)
            ]
            if len(data_split[-1]) == 0:
                logger.info("Last data split is empty, removing it.")
                data_split = data_split[:-1]
            logger.info(f"Data split into {len(data_split)} groups of size {elements_per_bin}.")
            for i, d in enumerate(data_split):
                logger.info(f"Group {i + 1}: {len(d)} measurements.")
                self.bw_calcs.append(
                    BandwidthCalculator(
                        [np.array(dd) for dd in d],
                        self.kernel_pdf,
                        self.kernel_cdf,
                        opt_pos,
                        target_distribution=target_distribution,
                        use_scaling=use_scaling
                    )
                )
            
            self.h_bin_estimates = np.concatenate([bw_calc.compute_optimal_binwise_bandwidth() for bw_calc in self.bw_calcs])
            self.h_global_estimate = np.mean([bw_calc.compute_optimal_global_bandwidth() for bw_calc in self.bw_calcs])
        else:
            logger.info(f"Data not split, using all {len(data)} measurements.")
            self.bw_calcs = [BandwidthCalculator(
                [np.array(d) for d in data],
                self.kernel_pdf,
                self.kernel_cdf,
                opt_pos,
                target_distribution=target_distribution,
                use_scaling=use_scaling
            )]

            self.h_bin_estimates = self.bw_calcs[0].compute_optimal_binwise_bandwidth()
            self.h_global_estimate = self.bw_calcs[0].compute_optimal_global_bandwidth()
        if self.h_global_estimate is None:
            raise ValueError("Unable to find optimal bandwidth.")
        elif self.h_bin_estimates is None or any(h is None for h in self.h_bin_estimates):
            logger.warning("The binwise estimate of the optimal bandwidth failed. Using global estimate.")
            self.h_bin_estimates = [self.h_global_estimate for _ in range(len(data))]
        else:
            logger.info("Successful optimization of bandwidths.")

    def _cdf_estimate(self, y: np.ndarray, measurement: np.ndarray, h: np.ndarray):
        """Estimate the base data distribution CDF for a given measurement using NumPy operations.

        Args:
          y (np.ndarray): The points at which to evaluate the CDF.
          measurement (np.ndarray): The measurement for which to estimate the CDF.
          h (np.ndarray): The bandwidth for the measurement.
          shape_param (float): The shape parameter for the Weibull distribution.

        Returns:
          np.ndarray: The estimated CDF values.
        """
        # Ensure y is at least a 1D array
        y = np.atleast_1d(y)
        measurement = np.atleast_1d(measurement)
        if h.shape == y.shape:
            diff = (y[:, None] - measurement[None, :]) / h[None, :]
        else:
            diff = (y[:, None] - measurement[None, :]) / h
        return np.mean(self.kernel_cdf(diff), axis=1)

    def cdf(self, y: np.ndarray, mode="binwise", alternative_data=None):
        """The CDF for the DDEVD using vectorized operations.

        The extreme value CDF is computed from the base data distribution CDF.

        Args:
          y (np.ndarray): The points at which to evaluate the CDF.
          mode (str): The mode to use for the CDF estimation. Can be either "binwise" or "global".
          alternative_data (list[list[float]]): Alternative data to use for the CDF estimation (instead of self.data).

        Returns:
          np.ndarray: The CDF values.

        Note:
          If a single value is provided for y, a single value is returned.
        """
        if np.isscalar(y):
            scalar_input = True
        else:
            scalar_input = False
        y = np.atleast_1d(y)

        match mode:
            case "binwise":
                h = self.h_bin_estimates
            case "global":
                h = [self.h_global_estimate for _ in range(self.m)]
            case _:
                raise ValueError("The mode should be either 'binwise' or 'global'.")

        if alternative_data is not None:
            cdf_values = np.array(
                [self._cdf_estimate(y, meas, h) ** len(meas) for meas, h in zip(alternative_data, h, strict=True)]
            )
        else:
            cdf_values = np.array(
                [self._cdf_estimate(y, meas, h) ** len(meas) for meas, h in zip(self.data, h, strict=True)]
            )

        if scalar_input:
            return np.mean(cdf_values, axis=0).item()

        return np.mean(cdf_values, axis=0)
        
    def bootstrap_return_levels(self, return_periods: list[float], n_resample: int = 1000, mode="binwise"):
        """The bootstrap method for the DDEVD.

        Args:
          return_periods (list[float]): The return periods for which to compute the return levels.
          n_resample (int): The number of bootstrap samples to generate.
          mode (str): The mode to use for the CDF estimation. Can be either "binwise" or "global".

        Returns:
          list: A list of bootstrap samples.
        """
        return_level_list = []
        generator = np.random.default_rng()
        for _ in tqdm(range(n_resample), desc=f"Bootstrapping, mode: {mode}", unit="sample"):
            resampled_data = [
                self.data[i] for i in generator.choice(range(len(self.data)), size=len(self.data), replace=True)
            ]
            return_level_list.append(self.return_levels(return_periods, mode=mode, alternative_data=resampled_data))

        return np.array(return_level_list)
