"""The Metastatistical Extreme Value Distribution.

This file contains a class that implements the MEV distribution.

The optional ``transform`` argument follows the same convention as
``DDEVD`` and ``GEV``: data is forward-mapped once before fitting, and
predictions accept and return values on the original y-scale.
"""

import numpy as np

from scipy.stats import weibull_min

from ddevd.distributions import WeibullDistributionML
from ddevd.evd import ExtremeValueDistribution
from ddevd.transforms import Transform


class MEV(ExtremeValueDistribution):
    """The Metastatistical Extreme Value Distribution (MEV).

    This class implements the MEV distribution function.  It is initialised
    with a list of lists of measured values: the inner lists correspond to
    the observations in a measurement and the outer list corresponds to
    different measurements of the same sample.
    """

    def __init__(self, data: list[list[float]],
                 transform: Transform | str | None = None) -> None:
        """Initialize MEV.

        Args:
            data: list of measurement blocks (years × wet-day intensities).
            transform: optional monotone Y -> Z transform applied to the
                data before fitting (e.g. ``"log"``).  Predictions accept
                and return values on the original y-scale; the transform is
                applied internally.  Default: identity (no-op).
        """
        super().__init__(data, transform=transform)
        # Weibull base lives on (0, inf).  Reject transforms that push any
        # observation onto a non-positive z-value, and surface NaN MLE fits
        # as errors rather than letting them propagate silently into
        # ``_cdf_z`` (where they turn the entire CDF into NaN).
        self.weibull_dist = []
        for i in range(self.m):
            block = self.data[i]
            if np.any(block <= 0):
                n_bad = int(np.sum(block <= 0))
                raise ValueError(
                    f"MEV requires strictly positive values on the "
                    f"(transformed) scale; block {i} has "
                    f"{n_bad}/{len(block)} non-positive entries "
                    f"(transform={self.transform.name!r}). The Weibull base "
                    f"distribution is supported on (0, inf)."
                )
            shape, scale = WeibullDistributionML.fit(block)
            if not (np.isfinite(shape) and np.isfinite(scale)):
                raise ValueError(
                    f"Weibull MLE returned NaN for block {i} "
                    f"(transform={self.transform.name!r}); aborting MEV fit."
                )
            self.weibull_dist.append((shape, scale))

    def _cdf_z(self, z, **kwargs):
        return 1 / self.m * sum(
            [
                weibull_min.cdf(
                    z,
                    self.weibull_dist[i][0],
                    scale=self.weibull_dist[i][1],
                ) ** self.n[i]
                for i in range(self.m)
            ]
        )
