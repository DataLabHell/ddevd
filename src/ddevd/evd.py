"""Base class for extreme value distributions.

This file contains a base class that can be extended to implement different extreme value distributions.
It is initialized with a list of lists of measured values. The inner lists correspond to the observations
in a measurement, while the outer list corresponds to different measurements of the same sample.
The class has methods that return the distribution function, probability density function,
quantile function, and return levels for given values.

All subclasses share a single ``transform`` mechanism: data is forward-mapped
once in ``__init__`` and predictions are pulled back through the inverse map.
Subclasses are expected to implement ``_cdf_z`` (the CDF on the *transformed*
z-scale); the public ``cdf`` and ``quantile`` methods on the base class wrap
``_cdf_z`` to operate on the original y-scale.
"""

import scipy
import scipy.optimize as opt
import scipy.stats
import numpy as np

from ddevd.transforms import Identity, Transform, resolve_transform


class ExtremeValueDistribution:
    """Base class for extreme value distributions.

    Common machinery for: applying an optional monotone data transform
    (``transform``) once to the input blocks, evaluating the CDF in the
    transformed scale, and inverting via z-space bisection plus a single
    inverse-transform call.
    """

    def __init__(self, data: list[list[float]],
                 transform: Transform | str | None = None) -> None:
        """Initialise the EVD with a list-of-lists of measured values.

        Args:
            data (list[list[float]]): A list of lists of measured values.
                The inner lists correspond to the observations in a
                measurement, while the outer list corresponds to different
                measurements of the same sample.
            transform: Optional monotone Y -> Z transform applied to the
                data before fitting (e.g. ``"log"`` or a ``Sqrt()``
                instance).  Predictions accept and return values on the
                original y-scale; the transform is applied internally.
                Default: identity (no-op, fully backward compatible).
        """
        self.transform: Transform = resolve_transform(transform)
        # Apply forward transform once; everything downstream sees z-space.
        if not isinstance(self.transform, Identity):
            data = [self.transform.forward(np.asarray(d, dtype=float))
                    for d in data]
        else:
            # Coerce to numpy arrays so subclasses don't have to.
            data = [np.asarray(d, dtype=float) for d in data]
        self.data = data
        self.m = len(data)
        self.n = [len(d) for d in data]

    # ------------------------------------------------------------------ #
    def _cdf_z(self, z, **kwargs):
        """CDF on the transformed z-scale.  Subclasses override this."""
        raise NotImplementedError("Subclasses must implement _cdf_z.")

    def cdf(self, x, **kwargs):
        """Distribution function on the original y-scale.

        Internally evaluates ``F_Z(T(x))`` where ``T`` is ``self.transform``.
        """
        scalar_input = np.isscalar(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if isinstance(self.transform, Identity):
            z = x_arr
        else:
            z = self.transform.forward(x_arr)
        result = self._cdf_z(z, **kwargs)
        if scalar_input:
            try:
                return float(result.item() if hasattr(result, "item") else result)
            except Exception:
                return float(np.asarray(result).reshape(-1)[0])
        return result

    def pdf(self, x: float) -> float:
        """Numerical PDF (central differences on cdf)."""
        dx = 1e-6
        return (self.cdf(x + dx) - self.cdf(x - dx)) / (2 * dx)

    def quantile(self, q: float, **kwargs) -> float:
        """Quantile function on the original y-scale.

        With identity transform: bisect ``cdf - q`` on a wide y-bracket.
        With a non-trivial transform: bisect ``_cdf_z - q`` on a z-space
        bracket inferred from the (transformed) data, then map the answer
        back through ``self.transform.inverse``.  z-space bisection is
        much more numerically stable when the transform is logarithmic.
        """
        if isinstance(self.transform, Identity):
            return float(opt.bisect(
                lambda y: self.cdf(y, **kwargs) - q, -1e6, 1e6
            ))
        # bisect in z-space
        try:
            all_z = np.concatenate([np.atleast_1d(d) for d in self.data])
            z_std = float(np.std(all_z) + 1e-9)
            z_lo = float(all_z.min()) - 6.0 * z_std
            z_hi = float(all_z.max()) + 6.0 * z_std
        except Exception:
            z_lo, z_hi = -1e6, 1e6
        z_star = opt.bisect(
            lambda z: self._cdf_z(z, **kwargs) - q, z_lo, z_hi
        )
        return float(self.transform.inverse(np.asarray([z_star]))[0])

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

    The ``transform`` argument is supported for API symmetry, though for a
    fully-known parametric distribution it has no real use case.
    """

    def __init__(self, distribution, sample_sizes: list[int],
                 transform: Transform | str | None = None) -> None:
        self.distribution = distribution
        data = []
        for s in sample_sizes:
            data.append(distribution.rvs(size=s))
        super().__init__(data, transform=transform)

    def _cdf_z(self, z, **kwargs):
        # The known base distribution lives in *original* (y) space, so we
        # have to undo the forward map on the kernel argument before
        # querying it.  Use inverse(z) to recover y, then evaluate the
        # parametric CDF.
        if isinstance(self.transform, Identity):
            y = z
        else:
            y = self.transform.inverse(z)
        return 1 / self.m * sum(
            [self.distribution.cdf(y) ** self.n[i] for i in range(self.m)]
        )


class GEV(ExtremeValueDistribution):
    """Class for the Generalized Extreme Value Distribution (GEV).

    Fitting and prediction happen on the transformed (z) scale; ``cdf`` and
    ``quantile`` on the base class wrap that to operate on y-scale.
    """

    def __init__(self, data: list[list[float]],
                 transform: Transform | str | None = None) -> None:
        super().__init__(data, transform=transform)
        # ``self.data`` is already on the z-scale.  Fit GEV to the
        # transformed annual maxima.
        all_maxima = [float(np.max(d)) for d in self.data]
        self.xi, self.mu, self.sigma = scipy.stats.genextreme.fit(all_maxima)

    def _cdf_z(self, z, **kwargs):
        return scipy.stats.genextreme.cdf(
            z, c=self.xi, loc=self.mu, scale=self.sigma
        )

    def pdf(self, x: float) -> float:
        # Y-space PDF via change-of-variables: f_Y(y) = f_Z(T(y)) * |T'(y)|
        if isinstance(self.transform, Identity):
            return scipy.stats.genextreme.pdf(
                x, c=self.xi, loc=self.mu, scale=self.sigma
            )
        # Fall back to numerical differentiation on the y-scale.
        return super().pdf(x)
