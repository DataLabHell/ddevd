"""Monotone data transformations for DDEVD.

Provides a small ``Transform`` base class and a handful of concrete subclasses
so users can fit DDEVD on a transformed scale (e.g. ``log y`` or ``sqrt y``)
while still asking for return levels in the original y-units.

A transform is any strictly monotone differentiable map ``T : Y -> Z``.
Fitting kernel-based DDEVD on ``Z = T(Y)`` is equivalent to a *locally
adaptive* bandwidth in y-space:

    h_Y(y) = h_Z / |T'(y)|

so a "compressing" transform (log, sqrt) automatically yields a wider
bandwidth in the heavy upper tail.

The CDF is monotone-invariant, ``F_Y(y) = F_Z(T(y))``, so:
  * to evaluate the y-space CDF, push y through ``forward`` and ask the
    z-space estimator;
  * to invert (quantile / return-level), bisect on z and pull the answer
    back through ``inverse``.

Implementations should accept array-like input and return numpy arrays of
the same shape, with the exception of ``inverse`` which is also called on
scalars during quantile inversion.
"""
from __future__ import annotations

import numpy as np


class Transform:
    """Strictly monotone differentiable transform Y -> Z."""

    name: str = "identity"

    def forward(self, y):
        raise NotImplementedError

    def inverse(self, z):
        raise NotImplementedError

    def derivative(self, y):
        """T'(y).  Only required if you want the implicit local-bandwidth
        ``h_Y(y) = h_Z / |T'(y)|`` exposed; the kernel CDF / return-level
        machinery does not need it."""
        raise NotImplementedError(f"{self.name}.derivative not implemented")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_callable(cls, forward, inverse, *, derivative=None,
                       name: str = "custom") -> "Transform":
        """Build a transform from user-supplied callables.

        Useful for one-off transforms that don't fit into the standard
        families.  The caller is responsible for ensuring monotonicity and
        differentiability.
        """
        t = cls()
        t.name = name
        t.forward = lambda y: np.asarray(forward(np.asarray(y, dtype=float)))
        t.inverse = lambda z: np.asarray(inverse(np.asarray(z, dtype=float)))
        if derivative is not None:
            t.derivative = lambda y: np.asarray(derivative(np.asarray(y, dtype=float)))
        return t


# --------------------------------------------------------------------------- #
class Identity(Transform):
    name = "identity"

    def forward(self, y):
        return np.asarray(y, dtype=float)

    def inverse(self, z):
        return np.asarray(z, dtype=float)

    def derivative(self, y):
        return np.ones_like(np.asarray(y, dtype=float))


class Log(Transform):
    """``z = log(y + eps)``.  ``eps`` keeps zero/negative values finite."""

    def __init__(self, eps: float = 0.0):
        self.eps = float(eps)
        self.name = f"log(eps={eps})" if eps else "log"

    def forward(self, y):
        return np.log(np.asarray(y, dtype=float) + self.eps)

    def inverse(self, z):
        return np.exp(np.asarray(z, dtype=float)) - self.eps

    def derivative(self, y):
        return 1.0 / (np.asarray(y, dtype=float) + self.eps)


class Sqrt(Transform):
    """``z = sqrt(y)`` for non-negative y."""

    name = "sqrt"

    def forward(self, y):
        y = np.asarray(y, dtype=float)
        return np.sqrt(np.maximum(y, 0.0))

    def inverse(self, z):
        z = np.asarray(z, dtype=float)
        return np.maximum(z, 0.0) ** 2

    def derivative(self, y):
        y = np.asarray(y, dtype=float)
        # T'(y) = 1/(2 sqrt(y)); guard against y=0
        return 0.5 / np.sqrt(np.maximum(y, 1e-12))


class Power(Transform):
    """``z = y^p`` for p > 0."""

    def __init__(self, p: float):
        if p <= 0:
            raise ValueError(f"Power exponent must be > 0; got {p}")
        self.p = float(p)
        self.name = f"power(p={p})"

    def forward(self, y):
        y = np.asarray(y, dtype=float)
        return np.maximum(y, 0.0) ** self.p

    def inverse(self, z):
        z = np.asarray(z, dtype=float)
        return np.maximum(z, 0.0) ** (1.0 / self.p)

    def derivative(self, y):
        y = np.asarray(y, dtype=float)
        return self.p * np.maximum(y, 1e-12) ** (self.p - 1.0)


class Affine(Transform):
    """``z = a * y + b`` -- pure linear rescaling.  Doesn't change the shape
    of the optimisation, included mostly for completeness."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        if a == 0:
            raise ValueError("Affine slope must be non-zero")
        self.a = float(a)
        self.b = float(b)
        self.name = f"affine(a={a},b={b})"

    def forward(self, y):
        return self.a * np.asarray(y, dtype=float) + self.b

    def inverse(self, z):
        return (np.asarray(z, dtype=float) - self.b) / self.a

    def derivative(self, y):
        return np.full_like(np.asarray(y, dtype=float), self.a)


class BoxCox(Transform):
    """One-parameter Box-Cox.  ``z = (y^lam - 1)/lam`` for ``lam != 0``,
    ``log y`` for ``lam == 0``.  Requires y > 0."""

    def __init__(self, lam: float):
        self.lam = float(lam)
        self.name = f"boxcox(lam={lam})"

    def forward(self, y):
        y = np.asarray(y, dtype=float)
        if self.lam == 0.0:
            return np.log(y)
        return (np.power(y, self.lam) - 1.0) / self.lam

    def inverse(self, z):
        z = np.asarray(z, dtype=float)
        if self.lam == 0.0:
            return np.exp(z)
        return np.power(self.lam * z + 1.0, 1.0 / self.lam)

    def derivative(self, y):
        y = np.asarray(y, dtype=float)
        if self.lam == 0.0:
            return 1.0 / y
        return np.power(y, self.lam - 1.0)


# --------------------------------------------------------------------------- #
def resolve_transform(spec) -> Transform:
    """Convenience: accept either a Transform instance or a string name and
    return a Transform.  Falls back to ``Identity`` for ``None``."""
    if spec is None:
        return Identity()
    if isinstance(spec, Transform):
        return spec
    if isinstance(spec, str):
        spec = spec.lower()
        if spec in ("none", "identity"):
            return Identity()
        if spec == "log":
            return Log()
        if spec == "sqrt":
            return Sqrt()
        # power(...), boxcox(...) -- not parsed here; pass instances directly.
    raise TypeError(f"unsupported transform spec: {spec!r}")
