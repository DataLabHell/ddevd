"""Tests for ``ddevd.transforms`` and the ``DDEVD(..., transform=...)`` flag.

Three groups of tests:

  1. unit tests on each Transform: forward/inverse roundtrip, derivative
     consistency with finite differences, monotonicity.
  2. ``resolve_transform`` parsing: None / "log" / "sqrt" / instances /
     unsupported.
  3. DDEVD integration: identity transform reproduces default DDEVD; log
     transform produces well-defined CDF / quantile / return_levels;
     end-to-end ``predict_RL_y`` matches a manual log-then-untransform
     recipe to several decimal places.
"""
from __future__ import annotations

import numpy as np
import pytest

from ddevd.ddevd import DDEVD
from ddevd.evd import GEV
from ddevd.mev import MEV
from ddevd.transforms import (
    Affine,
    BoxCox,
    Identity,
    Log,
    Power,
    Sqrt,
    Transform,
    resolve_transform,
)


# =========================================================================
# 1. Transform unit tests
# =========================================================================
@pytest.mark.parametrize("transform,sample", [
    (Identity(),                        np.array([0.5, 1.0, 5.0, 17.3])),
    (Log(),                              np.array([0.1, 1.0, 5.0, 17.3])),
    (Log(eps=0.5),                       np.array([0.0, 0.5, 5.0, 17.3])),
    (Sqrt(),                             np.array([0.0, 0.25, 4.0, 17.3])),
    (Power(p=0.5),                       np.array([0.1, 1.0, 5.0, 17.3])),
    (Power(p=2.0),                       np.array([0.1, 1.0, 5.0, 17.3])),
    (Power(p=3.0),                       np.array([0.1, 1.0, 5.0, 17.3])),
    (Affine(a=2.0, b=-1.0),              np.array([0.0, 1.0, 5.0, 17.3])),
    (BoxCox(lam=0.5),                    np.array([0.1, 1.0, 5.0, 17.3])),
    (BoxCox(lam=0.0),                    np.array([0.1, 1.0, 5.0, 17.3])),
    (BoxCox(lam=2.0),                    np.array([0.1, 1.0, 5.0, 17.3])),
])
def test_forward_inverse_roundtrip(transform: Transform, sample: np.ndarray):
    """For every implemented transform, ``inverse(forward(y)) == y`` to high
    precision on a representative range of positive y."""
    z = transform.forward(sample)
    y = transform.inverse(z)
    np.testing.assert_allclose(y, sample, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("transform,sample", [
    (Log(),                              np.array([0.5, 1.0, 5.0, 17.3])),
    (Sqrt(),                             np.array([0.25, 1.0, 4.0, 17.3])),
    (Power(p=0.5),                       np.array([0.5, 1.0, 5.0, 17.3])),
    (Power(p=2.0),                       np.array([0.1, 1.0, 5.0])),
    (Affine(a=3.0, b=2.0),               np.array([0.0, 1.0, 5.0, 17.3])),
    (BoxCox(lam=0.3),                    np.array([0.5, 1.0, 5.0, 17.3])),
])
def test_derivative_matches_finite_difference(transform: Transform,
                                                sample: np.ndarray):
    """Analytical T'(y) should agree with central finite differences."""
    eps = 1e-6
    fd = (transform.forward(sample + eps) - transform.forward(sample - eps)) / (2 * eps)
    analytic = transform.derivative(sample)
    np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("transform", [
    Identity(), Log(), Log(eps=0.1), Sqrt(),
    Power(p=0.7), Power(p=2.0),
    Affine(a=1.5, b=-0.5),
    BoxCox(lam=0.5),
])
def test_forward_is_monotonic_increasing(transform: Transform):
    """forward(y) must be strictly increasing on its valid domain."""
    # Use a domain that is valid for every transform we test.
    y = np.linspace(0.1, 50.0, 200)
    z = transform.forward(y)
    diffs = np.diff(z)
    # Allow tiny numerical noise but require overall monotone increase.
    assert np.all(diffs > -1e-9), \
        f"{transform.name} forward is non-monotonic: min diff = {diffs.min()}"


def test_power_rejects_nonpositive_exponent():
    with pytest.raises(ValueError):
        Power(p=0)
    with pytest.raises(ValueError):
        Power(p=-1.5)


def test_affine_rejects_zero_slope():
    with pytest.raises(ValueError):
        Affine(a=0.0, b=1.0)


def test_from_callable_factory():
    """from_callable wires up forward/inverse/derivative correctly."""
    t = Transform.from_callable(
        forward=lambda y: y ** 3,
        inverse=lambda z: np.cbrt(z),
        derivative=lambda y: 3 * y ** 2,
        name="cubic",
    )
    sample = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(t.inverse(t.forward(sample)), sample, atol=1e-10)
    np.testing.assert_allclose(t.derivative(np.array([2.0])), [12.0])
    assert t.name == "cubic"


def test_log_eps_handles_zero():
    """Log(eps>0) should be finite at y=0."""
    t = Log(eps=1.0)
    z = t.forward(np.array([0.0]))
    assert np.isfinite(z[0])
    np.testing.assert_allclose(t.inverse(z), np.array([0.0]), atol=1e-12)


# =========================================================================
# 2. resolve_transform parsing
# =========================================================================
def test_resolve_transform_none_returns_identity():
    t = resolve_transform(None)
    assert isinstance(t, Identity)


@pytest.mark.parametrize("spec", ["none", "None", "identity", "Identity"])
def test_resolve_transform_identity_strings(spec):
    t = resolve_transform(spec)
    assert isinstance(t, Identity)


@pytest.mark.parametrize("spec, expected_cls", [
    ("log",  Log),
    ("LOG",  Log),
    ("sqrt", Sqrt),
    ("Sqrt", Sqrt),
])
def test_resolve_transform_known_names(spec, expected_cls):
    t = resolve_transform(spec)
    assert isinstance(t, expected_cls)


def test_resolve_transform_passthrough_instance():
    inst = BoxCox(lam=0.4)
    t = resolve_transform(inst)
    assert t is inst


def test_resolve_transform_unsupported_string():
    with pytest.raises(TypeError):
        resolve_transform("inverse-hyperbolic-pizza")


def test_resolve_transform_unsupported_type():
    with pytest.raises(TypeError):
        resolve_transform(42.0)


# =========================================================================
# 3. DDEVD integration
# =========================================================================
def test_ddevd_identity_transform_matches_default(small_block_data):
    """transform=None and transform=Identity() should yield identical fits
    bit-for-bit (same RNG-free pipeline)."""
    a = DDEVD(small_block_data, h_opt_position="global", transform=None)
    b = DDEVD(small_block_data, h_opt_position="global", transform=Identity())
    np.testing.assert_allclose(a.h_global_estimate, b.h_global_estimate)
    grid = np.linspace(0.5, 10, 25)
    np.testing.assert_allclose(a.cdf(grid), b.cdf(grid))


def test_ddevd_log_transform_runs(small_block_data):
    """DDEVD with log transform constructs without error and gives a usable
    CDF / return-level surface."""
    dist = DDEVD(small_block_data, h_opt_position="global", transform="log")
    grid = np.linspace(0.5, 10, 25)
    F = dist.cdf(grid)
    # CDF in [0, 1] and monotone non-decreasing.
    assert np.all(F >= -1e-9)
    assert np.all(F <= 1 + 1e-9)
    assert np.all(np.diff(F) >= -1e-6)
    # Return levels are finite and increasing in T.
    rls = dist.return_levels([2, 5, 10, 20])
    assert all(np.isfinite(rls))
    assert all(rls[i] <= rls[i+1] for i in range(len(rls)-1))


def test_ddevd_log_transform_matches_manual_pipeline(small_block_data):
    """Fitting with ``transform="log"`` produces the same return-level
    estimate as fitting on log(data) externally and exponentiating the
    result."""
    # internal-transform pipeline
    dist_internal = DDEVD(small_block_data, h_opt_position="global",
                           transform="log")
    rl_internal = dist_internal.return_levels([5, 10, 20])

    # manual-transform pipeline: log inputs, then exp the answers
    log_data = [np.log(b) for b in small_block_data]
    dist_manual = DDEVD(log_data, h_opt_position="global", transform=None)
    rl_manual_z = dist_manual.return_levels([5, 10, 20])
    rl_manual = [float(np.exp(z)) for z in rl_manual_z]

    np.testing.assert_allclose(rl_internal, rl_manual, rtol=1e-5, atol=1e-6)


def test_ddevd_sqrt_transform_matches_manual_pipeline(small_block_data):
    """Same equivalence for the sqrt transform."""
    dist_internal = DDEVD(small_block_data, h_opt_position="global",
                           transform=Sqrt())
    rl_internal = dist_internal.return_levels([5, 10, 20])

    sqrt_data = [np.sqrt(b) for b in small_block_data]
    dist_manual = DDEVD(sqrt_data, h_opt_position="global", transform=None)
    rl_manual_z = dist_manual.return_levels([5, 10, 20])
    rl_manual = [float(z) ** 2 for z in rl_manual_z]

    np.testing.assert_allclose(rl_internal, rl_manual, rtol=1e-5, atol=1e-6)


def test_ddevd_log_cdf_invariance(small_block_data):
    """F_Y(y) = F_Z(T(y)) must hold by construction; verify on a grid."""
    dist = DDEVD(small_block_data, h_opt_position="global", transform=Log())
    y = np.array([0.5, 1.0, 2.0, 5.0, 8.0])
    # F_Y(y) -- the public API
    F_y = dist.cdf(y)
    # F_Z(T(y)) -- via the private z-space helper
    F_z = dist._cdf_z(np.log(y))
    np.testing.assert_allclose(F_y, F_z, rtol=1e-9, atol=1e-12)


def test_ddevd_quantile_inverts_cdf_with_transform(small_block_data):
    """For random q values, ``cdf(quantile(q)) == q`` to bisection
    tolerance.  Tests that the z-space bisection in the new ``quantile``
    is correct."""
    dist = DDEVD(small_block_data, h_opt_position="global", transform=Log())
    rng = np.random.default_rng(2026)
    qs = rng.uniform(0.1, 0.95, size=4)
    for q in qs:
        y_star = dist.quantile(float(q))
        F_star = dist.cdf(y_star)
        assert abs(F_star - q) < 1e-6, \
            f"quantile<->cdf inconsistent at q={q}: F(y*)={F_star}, y*={y_star}"


# --------------------------------------------------------------------------- #
# 4. Parallel checks on MEV and GEV
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Estimator", [MEV, GEV])
def test_estimator_identity_transform_matches_default(small_block_data, Estimator):
    """transform=None and transform=Identity() should give identical
    parameters and CDF for MEV and GEV alike."""
    a = Estimator(small_block_data, transform=None)
    b = Estimator(small_block_data, transform=Identity())
    grid = np.linspace(0.5, 10, 25)
    np.testing.assert_allclose(a.cdf(grid), b.cdf(grid), rtol=1e-10, atol=1e-10)


# GEV (and DDEVD) accept any real-valued transform output, including ``log(y)``.
# MEV cannot, because its base Weibull fit requires *non-negative* data; for MEV
# we therefore restrict the transform tests to positivity-preserving transforms
# (``sqrt``, ``Power(p)`` with ``p > 0``).  The choice mirrors what makes
# physical sense in practice: MEV's Weibull base CDF is meaningless on negative
# numbers.
def _inverse_pair(transform):
    """Return (forward, inverse) callables for the manual pipeline."""
    if isinstance(transform, Log):
        return np.log, np.exp
    if isinstance(transform, Sqrt):
        return np.sqrt, lambda z: np.asarray(z) ** 2
    if isinstance(transform, Power):
        p = transform.p
        return lambda y: np.asarray(y) ** p, lambda z: np.asarray(z) ** (1.0 / p)
    raise NotImplementedError(transform)


_ESTIMATOR_TRANSFORM_CASES = [
    # (Estimator, transform-instance)
    (GEV, Log()),
    (GEV, Sqrt()),
    (GEV, Power(p=0.5)),
    (MEV, Sqrt()),
    (MEV, Power(p=0.5)),
    (MEV, Power(p=0.7)),
]


@pytest.mark.parametrize("Estimator,transform", _ESTIMATOR_TRANSFORM_CASES)
def test_estimator_transform_runs(small_block_data, Estimator, transform):
    dist = Estimator(small_block_data, transform=transform)
    grid = np.linspace(0.5, 10, 25)
    F = dist.cdf(grid)
    assert np.all(F >= -1e-9)
    assert np.all(F <= 1 + 1e-9)
    assert np.all(np.diff(F) >= -1e-6)
    rls = dist.return_levels([2, 5, 10, 20])
    assert all(np.isfinite(rls))
    assert all(rls[i] <= rls[i+1] for i in range(len(rls)-1))


@pytest.mark.parametrize("Estimator,transform", _ESTIMATOR_TRANSFORM_CASES)
def test_estimator_transform_matches_manual_pipeline(small_block_data,
                                                      Estimator, transform):
    """Internal-transform pipeline ≡ external (transform → fit → inverse)."""
    forward, inverse = _inverse_pair(transform)
    dist_internal = Estimator(small_block_data, transform=transform)
    rl_internal = dist_internal.return_levels([5, 10, 20])

    transformed_data = [forward(b) for b in small_block_data]
    dist_manual = Estimator(transformed_data, transform=None)
    rl_manual_z = dist_manual.return_levels([5, 10, 20])
    rl_manual = [float(inverse(z)) for z in rl_manual_z]

    np.testing.assert_allclose(rl_internal, rl_manual, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("Estimator,transform", _ESTIMATOR_TRANSFORM_CASES)
def test_estimator_quantile_inverts_cdf_with_transform(small_block_data,
                                                        Estimator, transform):
    """cdf(quantile(q)) == q to bisection tolerance."""
    dist = Estimator(small_block_data, transform=transform)
    rng = np.random.default_rng(7)
    qs = rng.uniform(0.1, 0.95, size=4)
    for q in qs:
        y_star = dist.quantile(float(q))
        F_star = dist.cdf(y_star)
        assert abs(F_star - q) < 1e-6, \
            f"{Estimator.__name__}/{transform.name}: q={q}, F(y*)={F_star}, y*={y_star}"


def test_mev_log_transform_raises_on_negative_data(small_block_data):
    """Documented limitation: MEV cannot consume log-transformed data when
    any block contains values <1 (because log(y)<0 and Weibull MLE rejects
    non-positive values).  Either the constructor's Weibull fitter raises,
    or its results are NaN -- both fail loudly downstream.  This test pins
    the behaviour.
    """
    # Weibull MLE issues a logger warning + returns NaN params; the
    # downstream cdf evaluation then yields NaN, which propagates into
    # quantile bisection.  Either way, computing a quantile must fail.
    dist = MEV(small_block_data, transform="log")
    with pytest.raises(ValueError):
        dist.return_levels([10.0])


def test_ddevd_transform_uses_log_for_concentrated_data(rng):
    """A constructed dataset with bulk near zero plus heavy-tail outliers
    should yield a *larger* return level under ``transform='log'`` than
    under no transform (because the log transform smooths the tail rather
    than letting it staircase).  This is a regression-style check that the
    transform actually changes the prediction in the expected direction."""
    blocks = []
    for _ in range(20):
        bulk = rng.exponential(scale=0.2, size=120)         # near-zero spike
        tail = rng.pareto(a=2.0, size=10) * 2.0 + 1.0        # heavy upper tail
        blocks.append(np.concatenate([bulk, tail]))
    rl_none = DDEVD(blocks, h_opt_position="global", transform=None
                     ).return_levels([100.0])[0]
    rl_log = DDEVD(blocks, h_opt_position="global", transform=Log()
                    ).return_levels([100.0])[0]
    assert rl_log > rl_none, \
        f"expected log-transform tail RL_100 > no-transform; got {rl_log:.3f} vs {rl_none:.3f}"
