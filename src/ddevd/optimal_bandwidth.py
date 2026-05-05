"""Implementation of the optimal bandwidth selection algorithm for DDEVD."""

import functools
import logging

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats

logger = logging.getLogger(__name__)
EPS = np.finfo(float).eps


@functools.lru_cache(maxsize=None)
def log_double_factorial(n):
    """Computes log(n_arg!!)."""
    if n < -1:
        raise ValueError("Double factorial is not defined for negative numbers.")
    if n <= 0:  # 0!! = 1
        return 0.0
    if n == 1:  # 1!! = 1
        return 0.0
    if n % 2 == 1:
        return np.sum(np.log(np.arange(1, n + 1, 2)))
    return np.sum(np.log(np.arange(2, n + 1, 2)))


@functools.lru_cache(maxsize=None)
def log_factorial(n):
    """Compute log(n!)."""
    if n <= 1:
        return 0.0
    return scipy.special.gammaln(n + 1)


@functools.lru_cache(maxsize=None)
def log_binomial(n, k):
    """Computes log(n choose k)."""
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n. Got k = {}, n = {}".format(k, n))
    if k == 0 or k == n:
        return 0.0
    return log_factorial(n) - log_factorial(k) - log_factorial(n - k)


class BandwidthCalculator:
    """Class to compute the optimal bandwidth for DDEVD."""

    def __init__(self,
                 samples: list[list[float]],
                 kernel_pdf_func,
                 kernel_cdf_func,
                 optimization_position="global",
                 target_distribution=None,
                 use_scaling=False,
                 verbose_compute=False,
                 no_distribution_fit=False,
                 kernel_pdf_prime_func=None,
                 pilot_factor: float = 1.06,
                 min_bandwidth: float = 1e-3):
        """Initialize the BandwidthCalculator with samples and kernel functions.

        Parameters
        ----------
        samples : list[list[float]]
            A list of one-dimensional sample arrays, one for each margin used
            in the DDEVD estimation.
        kernel_pdf_func : callable
            Probability density function of the kernel. It must accept an array
            of points and return the corresponding kernel densities.
        kernel_cdf_func : callable
            Cumulative distribution function of the kernel corresponding to
            ``kernel_pdf_func``.
        optimization_position : {"global", "local"}, optional
            Strategy used when optimizing the bandwidth. The exact meaning of
            the options depends on the implementation but typically controls
            whether a single global bandwidth or position-dependent bandwidths
            are used.
        target_distribution : object or None, optional
            The theoretical target distribution to which the DDEVD estimator is
            tuned (for example, a SciPy ``rv_continuous`` instance). If
            provided, it is used directly in the plug-in bandwidth selection.
            If ``None``, an empirical plug-in estimator with iterative
            bandwidth selection based solely on the supplied samples is used.
        use_scaling : bool, optional
            If True, apply scaling of the data or bandwidths before performing
            the optimization.
        verbose_compute : bool, optional
            If True, enable more verbose logging during bandwidth computation.
        no_distribution_fit : bool, optional
            If True, skip fitting a parametric distribution to the data even if
            a ``target_distribution`` is provided.
        kernel_pdf_prime_func : callable or None, optional
            Derivative of ``kernel_pdf_func``. If ``None``, a default
            derivative implementation (for example, a numerical derivative) is
            used internally.
        pilot_factor : float, optional
            Multiplicative factor applied to the pilot bandwidth used in the
            plug-in estimation procedure. This controls the smoothness of the
            pilot estimate that drives the iterative bandwidth selection.
        min_bandwidth : float, optional
            Minimum allowed bandwidth value. This lower bound is enforced
            during optimization to avoid degenerate or numerically unstable
            bandwidths.
        """
        self.use_scaling = use_scaling
        # Coerce each sample to a numpy float array.  The pilot KDE helpers
        # below do ``y_val - sample`` and rely on broadcasting, so a Python
        # ``list`` here would raise ``TypeError: unsupported operand type(s)
        # for -: 'float' and 'list'``.
        self.samples = [np.asarray(s, dtype=float) for s in samples]
        self.m = len(self.samples)
        self.n_vec = np.array([len(sample) for sample in self.samples])
        self.kernel_pdf_func = kernel_pdf_func
        self.kernel_cdf_func = kernel_cdf_func
        self.optimization_position = optimization_position
        self.target_distribution = target_distribution
        self.verbose_compute = verbose_compute
        self.kernel_pdf_prime_func = kernel_pdf_prime_func or self._default_kernel_pdf_prime
        self.pilot_factor = pilot_factor
        self.min_bandwidth = min_bandwidth
        self.iterative = False
        logging.info("Scaling mode: %s", "Enabled" if use_scaling else "Disabled")

        self.mu_k_1 = scipy.integrate.quad(lambda u: u * kernel_pdf_func(u), -np.inf, np.inf)[0]
        self.mu_k_2 = scipy.integrate.quad(lambda u: u**2 * kernel_pdf_func(u), -np.inf, np.inf)[0]

        # Compute K^2 moments
        self.mu_k2_1 = scipy.integrate.quad(lambda u: u * 2 * kernel_pdf_func(u) * kernel_cdf_func(u), -np.inf, np.inf)[
            0
        ]
        self.mu_k2_2 = scipy.integrate.quad(
            lambda u: u**2 * 2 * kernel_pdf_func(u) * kernel_cdf_func(u), -np.inf, np.inf
        )[0]

        logger.info("mu_k_1: %s", self.mu_k_1)
        logger.info("mu_k_2: %s", self.mu_k_2)
        logger.info("mu_k2_1: %s", self.mu_k2_1)
        logger.info("mu_k2_2: %s", self.mu_k2_2)

        self.h_map = []
        for i in range(self.m):
            for j in range(self.n_vec[i]):
                self.h_map.append((i, j))

        # default scale factor to 1.0; may be overwritten below
        self.scale_factor = 1.0

        # If target_distribution is None, fall back to fully empirical plug-in estimator
        if self.target_distribution is None:
            logger.info("Using empirical pilot estimator for bandwidth calculation.")
            self.iterative = True
            self.h_pilot = self._compute_pilot_bandwidths()
            self.pdf = functools.lru_cache(maxsize=None)(self._pilot_pdf_scalar)
            self.cdf = functools.lru_cache(maxsize=None)(self._pilot_cdf_scalar)
            self.pdf_prime = functools.lru_cache(maxsize=None)(self._pilot_pdf_prime_scalar)
            return

        if no_distribution_fit:
            # when no_distribution_fit is requested but no target provided, use standard normal
            dist = self.target_distribution or scipy.stats.norm
            self.pdf = dist.pdf
            self.cdf = dist.cdf
            self.pdf_prime = lambda y, delta=1e-6: (self.pdf(y + delta) - self.pdf(y - delta)) / (2 * delta)
            return

        pooled_data = [x for bin_data in self.samples for x in bin_data]
        if self.use_scaling:
            self.scale_factor = np.quantile(pooled_data, 0.95)
        else:
            self.scale_factor = 1.0
        logger.info("Scale factor: %s", self.scale_factor)
        scaled_data = [x / self.scale_factor for x in pooled_data]
        if len(pooled_data) == 0:
            raise ValueError("Cannot estimate from empty data.")
        if len(pooled_data) == 1:
            val = pooled_data[0]
            print("Warning: Estimating from single data point, using delta approximations.")

            def pdf(y):
                return 1.0 if np.isclose(y, val) else 0.0

            def cdf(y):
                return 1.0 if y >= val else 0.0

            def pdf_prime(_):
                return 0.0

            self.pdf = pdf
            self.cdf = cdf
            self.pdf_prime = pdf_prime

        else:
            distribution_fit = self.target_distribution.fit(scaled_data)
            logger.info("Distribution parameters: %s", distribution_fit)
            @functools.lru_cache(maxsize=None)
            def pdf_est(y):
                return self.target_distribution.pdf(y, *distribution_fit)

            @functools.lru_cache(maxsize=None)
            def cdf_est(y):
                return self.target_distribution.cdf(y, *distribution_fit)

            @functools.lru_cache(maxsize=None)
            def pdf_prime_est(y, delta=1e-6):
                return (pdf_est(y + delta) - pdf_est(y - delta)) / (2 * delta)

            self.pdf = pdf_est
            self.cdf = cdf_est
            self.pdf_prime = pdf_prime_est
            

    def alpha(self, y):
        """Computes the alpha function."""
        logger.debug(
            "Value of alpha at y = %s: %s", y, self.pdf(y) * self.mu_k2_1 - 2 * self.cdf(y) * self.pdf(y) * self.mu_k_1
        )
        return self.pdf(y) * self.mu_k2_1 - 2 * self.cdf(y) * self.pdf(y) * self.mu_k_1

    def beta(self, y):
        """Computes the beta function."""
        term1 = self.pdf_prime(y) * self.mu_k2_2 / 2
        term2 = self.pdf_prime(y) * self.cdf(y) * self.mu_k_2
        term3 = self.pdf(y) ** 2 * self.mu_k_1**2
        logger.debug("Value of beta at y = %s: %s", y, term1 - term2 - term3)
        return term1 - term2 - term3

    def log_rho(self, N, n, y):
        """Computes the logarithm of the function rho_n,n_i(y)."""
        cdf_value = self.cdf(y)
        return 0.5 * np.log(
            n * cdf_value / ((1 - cdf_value) * N + cdf_value * n)
            ) + N**2 * (1 - cdf_value) / 2 / ((1 - cdf_value) * N + cdf_value * n)

    def factor_z(self, N, n, y):
        cdf_value = self.cdf(y)
        c = np.sqrt((1-cdf_value)/n/cdf_value)
        return c*(N- N*c**2-1)/(N*c**2 + 1)**2

    def fy_0(self, y, n, n_i):
        """Computes the function F_X^{n} Y_{0,n} in the large n regime."""
        cdf_value = self.cdf(y)
        if cdf_value <= EPS:
            return 0.0
        if cdf_value >= 1 - EPS:
            return 1.0
        if n_i <= 10:
            raise ValueError("n_i cannot be less than or equal to 10. This method is only valid for large n. Current n_i: {}".format(n_i))
        logger.debug(
            "log_fy_0(y={}, n={}, n_i={}) = {}".format(y, n, n_i, n * np.log(cdf_value) + self.log_rho(n, n_i, y))
        )
        return np.exp(n * np.log(cdf_value) + self.log_rho(n, n_i, y))

    def fy_1(self, y, n, n_i):
        """Computes the function F_X^{n} Y_{1,n} in the large n regime."""
        cdf_value = self.cdf(y)
        if cdf_value <= EPS:
            return 0.0
        if cdf_value >= 1 - EPS:
            return -n * (n - 1) / 2.0 / n_i * self.alpha(y)

        c = np.sqrt((1 - cdf_value) / cdf_value)
        if n_i <= 10:
            raise ValueError("n_i cannot be less than or equal to 10. This method is only valid for large n. Current n_i: {}".format(n_i))
        first_part = (
            np.log(self.alpha(y))
            + np.log(n)
            - 0.5 * np.log(n_i)
            - np.log(2)
            + (n - 1) * np.log(cdf_value)
            - 0.5 * np.log((1 - cdf_value) * cdf_value)
        )
        expectation = self.log_rho(n, n_i, y)
        #expectation = np.log((n - 1) * c * np.sqrt(n_i) / (c**2 * (n - 1) + n_i)) + self.log_rho(n - 1, n_i, y)
        logging.debug("log_fy_1(y={}, n={}, n_i={}) = {}".format(y, n, n_i, first_part + expectation))
        return -np.exp(first_part + expectation)*self.factor_z(n, n_i, y)

    def factor_z2(self, n, n_i, y):
        """Calculates the multiplicative factor for the E[Z^2(1+cZ)^(N-2)] approximation.

        This corresponds to the fractional part of the formula for the second derivative of rho.
        """
        if n < 2:
            return 0.0

        cdf_value = self.cdf(y)
        if cdf_value <= EPS or cdf_value >= 1-EPS: return 0.0

        c_squared = (1 - cdf_value) / (n_i * cdf_value)
        c4 = c_squared**2
        c6 = c_squared**3

        return (2*n**3*c6 + (3*n**2 - 5*n**3)*c4 + (n**3 - 4*n**2)*c_squared + n - 1) / ((n * c_squared + 1)**4 * (n - 1))

    def fy_2_alpha(self, y, n, n_i):
        """Approximates the function FY_{2,alpha,n} using an analytical approximation."""
        cdf_value = self.cdf(y)
        if cdf_value <= EPS:
            return 0.0
        if cdf_value >= 1 - EPS:
            return scipy.special.binom(n, 4) * 3 / n_i**2 * self.alpha(y) ** 2

        if n_i <= 10:
            raise ValueError("n_i cannot be less than or equal to 10. This method is only valid for large n. Current n_i: {}".format(n_i))
        if n < 2:
            return 0.0

        alpha_val = self.alpha(y)
        c = np.sqrt((1 - cdf_value) / (n_i * cdf_value))

        log_P = (
            2 * np.log(np.abs(alpha_val)) + 
            np.log(n) - 
            np.log(8) - 
            np.log(n_i) + 
            (n - 3) * np.log(cdf_value) - 
            np.log(1 - cdf_value)
        )

        factor_1 = (n - 2) * self.factor_z2(n, n_i, y)
        factor_2 = (1 / c) * self.factor_z(n - 1, n_i, y)

        full_log_factor = log_P + self.log_rho(n, n_i, y)

        return np.exp(full_log_factor) * (factor_1 - factor_2 * np.exp(self.log_rho(n - 1, n_i, y) - self.log_rho(n, n_i, y)))


    def fy_2_beta(self, y, n, n_i):
        """Computes the function F_X^{n} Y_{2,beta}."""
        cdf_value = self.cdf(y)
        if cdf_value <= EPS:
            return 0.0
        if cdf_value >= 1 - EPS:
            return n * (n - 1) / 2.0 / n_i * self.beta(y)

        if cdf_value <= EPS or cdf_value >= 1 - EPS:
            raise ValueError("F_X(y) must be in the range (0, 1). Got F_X(y) = {}".format(cdf_value))
        c = np.sqrt((1 - cdf_value) / cdf_value)
        if n_i <= 10:
            raise ValueError("n_i cannot be less than or equal to 10. This method is only valid for large n. Current n_i: {}".format(n_i))
        first_part = np.log(n) - 0.5 * np.log(n_i) - np.log(2) + (n - 2) * np.log(cdf_value) - np.log(c)
        expectation = self.log_rho(n, n_i, y)
        logging.debug("log_fy_2_beta_n_approx(y={}, n={}, n_i={}) = {}".format(y, n, n_i, first_part + expectation))

        return self.beta(y) * np.exp(first_part + expectation) * self.factor_z(n, n_i, y)

    E_0_i = fy_0

    @functools.lru_cache(maxsize=None)
    def E_1_i(self, y, n, n_i):
        """Computes the function E_1,i.

        From the paper: E_{1,N} = FY_{1,N} - N * f_X * mu_{K,1} * FY_{0,N-1}
        Note: the second term vanishes for symmetric (zero-mean) kernels.
        """
        if n_i <= 2:
            raise ValueError("n_i cannot be less than or equal to 2.")
        return self.fy_1(y, n, n_i) - n * self.fy_0(y, n - 1, n_i) * self.pdf(y) * self.mu_k_1

    @functools.lru_cache(maxsize=None)
    def E_2a_i(self, y, n, n_i):
        """Computes the function E_2a,i."""
        if n_i <= 2:
            raise ValueError("n_i cannot be less than or equal to 2.")

        return n / 2.0 * self.fy_0(y, n - 1, n_i) * self.pdf_prime(y) * self.mu_k_2 + self.fy_2_beta(
            y, n, n_i
        )

    @functools.lru_cache(maxsize=None)
    def E_2b_i(self, y, n, n_i):
        """Computes the function E_2b,i."""
        if n_i <= 2:
            raise ValueError("n_i cannot be less than or equal to 2.")
        return (
            n * (n - 1) / 2.0 * self.fy_0(y, n - 2, n_i) * self.pdf(y) ** 2 * self.mu_k_1**2
            + self.fy_2_alpha(y, n, n_i)
            - n * self.fy_1(y, n - 1, n_i) * self.pdf(y) * self.mu_k_1
        )

    @functools.lru_cache(maxsize=None)
    def b_0_i(self, y, n_i):
        """Computes the function b_0,i."""
        if n_i <= 2:
            raise ValueError("n_i cannot be less than or equal to 2.")
        return self.fy_0(y, n_i, n_i) - self.cdf(y) ** n_i

    @functools.lru_cache(maxsize=None)
    def b_1_i(self, y, n_i):
        return self.E_1_i(y, n_i, n_i)

    @functools.lru_cache(maxsize=None)
    def b_2a_i(self, y, n_i):
        return self.E_2a_i(y, n_i, n_i)

    @functools.lru_cache(maxsize=None)
    def b_2b_i(self, y, n_i):
        return self.E_2b_i(y, n_i, n_i)

    @functools.lru_cache(maxsize=None)
    def V_0_i(self, y, n_i):
        """Computes the function V_0,i."""
        return self.E_0_i(y, 2 * n_i, n_i) - self.E_0_i(y, n_i, n_i) ** 2

    @functools.lru_cache(maxsize=None)
    def V_1_i(self, y, n_i):
        """Computes the function V_1,i."""
        return self.E_1_i(y, 2 * n_i, n_i) - 2 * self.E_1_i(y, n_i, n_i) * self.E_0_i(y, n_i, n_i)

    @functools.lru_cache(maxsize=None)
    def V_2a_i(self, y, n_i):
        """Computes the function V_2a,i."""
        return (self.E_2a_i(y, 2 * n_i, n_i) - 2 * self.E_2a_i(y, n_i, n_i) * self.E_0_i(y, n_i, n_i))

    @functools.lru_cache(maxsize=None)
    def V_2b_i(self, y, n_i):
        """Computes the function V_2b,i."""
        return (
            self.E_2b_i(y, 2 * n_i, n_i)
            - 2 * self.E_2b_i(y, n_i, n_i) * self.E_0_i(y, n_i, n_i)
            - self.E_1_i(y, n_i, n_i) ** 2
        )
    
    def _clear_all_caches(self):
        """Clear all LRU caches on this class.

        ``@functools.lru_cache`` stores its cache on the class-level function
        object, not on bound methods.  Calling ``cache_clear()`` via a bound
        method reference silently does nothing; we must go through the
        unbound class attribute instead.
        Note: this clears the shared cache for *all* instances.

        In iterative mode the pilot pdf/cdf/pdf_prime are instance-level
        lru_cache wrappers whose values depend on self.h_pilot, so they must
        be cleared separately whenever h_pilot changes.
        """
        for method_name in [
            'E_1_i', 'E_2a_i', 'E_2b_i',
            'b_0_i', 'b_1_i', 'b_2a_i', 'b_2b_i',
            'V_0_i', 'V_1_i', 'V_2a_i', 'V_2b_i',
        ]:
            fn = getattr(BandwidthCalculator, method_name, None)
            if fn is not None and hasattr(fn, 'cache_clear'):
                fn.cache_clear()

        if self.iterative:
            for attr in ('pdf', 'cdf', 'pdf_prime'):
                fn = getattr(self, attr, None)
                if fn is not None and hasattr(fn, 'cache_clear'):
                    fn.cache_clear()

    @staticmethod
    def find_upper_lower_limits(func, start_lower, start_upper,
                                start_step = 1.0,
                                threshold_lower = 1e-6,
                                threshold_upper = 1 - 1e-6):
        """Finds approximate upper and lower limits for F_X integration"""
        lower = start_lower
        upper = start_upper
        step = start_step
        while func(lower) > threshold_lower:
            lower -= step
            step *= 2
        while func(upper) < threshold_upper:
            upper += step
            step *= 2
        return lower, upper

    def _integrate_y(self, integrand, q=None):
        """Integrate ``integrand(y)`` over the y-axis or, when ``q`` is given,
        over ``{y : F_X(y) >= q}`` using the change of variables ``t = 1-F_X(y)``.

        For heavy-tailed distributions (Cauchy, Pareto with small alpha, etc.)
        the right endpoint of the y-axis integral is at ``y -> infty`` and the
        Jacobian ``1/f_X(y)`` blows up faster than the integrand decays, so plain
        ``scipy.integrate.quad`` on y picks up O(1/eps) cancellation noise and
        can return values that disagree with the true integral by orders of
        magnitude (and by sign).  The Watson-lemma proof in Appendix A.1 works
        in ``t = 1 - F_X(y)``; for finite-q-MISE integrals this is the natural
        variable: ``dy = -1/f_X(y) dt`` and the relevant region is
        ``t in (0, 1-q]``, where the integrand is well-behaved.

        Parameters
        ----------
        integrand : callable
            Function ``y -> R`` to integrate over the y-axis.
        q : float in (0, 1) or None
            If ``None``, integrate over (essentially) the full support using
            the legacy bracketing search.  If a quantile, integrate over
            ``{y : F_X(y) >= q}`` using the t-substitution when the underlying
            distribution provides ``ppf`` and ``pdf``; otherwise fall back to
            a y-axis quad with a finite, n-aware upper limit.
        """
        if q is None:
            lower, upper = self.find_upper_lower_limits(self.cdf, 100, 100)
            return scipy.integrate.quad(integrand, lower, upper)[0]

        if not (0.0 < q < 1.0):
            raise ValueError("q must be in (0, 1); got {}".format(q))

        ppf = getattr(self.target_distribution, "ppf", None)
        pdf = getattr(self.target_distribution, "pdf", None)
        if ppf is None or pdf is None:
            # Fallback: y-axis quad with a finite-but-not-pathological upper.
            lower, upper = self.find_upper_lower_limits(
                self.cdf, 100, 100, threshold_lower=q
            )
            try:
                _, upper_safe = self.find_upper_lower_limits(
                    self.cdf, 100, 100, threshold_upper=1 - 1e-5
                )
                upper = min(upper, float(upper_safe))
            except Exception:
                pass
            return scipy.integrate.quad(integrand, lower, upper)[0]

        # ----- Heavy-tail-safe integration via t = 1 - F_X(y).
        def t_integrand(t):
            if t <= 0.0 or t >= 1.0:
                return 0.0
            y = float(ppf(1.0 - t))
            f = float(pdf(y))
            if f <= 0.0 or not np.isfinite(f):
                return 0.0
            v = integrand(y) / f
            if not np.isfinite(v):
                return 0.0
            return v

        t_max = 1.0 - q
        # Geometrically subdivide near t=0 to help adaptive quad concentrate
        # samples where the Watson-lemma integrand has its mass.
        breakpoints = list(np.geomspace(t_max * 1e-6, t_max, num=8))
        return scipy.integrate.quad(
            t_integrand, 0.0, t_max, limit=200, points=breakpoints
        )[0]

    def _resolve_q(self):
        """Translate ``self.optimization_position`` into either a quantile q
        in (0, 1) or ``None`` for full-support integration.  Returns
        (mode, value): mode is one of {'global', 'quantile', 'point'}."""
        op = self.optimization_position
        if op == "global":
            return ("global", None)
        if isinstance(op, str) and op.startswith("quantile_"):
            return ("quantile", float(op.split("_")[1]))
        if isinstance(op, (int, float)):
            return ("point", float(op))
        raise ValueError(
            "Optimization position must be 'global', 'quantile_X' or a number. "
            "Got: {}".format(op)
        )


    ### Coefficient computation
    def c(self):
        """Compute the coefficient vector."""
        n_vec = np.array([len(sample) for sample in self.samples])
        logger.info("Computing coefficients for n_vec: %s", n_vec)

        def sum_b_0(y):
            return np.sum([self.b_0_i(y, n_i) for n_i in n_vec])

        # c_i(y) = 2 * sum_b_0(y) * b_1_i(y) + V_1_i(y)
        def c_i(y, i):
            return 2 * sum_b_0(y) * self.b_1_i(y, n_vec[i]) + self.V_1_i(y, n_vec[i])

        mode, value = self._resolve_q()
        if mode == "point":
            return [c_i(value, i) for i in range(len(n_vec))]
        q_arg = None if mode == "global" else value
        logger.info("c(): integration mode=%s, q=%s", mode, q_arg)
        return [
            self._integrate_y(lambda y, i=i: c_i(y, i), q=q_arg)
            for i in range(len(n_vec))
        ]

    def Q(self):
        """Compute the Q matrix."""
        logger.info("Computing Q matrix...")
        n_vec = np.array([len(sample) for sample in self.samples])

        Q = np.zeros((len(n_vec), len(n_vec)))

        def q_fun(y, i, j):
            additional_term = 0
            if i == j:
                additional_term = (
                    2
                    * sum([self.b_0_i(y, n_vec[m]) for m in range(len(n_vec))])
                    * (self.b_2a_i(y, n_vec[i]) + self.b_2b_i(y, n_vec[i]))
                )
                additional_term += self.V_2a_i(y, n_vec[i]) + self.V_2b_i(y, n_vec[i])
            return self.b_1_i(y, n_vec[i]) * self.b_1_i(y, n_vec[j]) + additional_term

        mode, value = self._resolve_q()
        q_arg = None if mode == "global" else (value if mode == "quantile" else None)
        logger.info("Q(): integration mode=%s, q=%s", mode, q_arg)
        for k in range(len(n_vec)):
            for r in range(len(n_vec)):
                if mode == "point":
                    Q[k, r] = q_fun(value, k, r)
                else:
                    Q[k, r] = self._integrate_y(
                        lambda y, k=k, r=r: q_fun(y, k, r), q=q_arg
                    )
        logger.info("Q matrix computed: %s", Q)
        return Q

    def D(self, q=None):
        """Compute the stability criterion D from the Lemma (positive definiteness of Q).

        D_q = integral_{F_X(y) >= q} [2*m*b_0(y)*b_2(y) + V_2(y)] dy

        With equal block sizes n_i=n, the Hessian Q has eigenvalues:
          - a = D  (multiplicity m-1, smallest)
          - a + m*b  (multiplicity 1, largest)  where b = integral b_1^2 dy

        Q is positive definite iff D > 0 (Lemma in the paper).

        Parameters
        ----------
        q : float in (0, 1) or None, optional
            Lower-tail quantile defining the integration domain
            ``{y : F_X(y) >= q}``.  This matches the ``D_q`` definition used in
            the asymptotic analysis (Appendix A.1).
            If ``None`` (default) the value is taken from
            ``self.optimization_position``:
              * ``"global"``                 -> integrate over (essentially) the
                full support of ``F_X`` (using ``threshold_lower=1e-6``).  This
                is the *legacy* behaviour and is **not** the quantity bounded
                by Theorem 4.3 -- for heavy-tailed distributions the bulk of
                ``F_X`` can dominate the integral and produce phase boundaries
                that do not follow ``m < C * n^(1+gamma/2)``.
              * ``"quantile_X"`` (with ``X`` a float) -> use ``q = X``.
              * a number -> evaluate the integrand pointwise at that y (no
                integration).
            An explicit ``q`` argument takes precedence over
            ``optimization_position``.
        """
        n = np.mean(self.n_vec)

        def q_fun(y):
            additional_term = (
                    2
                    * self.m * self.b_0_i(y, n)
                    * (self.b_2a_i(y, n) + self.b_2b_i(y, n))
                )
            additional_term += self.V_2a_i(y, n) + self.V_2b_i(y, n)
            return additional_term

        # Resolve which integration domain to use.
        if q is None:
            mode, value = self._resolve_q()
            if mode == "point":
                return q_fun(value)
            q = None if mode == "global" else value

        return self._integrate_y(q_fun, q=q)

    def compute_optimal_global_bandwidth(self):
        """Compute the optimal bandwidth for the given samples."""
        if self.iterative:
            logger.info("Starting iterative computation for global bandwidth...")
            prev_h = self.h_pilot
            early = False
            lambda_ = 0.5
            for iteration in range(100):
                lin = np.sum(self.c())
                qua = np.sum(self.Q())
                bandwidth = -0.5 * lin / qua
                logger.info("Optimal global bandwidth (before rescaling): %s", bandwidth)
                logger.info("Optimal global bandwidth (after rescaling): %s", bandwidth * self.scale_factor)
                if bandwidth < 0:
                    logger.warning("Negative global bandwidth detected.")
                    return None
                self.h_pilot = [bandwidth * lambda_ + (1 - lambda_) * prev for prev in prev_h]
                # Clear caches so updated h_pilot values are used in next iteration
                self._clear_all_caches()
                if np.allclose(prev_h, self.h_pilot, rtol=1e-5, atol=1e-8):
                    logger.info("Converged after %s iterations.", iteration + 1)
                    early = True
                    break
                prev_h = self.h_pilot
            if not early:
                logger.info("Maximum iterations reached without convergence.")
            return bandwidth * self.scale_factor
        else:
            lin = np.sum(self.c())
            qua = np.sum(self.Q())
            bandwidth = -0.5 * lin / qua
            logger.info("Optimal global bandwidth (before rescaling): %s", bandwidth)
            logger.info("Optimal global bandwidth (after rescaling): %s", bandwidth * self.scale_factor)
            if bandwidth < 0:
                logger.warning("Negative global bandwidth detected.")
                return None
            return bandwidth * self.scale_factor

    def compute_optimal_binwise_bandwidth(self):
        """Compute the optimal bandwidth for the given samples."""
        if self.iterative:
            early = False
            lambda_ = 0.5
            logger.info("Starting iterative computation for bin-wise bandwidth...")
            prev_h = self.h_pilot
            for iteration in range(100):
                coeffs = self.c()
                q_matrix = self.Q()
                logger.info("Coefficients: %s", coeffs)
                logger.info("Q matrix: %s", q_matrix)
                if self.verbose_compute:
                    logger.info("Computing spectral decomposition of Q matrix...")
                    # compute the spectral decomposition of the Q matrix
                    try:
                        eigvals, eigvecs = np.linalg.eigh(q_matrix)
                    except np.linalg.LinAlgError as e:
                        logger.error("Error in computing eigenvalues/eigenvectors: %s", e)
                        raise
                    # check if the matrix is positive definite
                    logger.info("Eigenvalues: %s", eigvals)
                    logger.info("Trace: %s", np.trace(q_matrix))
                    logger.info("Determinant: %s", np.linalg.det(q_matrix))

                optimal_bandwidth = -0.5 * np.linalg.solve(q_matrix, coeffs)
                logger.info("Optimal bandwidths (before rescaling): %s", optimal_bandwidth)
                logger.info("Optimal bandwidths (after rescaling): %s", optimal_bandwidth * self.scale_factor)

                # account for negative bandwidths due to numerical errors
                if np.any(optimal_bandwidth < 0):
                    logger.warning("Negative bandwidths detected.")
                    return None
                self.h_pilot = optimal_bandwidth * lambda_ + (1 - lambda_) * np.array(prev_h)
                # Clear caches so updated h_pilot values are used in next iteration
                self._clear_all_caches()
                if np.allclose(prev_h, self.h_pilot, rtol=1e-5, atol=1e-8):
                    logger.info("Converged after %s iterations.", iteration + 1)
                    early = True
                    break
                prev_h = self.h_pilot
            if not early:
                logger.info("Maximum iterations reached without convergence.")
            return optimal_bandwidth * self.scale_factor
        else:
            coeffs = self.c()
            q_matrix = self.Q()
            logger.info("Coefficients: %s", coeffs)
            logger.info("Q matrix: %s", q_matrix)
            optimal_bandwidth = -0.5 * np.linalg.solve(q_matrix, coeffs)
            logger.info("Optimal bandwidths (before rescaling): %s", optimal_bandwidth)
            logger.info("Optimal bandwidths (after rescaling): %s", optimal_bandwidth * self.scale_factor)

            # account for negative bandwidths due to numerical errors
            if np.any(optimal_bandwidth < 0):
                logger.warning("Negative bandwidths detected.")
                return None
            return optimal_bandwidth * self.scale_factor

    # --- Empirical plug-in helpers (used when target_distribution is None) ---
    def _default_kernel_pdf_prime(self, u):
        """Derivative of the kernel PDF with Gaussian fast-path."""
        if self.kernel_pdf_func is scipy.stats.norm.pdf:
            return -u * scipy.stats.norm.pdf(u)
        delta = 1e-5
        return (self.kernel_pdf_func(u + delta) - self.kernel_pdf_func(u - delta)) / (2 * delta)

    def _compute_pilot_bandwidths(self) -> np.ndarray:
        """Silverman pilot bandwidth per block with safeguards."""
        bws = []
        for sample in self.samples:
            n = len(sample)
            if n < 2:
                raise ValueError("Each sample needs at least two observations for bandwidth estimation.")
            std = np.std(sample, ddof=1)
            bw = self.pilot_factor * std * (n ** (-1 / 5))

            if not np.isfinite(bw) or bw <= 0:
                iqr_val = scipy.stats.iqr(sample)
                scale = iqr_val / 1.349 if iqr_val > 0 else std
                bw = self.pilot_factor * scale * (n ** (-1 / 5))

            if not np.isfinite(bw) or bw <= 0:
                logger.warning(
                    (
                        "Pilot bandwidth computation fell back to min_bandwidth=%s for a sample "
                        "with n=%d, std=%g, iqr=%g. This often indicates zero variance or "
                        "numerical issues in the data and may lead to a non-informative "
                        "bandwidth estimate."
                    ),
                    self.min_bandwidth,
                    n,
                    std,
                    iqr_val if 'iqr_val' in locals() else float('nan'),
                )
                bw = self.min_bandwidth

            bws.append(bw)
        return np.asarray(bws, dtype=float)

    def _pilot_pdf_scalar(self, y: float) -> float:
        y_val = float(y)
        total = 0.0
        for h, sample in zip(self.h_pilot, self.samples, strict=False):
            u = (y_val - sample) / h
            total += np.mean(self.kernel_pdf_func(u) / h)
        return total / self.m

    def _pilot_cdf_scalar(self, y: float) -> float:
        y_val = float(y)
        total = 0.0
        for h, sample in zip(self.h_pilot, self.samples, strict=False):
            u = (y_val - sample) / h
            total += np.mean(self.kernel_cdf_func(u))
        return total / self.m

    def _pilot_pdf_prime_scalar(self, y: float) -> float:
        y_val = float(y)
        total = 0.0
        for h, sample in zip(self.h_pilot, self.samples, strict=False):
            u = (y_val - sample) / h
            total += np.mean(self.kernel_pdf_prime_func(u) / (h ** 2))
        return total / self.m

if __name__ == "__main__":
    # set logging level to info
    rng = np.random.default_rng(seed=69)
    samples = []
    samples.append(rng.normal(loc=0, scale=5, size=100))
    samples.append(rng.normal(loc=0, scale=5, size=21))
    samples.append(rng.normal(loc=0, scale=5, size=40))

    kernel_pdf_func = scipy.stats.norm.pdf
    kernel_cdf_func = scipy.stats.norm.cdf

    # Create an instance of BandwidthCalculator
    bandwidth_calculator = BandwidthCalculator(samples, kernel_pdf_func, kernel_cdf_func)

    optimal_global_bandwidth = bandwidth_calculator.compute_optimal_global_bandwidth()
    print("Optimal global bandwidth:", optimal_global_bandwidth)

    optimal_binwise_bandwidth = bandwidth_calculator.compute_optimal_binwise_bandwidth()
    print("Optimal binwise bandwidth:", optimal_binwise_bandwidth)

    gauss_bandwidth_calculator = BandwidthCalculator(samples, kernel_pdf_func, kernel_cdf_func, target_distribution=scipy.stats.norm)
    optimal_global_bandwidth = gauss_bandwidth_calculator.compute_optimal_global_bandwidth()
    print("Optimal global bandwidth (Gaussian fit):", optimal_global_bandwidth)
    optimal_binwise_bandwidth = gauss_bandwidth_calculator.compute_optimal_binwise_bandwidth()
    print("Optimal binwise bandwidth (Gaussian fit):", optimal_binwise_bandwidth)

    samples = []
    samples.append(rng.standard_cauchy(size=30))
    samples.append(rng.standard_cauchy(size=30))
    samples.append(rng.standard_cauchy(size=30))
    samples.append(rng.standard_cauchy(size=30))
    samples.append(rng.standard_cauchy(size=21))
    samples.append(rng.standard_cauchy(size=30))

    kernel_pdf_func = scipy.stats.norm.pdf
    kernel_cdf_func = scipy.stats.norm.cdf

    # Create an instance of BandwidthCalculator
    bandwidth_calculator = BandwidthCalculator(samples, kernel_pdf_func, kernel_cdf_func, target_distribution=None)

    optimal_global_bandwidth = bandwidth_calculator.compute_optimal_global_bandwidth()
    print("Optimal global bandwidth:", optimal_global_bandwidth)

    optimal_binwise_bandwidth = bandwidth_calculator.compute_optimal_binwise_bandwidth()
    print("Optimal binwise bandwidth:", optimal_binwise_bandwidth)

    gauss_bandwidth_calculator = BandwidthCalculator(samples, kernel_pdf_func, kernel_cdf_func, target_distribution=scipy.stats.norm)
    optimal_global_bandwidth = gauss_bandwidth_calculator.compute_optimal_global_bandwidth()
    print("Optimal global bandwidth (Gaussian fit):", optimal_global_bandwidth)
    optimal_binwise_bandwidth = gauss_bandwidth_calculator.compute_optimal_binwise_bandwidth()
    print("Optimal binwise bandwidth (Gaussian fit):", optimal_binwise_bandwidth)