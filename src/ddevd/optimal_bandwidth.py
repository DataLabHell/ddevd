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
                 target_distribution=scipy.stats.norm,
                 use_scaling=False):
        """Initialize the BandwidthCalculator with samples and kernel functions."""
        self.samples = samples
        self.m = len(samples)
        self.n_vec = np.array([len(sample) for sample in self.samples])
        self.kernel_pdf_func = kernel_pdf_func
        self.kernel_cdf_func = kernel_cdf_func
        self.optimization_position = optimization_position
        self.target_distribution = target_distribution
        self.use_scaling = use_scaling
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

        pooled_data = [x for bin_data in samples for x in bin_data]
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
            distribution_fit = target_distribution.fit(scaled_data)
            logger.info("Distribution parameters: %s", distribution_fit)
            @functools.lru_cache(maxsize=None)
            def pdf_est(y):
                return target_distribution.pdf(y, *distribution_fit)

            @functools.lru_cache(maxsize=None)
            def cdf_est(y):
                return target_distribution.cdf(y, *distribution_fit)

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
        """Computes the function E_1,i."""
        if n_i <= 2:
            raise ValueError("n_i cannot be less than or equal to 2.")
        return -n * self.fy_1(y, n - 1, n_i) * self.pdf(y) * self.mu_k_1 + self.fy_1(y, n, n_i)

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

    ### Coefficient computation
    def c(self):
        """Compute the coefficient vector."""
        n_vec = np.array([len(sample) for sample in self.samples])
        logger.info("Computing coefficients for n_vec: %s", n_vec)
        # Compute the coefficients
        def sum_b_0(y, n_vec):
            return np.sum([self.b_0_i(y, n_i) for n_i in n_vec])

        # c_i(y) = 2* sum_b_0 * b_1_i + V_1_i
        # compute the integral of c_i to get the coefficient
        def c_i(y, n_vec, i):
            return 2 * sum_b_0(y, n_vec) * self.b_1_i(y, n_vec[i]) + self.V_1_i(y, n_vec[i])

        if self.optimization_position == "global":
            return [
                scipy.integrate.quad(
                    c_i,
                    scipy.optimize.brentq(
                        lambda y, i=i: np.exp(self.cdf(y) - 1e-6) - 1,
                        -100,
                        10000,
                        xtol=1,
                    ),
                    scipy.optimize.brentq(
                        lambda y, i=i: np.exp(self.cdf(y) - 1 + 1e-6) - 1,
                        -100,
                        10000,
                        xtol=1,
                    ),
                    args=(
                        n_vec,
                        i,
                    ),
                )[0]
                for i in range(len(n_vec))
            ]
        # if optimization position is a number use it as argument of c_i
        if isinstance(self.optimization_position, str) and self.optimization_position.startswith("quantile_"):
            q = float(self.optimization_position.split("_")[1])
            return [
                scipy.integrate.quad(
                    c_i,
                    scipy.optimize.brentq(
                        lambda y, q=q: np.exp(self.cdf(y) - q) - 1,
                        -1000,
                        100000,
                        xtol=0.1,
                    ),
                    scipy.optimize.brentq(
                        lambda y, i=i: np.exp(self.cdf(y) - 1 + 1e-6) - 1,
                        -1000,
                        100000,
                        xtol=1,
                    ),
                    args=(
                        n_vec,
                        i,
                    ),
                )[0]
                for i in range(len(n_vec))
            ]
        if isinstance(self.optimization_position, (int, float)):
            return [
                c_i(
                    self.optimization_position,
                    n_vec,
                    i,
                )
                for i in range(len(n_vec))
            ]
        raise ValueError(
            "Optimization position must be 'global', 'quantile_x' or a number. Got: {}".format(self.optimization_position)
        )

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

        for k in range(len(n_vec)):
            for r in range(len(n_vec)):
                if self.optimization_position == "global":
                    lower_limit = scipy.optimize.brentq(
                        lambda y, k=k: np.exp(self.cdf(y) - 1e-6) - 1,
                        -10000,
                        100000,
                        xtol=1e-6,
                    )
                    upper_limit = scipy.optimize.brentq(
                        lambda y, k=k: np.exp(self.cdf(y) - 1 + 1e-6) -1,
                        -10000,
                        100000,
                        xtol=1e-6,
                    )
                    Q[k, r] = scipy.integrate.quad(
                        q_fun,
                        lower_limit,
                        upper_limit,
                        args=(k, r),
                    )[0]
                # if optimization position is a number use it as argument of q_fun
                elif isinstance(self.optimization_position, (int, float)):
                    Q[k, r] = q_fun(self.optimization_position, k, r)
                elif isinstance(self.optimization_position, str) and self.optimization_position.startswith("quantile_"):
                    q = float(self.optimization_position.split("_")[1])
                    lower_limit = scipy.optimize.brentq(
                        lambda y, q=q: np.exp(self.cdf(y) - q) - 1,
                        -10000,
                        100000,
                        xtol=1e-6,
                    )
                    upper_limit = scipy.optimize.brentq(
                        lambda y, k=k: np.exp(self.cdf(y) - 1 + 1e-6) - 1,
                        -10000,
                        100000,
                        xtol=1e-6,
                    )
                    Q[k, r] = scipy.integrate.quad(
                        q_fun,
                        lower_limit,
                        upper_limit,
                        args=(k, r),
                    )[0]
                else:
                    raise ValueError(
                        "Optimization position must be 'global' or a number. Got: {}".format(self.optimization_position)
                    )
        logger.info("Q matrix computed: %s", Q)
        return Q

    def compute_optimal_global_bandwidth(self):
        """Compute the optimal bandwidth for the given samples."""
        lin = np.sum(self.c())
        qua = np.sum(self.Q())
        bandwidth = -0.5 * lin / qua
        logger.info("Optimal global bandwidth (before rescaling): %s", bandwidth)
        logger.info("Optimal global bandwidth (after rescaling): %s", bandwidth * self.scale_factor)
        if bandwidth < 0:
            logger.warning("Negative global bandwidth detected.")
            bandwidth = None
        return bandwidth * self.scale_factor

    def compute_optimal_binwise_bandwidth(self):
        """Compute the optimal bandwidth for the given samples."""
        coeffs = self.c()
        q_matrix = self.Q()
        logger.info("Coefficients: %s", coeffs)
        logger.info("Q matrix: %s", q_matrix)
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
        return optimal_bandwidth * self.scale_factor

if __name__ == "__main__":
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