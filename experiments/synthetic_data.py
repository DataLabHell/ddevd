"""Compare performance of different models on synthetic data."""

from turtle import shape
import numpy as np
from ddevd.ddevd import DDEVD
from ddevd.distributions import WeibullDistributionML
from ddevd.mev import MEV
from ddevd.evd import DistributionEVD, GEV

from scipy.stats import weibull_min, norm, expon, gumbel_r, pareto
from scipy.integrate import quad

def generate_synthetic_data(num_measurements, num_observations, observation_num_variance,
                            distribution, random_seed=42):
    """Generate synthetic data from a specified distribution.

    Args:
        num_measurements (int): Number of measurements (samples).
        num_observations (int): Number of observations per measurement.
        observation_num_variance (float): Variance of the number of observations.
        distribution (scipy.stats): The distribution to sample from, scipy.stats distribution object.
        random_seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Synthetic data generated from the specified distribution.
    """
    np.random.seed(random_seed)
    data = []
    for _ in range(num_measurements):
        try:
            if observation_num_variance == 0:
                n_obs = num_observations
            else:
                n_obs = max(1, int(np.random.normal(num_observations, observation_num_variance)))
            sample = distribution.rvs(size=n_obs)
            data.append(list(sample))
        except ValueError:
            raise ValueError("Unsupported distribution type.")
    return data

def fit_and_evaluate_models(data):
    """Fit DDEVD, MEV and GEV models to the data and evaluate return levels.

    Args:
        data (list[list[float]]): Synthetic data to fit the models.
        return_periods (list[float]): Return periods for evaluation.
    Returns:
        dict: all models
    """
    ddevd_model = DDEVD(data)
    mev_model = MEV(data)
    gev_model = GEV(data)

    return {
        "DDEVD": ddevd_model,
        "MEV": mev_model,
        "GEV": gev_model
    }

def check_h_opt(distribution):
    """Check if the DDEVD bandwidth optimization works correctly.

    Args:
        data (list[list[float]]): Synthetic data to fit the DDEVD model.
    Returns:
        float: optimized global bandwidth
    """
    data = generate_synthetic_data(
            num_measurements=20,
            num_observations=100,
            observation_num_variance=0,
            distribution=distribution,
            random_seed=42
        )

    ddevd_model = DDEVD(data)
    original_h_global = ddevd_model.h_global_estimate
    evd_model = DistributionEVD(distribution, [len(d) for d in data])
    mises = []
    h_range = np.linspace(0.1 * original_h_global, 10 * original_h_global, 100)
    for h_global in h_range:
        ddevd_model.h_global_estimate = h_global
        ddevd_model.h_bin_estimates = [h_global for _ in range(ddevd_model.m)]
        mise = quad(
            lambda x: (ddevd_model.cdf(x) - evd_model.cdf(x))**2,
            -1000, 1000
        )[0]
        mises.append(mise)
    return mises, h_range, original_h_global


def performance_benchmark(distribution, n_samples = 100):

    mises = {
        "DDEVD": [],
        "MEV": [],
        "GEV": []
    }
    for i in range(n_samples):
        data = generate_synthetic_data(
            num_measurements=20,
            num_observations=100,
            observation_num_variance=0,
            distribution=distribution,
            random_seed=42 + i
        )

        results = fit_and_evaluate_models(data)

        evd_model = DistributionEVD(distribution, [len(d) for d in data])

        # compute the MISE for each model
        for model_name, model in results.items():
            mise = quad(
                lambda x: (model.cdf(x) - evd_model.cdf(x))**2,
                -1000, 1000
            )[0]
            mises[model_name].append(mise)
    return mises

if __name__ == "__main__":

    used_distributions = {
        "Gumbel": gumbel_r(loc=5, scale=10),
        "Normal": norm(loc=5, scale=10),
        "Pareto": pareto(b=5, scale=10)
    }

    for dist_name, dist in used_distributions.items():
        h_mises, h_range, original_h_global = check_h_opt(dist)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(h_range, h_mises)
        plt.axvline(original_h_global, color='red', linestyle='--', label='Original h_global')
        plt.xlabel("Bandwidth h")
        plt.ylabel("MISE")
        plt.title(f"Bandwidth Optimization for {dist_name} Distribution")
        plt.show()

        mises = performance_benchmark(dist, n_samples=10)
        print(f"Benchmarking for distribution: {dist_name}")
        for model_name, mise_values in mises.items():
            avg_mise = np.mean(mise_values)
            print(f"  Model: {model_name}, Average MISE: {avg_mise:.6f}")

        #mise boxplot
        plt.figure()
        plt.boxplot([mises[model_name] for model_name in mises.keys()], labels=mises.keys())
        plt.ylabel("MISE")
        plt.title(f"Model Performance Comparison for {dist_name} Distribution")
        plt.show()