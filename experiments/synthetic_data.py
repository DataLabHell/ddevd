"""Compare performance of different models on synthetic data."""

from turtle import shape
import pandas as pd
import numpy as np
from ddevd.ddevd import DDEVD
from ddevd.distributions import WeibullDistributionML
from ddevd.mev import MEV
from ddevd.evd import DistributionEVD, GEV
from ddevd.helpers import ddevd_weibull_kernel

from scipy.stats import weibull_min, norm, expon, gumbel_r, pareto, uniform
from scipy.integrate import quad

import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_synthetic_data(num_measurements, num_observations, observation_num_variance,
                            distribution, random_seed=None):
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
    if random_seed is not None:
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
    ddevd_weibull_model = DDEVD(data, target_distribution=WeibullDistributionML)
    mev_model = MEV(data)
    gev_model = GEV(data)

    return {
        "DDEVD": ddevd_model,
        "DDEVDWeibull": ddevd_weibull_model,
        "MEV": mev_model,
        "GEV": gev_model
    }

def check_h_opt(distribution, target_distribution = norm):
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

    ddevd_model = DDEVD(data, target_distribution=target_distribution)
    original_h_global = ddevd_model.h_global_estimate
    evd_model = DistributionEVD(distribution, [len(d) for d in data])
    mises = []
    h_range = np.linspace(0.1 * original_h_global, 10 * original_h_global, 100)

        # get integration bounds
    lower = min([min(d) for d in data])
    upper = max([max(d) for d in data])

    spread = upper - lower
    lower -= 0.1 * spread
    upper += 0.1 * spread

    for h_global in tqdm(h_range):
        ddevd_model.h_global_estimate = h_global
        ddevd_model.h_bin_estimates = [h_global for _ in range(ddevd_model.m)]
        mise = quad(
            lambda x: (ddevd_model.cdf(x) - evd_model.cdf(x))**2,
            lower, upper, limit=500
        )[0]
        mises.append(mise)
    return mises, h_range, original_h_global

def evaluate_stability_condition(distribution, n_range, m_range):
    """Evaluate the stability condition of the DDEVD model.

    Args:
        distribution (scipy.stats): The distribution to sample from, scipy.stats distribution object.
        n_samples (int): Number of synthetic datasets to generate.
    Returns:
        float: fraction of datasets satisfying the stability condition
    """

    from ddevd.optimal_bandwidth import BandwidthCalculator
    stable_count = 0
    result_matrix = np.zeros((len(n_range), len(m_range)), dtype=bool)
    last_failed_index = -1            
    kernel_pdf, kernel_cdf = ddevd_weibull_kernel(shape=1.5)

    for i, n in tqdm(enumerate(n_range), total=len(n_range), desc="Evaluating stability condition over n and m"):
        for j, m in enumerate(m_range):

            if j < last_failed_index-1:
                # lower values succeed, so skip
                result_matrix[i,j] = 1
                print("skip at: ", j)
                continue
            bw_calculator = BandwidthCalculator([[]], kernel_pdf, kernel_cdf, target_distribution=distribution, no_distribution_fit=True)
            bw_calculator.m = m
            bw_calculator.n_vec = np.array([n for _ in range(m)])

            D = bw_calculator.D()
            if D > 0:
                result_matrix[i,j] = 1
                print("good: ", D)
                if j == m_range[-1]:
                    last_failed_index = len(m_range)-1
            else:
                print("bad: ", D, " at n=", n, " m=", m)
                last_failed_index = j
                break


    return result_matrix

def performance_benchmark(distribution, n_samples = 100):

    mises = {
        "DDEVD": [],
        "DDEVDWeibull": [],
        "MEV": [],
        "GEV": []
    }

    for i in tqdm(range(n_samples)):
        data = generate_synthetic_data(
            num_measurements=20,
            num_observations=100,
            observation_num_variance=0,
            distribution=distribution
        )

        results = fit_and_evaluate_models(data)

        evd_model = DistributionEVD(distribution, [len(d) for d in data])

        # get integration bounds
        lower = min([min(d) for d in data])
        upper = max([max(d) for d in data])

        spread = upper - lower
        lower -= 0.1 * spread
        upper += 0.1 * spread

        for model_name, model in results.items():
            mise = quad(
                lambda x: (model.cdf(x) - evd_model.cdf(x))**2,
                lower, upper, limit=500,
            )[0]
            mises[model_name].append(mise)
    return mises

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DDEVD models")
    parser.add_argument("--experiment", type=str, choices=["stability", "bandwidth", "benchmark", "nvariance"], required=True,
                        help="The experiment to run: 'stability', 'nvariance', 'bandwidth', or 'benchmark'.")
    args = parser.parse_args()

    used_distributions = {
        "Weibull0.6": weibull_min(c=0.6, scale=50),
        "Weibull1.5": weibull_min(c=1.5, scale=50),
        "Weibull2.5": weibull_min(c=2.5, scale=50),
        "Uniform": uniform(loc=30, scale=20),
        "Pareto": pareto(b=30, scale=10),
        "Exponential": expon(scale=30),
        "Gumbel": gumbel_r(loc=30, scale=10),
        "Normal": norm(loc=30, scale=10),
    }

    if args.experiment == "stability":
        n_min = 100
        n_max = 10000
        m_min = 100
        m_max = 10000

        n_range = range(n_min, n_max, 50)
        m_range = range(m_min, m_max, 10)
        for dist_name, dist in used_distributions.items():
            print(f"Evaluating stability condition for distribution: {dist_name}")
            result_matrix = evaluate_stability_condition(dist, n_range, m_range)
            with open(f'stability_condition_{dist_name}.npy', 'wb') as f:
                np.save(f, result_matrix)
            plt.figure()
            plt.imshow(result_matrix.transpose(), extent=[n_min, n_max, m_min, m_max], aspect='auto', origin='lower', cmap='Greys')
            plt.colorbar(label='Stability Condition Satisfied')
            plt.ylabel('Number of Measurements (m)')
            plt.xlabel('Number of Observations per Measurement (n)')
            plt.title(f'DDEVD Stability Condition Evaluation ({dist_name} Distribution)')
            plt.savefig(f'experiments/results/stability_condition_heatmap_{dist_name}.png', dpi=300)

    elif args.experiment == "nvariance":
        observation_variances = [0, 5, 10, 20, 50]
        m_range = [20, 50, 100, 200]
        result_matrix = np.zeros((len(observation_variances), len(m_range)))

        for dist_name, dist in used_distributions.items():
            for i, obs_var in tqdm(enumerate(observation_variances), desc=f"Evaluating observation variance effect for {dist_name} distribution"):
                for j, m in enumerate(m_range):
                    data = generate_synthetic_data(
                        num_measurements=m,
                        num_observations=300,
                        observation_num_variance=obs_var,
                        distribution=dist,
                        random_seed=42
                    )
                    try:
                        results = fit_and_evaluate_models(data)
                        evd_model = DistributionEVD(dist, [len(d) for d in data])
                        # compute the MISE for DDEVD model
                        ddevd_model = results["DDEVD"]
                        mise = quad(
                            lambda x: (ddevd_model.cdf(x) - evd_model.cdf(x))**2,
                            0, 200
                        )[0]
                    except ValueError:
                        print("Weibull fit failed for observation variance: ", obs_var, " in distribution: ", dist_name)
                        mise = np.nan
                    result_matrix[i,j] = mise

            plt.figure()
            plt.imshow(result_matrix, extent=[min(m_range), max(m_range), min(observation_variances), max(observation_variances)], aspect='auto', origin='lower', cmap='viridis')
            plt.xlabel("Observation Number Variance")
            plt.ylabel("Number of Measurements (m)")
            plt.title(f"Effect of Observation Number Variance on DDEVD Performance, {dist_name} Distribution")
            plt.colorbar(label="MISE")
            plt.savefig(f"observation_variance_effect_{dist_name}.png", dpi=300)

    elif args.experiment == "bandwidth":
        results_table = []
        target_distributions = {"Normal": norm, "Weibull": weibull_min}
        
        for targ_name, targ_dist in target_distributions.items():
            for idx, (dist_name, dist) in enumerate(used_distributions.items()):
                print(f"Processing bandwidth optimization for {dist_name}...")
                h_mises, h_range, original_h_global = check_h_opt(dist, target_distribution=targ_dist)
                
                # Find optimal
                min_idx = np.argmin(h_mises)
                h_opt = h_range[min_idx]
                mise_opt = h_mises[min_idx]
                
                # Original MISE
                original_idx = np.argmin(np.abs(h_range - original_h_global))
                mise_original = h_mises[original_idx]
                
                # Calculate errors
                abs_error = original_h_global - h_opt
                rel_error = (original_h_global - h_opt) / h_opt * 100
                mise_ratio = mise_original / mise_opt
                
                results_table.append({
                    'TargetDistribution': targ_name,
                    'Distribution': dist_name,
                    'h_optimal': h_opt,
                    'h_algorithm': original_h_global,
                    'abs_error': abs_error,
                    'rel_error_pct': rel_error,
                    'MISE_optimal': mise_opt,
                    'MISE_algorithm': mise_original,
                    'MISE_ratio': mise_ratio
                })
                    
                bandwidth_data = pd.DataFrame({
                    'h_normalized': h_range / h_opt,
                    'mise_normalized': np.array(h_mises) / mise_opt,
                    'h_absolute': h_range,
                    'mise_absolute': h_mises
                })
                bandwidth_data.to_csv(f"experiments/results/bandwidth_{dist_name}_{targ_name}.csv", index=False)
        
        results_df = pd.DataFrame(results_table)
        
        for target in ['Normal', 'Weibull']:
            target_data = results_df[results_df['TargetDistribution'] == target]
            target_data.to_csv(f"experiments/results/bandwidth_{target}_target.csv", index=False)
        results_df.to_csv("experiments/results/bandwidth_optimization_results.csv", index=False)
        
        print("\n" + "="*80)
        print("BANDWIDTH OPTIMIZATION SUMMARY")
        print("="*80)
        print(f"\n{'Target':<15} {'Distribution':<15} {'h_opt':>10} {'h_alg':>10} {'Abs Err':>10} {'Rel Err %':>10} {'MISE Ratio':>12}")
        print("-"*80)
        for _, row in results_df.iterrows():
            print(f"{row['TargetDistribution']:<15} {row['Distribution']:<15} {row['h_optimal']:>10.4f} {row['h_algorithm']:>10.4f} "
                  f"{row['abs_error']:>+10.4f} {row['rel_error_pct']:>+10.1f} {row['MISE_ratio']:>12.2f}")
        print("-"*80)
        print(f"{'Mean':<15} {'':<10} {'':<10} "
              f"{results_df['abs_error'].mean():>+10.4f} {results_df['rel_error_pct'].mean():>+10.1f} "
              f"{results_df['MISE_ratio'].mean():>12.2f}")
        print(f"{'Std Dev':<15} {'':<10} {'':<10} "
              f"{results_df['abs_error'].std():>10.4f} {results_df['rel_error_pct'].std():>10.1f} "
              f"{results_df['MISE_ratio'].std():>12.2f}")
        print("="*80)

    elif args.experiment == "benchmark":
        benchmark_results = pd.DataFrame(columns=["Distribution", "Model", "MISE"])
        
        # Summary statistics for TikZ
        summary_stats = []
        
        for dist_name, dist in used_distributions.items():
            print(dist_name)           
            mises = performance_benchmark(dist, n_samples=50)
            print(f"Benchmarking for distribution: {dist_name}")
            
            for model_name, mise_values in mises.items():
                benchmark_results = pd.concat([benchmark_results, pd.DataFrame({
                    "Distribution": [dist_name]*len(mise_values),
                    "Model": [model_name]*len(mise_values),
                    "MISE": mise_values
                })], ignore_index=True)
                
                # Compute statistics for TikZ
                avg_mise = np.mean(mise_values)
                std_mise = np.std(mise_values)
                median_mise = np.median(mise_values)
                q25 = np.percentile(mise_values, 25)
                q75 = np.percentile(mise_values, 75)
                
                summary_stats.append({
                    'Distribution': dist_name,
                    'Model': model_name,
                    'Mean': avg_mise,
                    'Std': std_mise,
                    'Median': median_mise,
                    'Q25': q25,
                    'Q75': q75,
                    'Min': np.min(mise_values),
                    'Max': np.max(mise_values)
                })
                
                print(f"  Model: {model_name}, Average MISE: {avg_mise:.6f} ± {std_mise:.6f}")

            # Matplotlib boxplot
            plt.figure()
            plt.boxplot([mises[model_name] for model_name in mises.keys()], labels=mises.keys())
            plt.ylabel("MISE")
            plt.title(f"Model Performance Comparison for {dist_name} Distribution")
            plt.savefig(f"experiments/results/model_performance_comparison_{dist_name}.png", dpi=300)
            plt.close()
        
        # Save full results
        benchmark_results.to_csv("experiments/results/benchmark_results.csv", index=False)
        
        # Save summary for TikZ
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv("experiments/results/benchmark_summary.csv", index=False)
        
        # Generate separate files for each model (cleaner for TikZ)
        for model in ['DDEVD', 'MEV', 'GEV', 'DDEVDWeibull']:
            model_data = summary_df[summary_df['Model'] == model][['Distribution', 'Mean', 'Std']]
            model_data.to_csv(f"experiments/results/benchmark_{model}.csv", index=False)