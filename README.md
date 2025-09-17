<div align="center">

# ddevd

Data Driven Extreme Value Distribution (DDEVD) & related tools for modern extreme value analysis in Python.

<p>
	<strong>ddevd</strong> provides a non‑parametric / semi‑parametric framework to estimate extreme value (return level) behavior directly from collections of finite replicates (blocks / seasons / years / ensembles) without committing to a single parametric tail model per block.
</p>

</div>

## Key Features

- Data Driven Extreme Value Distribution (DDEVD) implementation (`DDEVD`)
- Metastatistical Extreme Value Distribution (MEV) baseline (`MEV`)
- Direct EVD from known underlying distribution (`DistributionEVD`)
- Automatic optimal kernel bandwidth selection (global or per group) via higher‑order MISE / pointwise criteria
- Flexible optimization target: global MISE, quantile‑restricted integration (`quantile_q`), or evaluation at a fixed value
- Custom kernel support (supply PDF & CDF of kernel)
- Built‑in normal kernel + optional data‑driven Weibull kernel helper
- Bootstrap return level uncertainty estimation
- Lightweight dependency footprint (NumPy / SciPy / tqdm)

## Installation

The project uses the [uv](https://github.com/astral-sh/uv) Python package manager (fast lock & resolver). Ensure you have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # or see uv docs for alternatives
```

When the project is published to PyPI you will be able to do:

```bash
uv add ddevd
```

For local development / cloning:

```bash
git clone https://github.com/DataLabHell/ddevd.git
cd ddevd
uv sync   # creates / updates virtualenv and installs dependencies
```

To enter an environment shell:

```bash
uv shell
```

To run a module or script without activating the shell explicitly:

```bash
uv run python -c "import ddevd; print(ddevd.__version__)"
```

## Quick Start

```python
import numpy as np
from ddevd.ddevd import DDEVD

# Suppose you have m groups (e.g. 30 years) of intra‑year observations (e.g. daily rainfall maxima per storm)
rng = np.random.default_rng(42)
data = [rng.weibull(a=2.2, size=rng.integers(25, 60)) * 10 for _ in range(30)]

dist = DDEVD(data, h_opt_position="global")

# Evaluate CDF at a grid
import numpy as np
y = np.linspace(0, 80, 200)
F_y = dist.cdf(y)          # binwise (default)
F_y_global = dist.cdf(y, mode="global")

# Return levels (e.g. 2, 5, 10, 50 year)
rl = dist.return_levels([2, 5, 10, 50])
print("Return levels:", rl)

# Bootstrap uncertainty
samples = dist.bootstrap_return_levels([10, 50], n_resample=200)
mean_rl_10, mean_rl_50 = samples.mean(axis=0)
print(mean_rl_10, mean_rl_50)
```

## Conceptual Overview

Classical block maxima methods fit a Generalized Extreme Value (GEV) distribution to maxima aggregated from each block (e.g. yearly maximum). The Metastatistical EVD (MEV) instead integrates the distribution of within‑block observations raised to the block size. The **DDEVD** generalizes this idea in a data‑driven way: it estimates the *within‑block* (base) distribution non‑parametrically using a kernel estimator with bandwidths tailored to minimize asymptotic mean squared error of the final extreme value CDF estimator.

Given groups (blocks) of observations \( X_{i1}, …, X_{i n_i} \) for \( i=1,…,m \), the block‑specific base CDF \(F_i\) is estimated via a kernel smoothed empirical CDF with bandwidth \(h_i\). The extreme value CDF for each block is \(F_i(x)^{n_i}\). The DDEVD aggregates these to

$$ F_{\text{DDEVD}}(x) = \frac{1}{m} \sum_{i=1}^m F_i(x)^{n_i}. $$

Bandwidths \(h_i\) (or a single global \(h\)) are chosen by minimizing (depending on user choice):

- Global integrated MISE ("global")
- Integrated error above a quantile threshold ("quantile_q")
- Pointwise MSE at a specific value \(x_0\)

The implementation derives higher‑order expansions of bias/variance terms yielding a quadratic form: minimize \( (1/2) h^T Q h + c^T h \), leading to closed‑form optimal bandwidth(s) (solving linear system or scalar form). Negative or numerically unstable solutions are detected and handled.

## Bandwidth Selection Details

Implemented in `BandwidthCalculator`:

- Computes kernel moment integrals (first and second raw moments of kernel and of the symmetrized product \(2 K(u) K_c(u)\))
- Approximates derivatives (finite differences) of fitted target distribution PDF
- Provides functions: `c()` (linear terms) and `Q()` (quadratic matrix)
- Solves for optimal binwise vector \(h = -0.5 Q^{-1} c\) or scalar global \(-0.5 (sum c) / (sum Q)`
- Supports scaling (95th percentile normalization) to improve numerical stability

Optimization position argument (`h_opt_position`):

- `"global"` – integrate whole support
- `"quantile_0.95"` – integrate from 95th percentile upwards
- numeric (float/int) – evaluate objective at a fixed value

## Bootstrap Return Levels

`DDEVD.bootstrap_return_levels(return_periods, n_resample)` resamples blocks with replacement, recomputes return levels, and returns an array of shape `(n_resample, len(return_periods))` for empirical confidence intervals.

Example for 95% CI:

```python
rl_samples = dist.bootstrap_return_levels([10], n_resample=500)
rl10 = rl_samples[:,0]
ci = np.quantile(rl10, [0.025, 0.975])
print("10‑year RL 95% CI:", ci)
```

## API Overview

### `DDEVD`
Constructor:
```python
DDEVD(data: list[list[float]],
			h_opt_position: str | int | float = "global",
			kernel_functions: tuple[Callable, Callable] | None = None,
			target_distribution = scipy.stats.norm,
			max_bins: int = 100,
			use_scaling: bool = False)
```
Methods:
- `cdf(y, mode="binwise"|"global", alternative_data=None)` – Vectorized CDF
- `return_levels(return_periods, mode=..., alternative_data=None)` – Inherited from base
- `quantile(q, mode=...)` – Numerical inversion of CDF
- `bootstrap_return_levels(return_periods, n_resample, mode)`

### `MEV`
Fits Weibull per block via `WeibullDistributionML.fit` then forms \( (1/m) \sum F_{W,i}(x)^{n_i} \).

### `DistributionEVD`
Analytical extreme value distribution when the underlying (base) distribution is known; useful for simulation studies.

### `BandwidthCalculator`
Internal class; expose only if performing research / diagnostics. Key public methods: `compute_optimal_global_bandwidth()`, `compute_optimal_binwise_bandwidth()`.

### Helper Functions
- `ddevd.helpers.get_empirical_return_periods(values)` – empirical plotting positions
- `ddevd.helpers.ddevd_weibull_kernel(data)` – returns a Weibull‑based kernel (PDF, CDF) centered via mean of fitted shape.

## Choosing Kernel & Bandwidth

Default kernel: standard normal PDF/CDF. For positively skewed hydrometeorological variables, a Weibull kernel may reduce boundary bias:

```python
from ddevd.helpers import ddevd_weibull_kernel
dist = DDEVD(data, kernel_functions=ddevd_weibull_kernel(data))
```

If binwise optimization fails (negative bandwidths) the implementation falls back to global bandwidth.

## Return Levels & Periods

Return period T (years) corresponds to non‑exceedance probability \( p = 1 - 1/T \). We estimate the quantile \( x_T = F^{-1}(p) \) using numerical bisection (robust to non‑closed form CDF).

## Limitations & Assumptions

- Independence within and across blocks assumed (no declustering performed internally)
- Large‑n approximations in bandwidth derivations: very small block sizes (< ~10) may degrade accuracy
- Kernel estimation currently univariate only
- No extrapolation beyond observed range except through kernel support

## Development

Project uses a simple `pyproject.toml` (PEP 621). Typical uv‑based workflow:

```bash
uv sync                 # install / update deps
uv run pytest -q        # run tests (if/when tests are added)
uv run python examples/demo.py  # run example (if created)
```

Add a new dependency (runtime):

```bash
uv add somepackage
```

Add a dev dependency (once you define optional groups):

```bash
uv add --dev pytest
```

Code style: keep dependencies minimal; logging is already instrumented for diagnostics.

### Suggested Future Enhancements
- Confidence bands via asymptotic variance formulas (not just bootstrap)
- Support for covariates / non‑stationarity (conditional kernels)
- Multivariate or spatial extension
- Automatic block size diagnostics
- Vectorized PDF derivative (avoid finite differences)

## Citation

If you use this package in academic work, please cite the repository (add DOI once archived, e.g. Zenodo). A BibTeX template:

```bibtex
@misc{ddevd2025,
	title  = {ddevd: Data Driven Extreme Value Distribution in Python},
	author = {Michael Sandbichler},
	year   = {2025},
	url    = {https://github.com/DataLabHell/ddevd}
}
```

## License

This project is licensed under the terms of the Apache 2.0 License – see `LICENSE`.

## Support & Questions

Open an issue or discussion on GitHub with reproducible snippets. PRs welcome.

---
Made with focus on transparent, data-driven extremes modelling.

