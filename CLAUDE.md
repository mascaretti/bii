# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian estimation of metric weights from triplet comparisons. Given noisy observations, the code fits weights **w** on the simplex that parametrize a weighted Euclidean metric, using both maximum likelihood (MLE) and full Bayesian inference (MCMC).

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests (includes coverage)
pytest tests

# Run a single test file
pytest tests/test_inference.py

# Run a single test function
pytest tests/test_inference.py::test_function_name -v

# Linting and formatting
ruff check src tests
ruff format src tests

# Type checking
mypy src
```

## Architecture

### Core Modules (src/bii/)

**data.py** - Data generation
- `make_iid()`: Generate IID triplet observations with fresh source/target points per triplet
- `make_data()`: Extract disjoint triplets from a single dataset
- `T_from_X()`: Convert distances to binary comparisons

**inference.py** - Inference engine
- `loglik_theta()`: Probit likelihood with softmax-constrained weights
- `log_posterior()`: Full posterior (likelihood + Dirichlet prior)
- `fit()`: MLE via Adam optimizer, returns weights on simplex

**mcmc.py** - Bayesian posterior sampling
- `sample_posterior_nuts()`: Multi-chain NUTS sampling via BlackJAX
- `compute_posterior_statistics()`: Mean, std, quantiles, MAP
- `compute_rhat()`, `compute_ess()`: Convergence diagnostics

**triplets.py** - Triplet likelihood
- `delta_V_one_triplet()`: Computes mean (delta) and variance (V) for Gaussian approximation
- Core formula: P(i closer to k than j) = Φ(-delta/√V)

**radial.py** - Geometric utilities for shell-based binning and DAG construction

**voronoi.py** - Voronoi tessellation tools

**utils.py** - Plotting functions for posteriors, traces, diagnostics

### Data Flow

```
Raw X → Add noise → Observed Z → Extract triplets → Binary T → Fit weights w
```

### Key Patterns

- **Weights**: Always constrained to simplex via softmax(theta)
- **Vectorization**: Heavy use of `jax.vmap()` for broadcasting
- **Random keys**: Always split before use with `random.split()`
- **Data shapes**: (n, p) for points, (n, 3, p) for triplets

### Typical Workflow

```python
from jax import random
from bii.data import make_iid
from bii.inference import fit
from bii.mcmc import sample_posterior_nuts, compute_posterior_statistics

key = random.PRNGKey(42)
X, Z = make_iid(key, n_triplets=1000, p=5, sig=0.1, tau=1.0, w_star=w_true)
w_hat, theta_hist, loss_hist = fit(key, X, Z, sig=0.1, steps=5000)

# For full posterior
samples = sample_posterior_nuts(key, X, Z, sig, alpha=jnp.ones(p), num_samples=2000)
stats = compute_posterior_statistics(samples['w_samples'])
```

## Key Dependencies

- **JAX**: Numerical computation and automatic differentiation
- **BlackJAX**: MCMC samplers (NUTS)
- **Optax**: Gradient-based optimization
