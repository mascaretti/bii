# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian estimation of metric weights from triplet comparisons. Given paired observations in a clean space X and noisy space Z, the code fits weights **w** on the simplex that parametrize a weighted Euclidean metric, using MCMC (NUTS) and mean-field variational inference.

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests (includes coverage)
pytest tests

# Run a single test file
pytest tests/test_fit.py

# Run a single test function
pytest tests/test_fit.py::test_fit_smoke -v

# Linting and formatting
ruff check src tests
ruff format src tests

# Type checking
mypy src

# Run an experiment (example)
python experiments/exp_recovery.py

# Build paper
cd reports/paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Architecture

### Core Modules (src/bii/)

**fit.py** - Unified pipeline (main entry point)
- `fit_bii()`: End-to-end: triplet formation → NUTS/VI → posterior summary + WAIC
- `_random_triplets()`: Partition pool into anchors/destinations, form triplet pairs
- `_compute_waic()`: WAIC from posterior samples
- `_rhat()`, `_ess()`: Convergence diagnostics

**inference.py** - Likelihood functions
- `delta_V_one_triplet()`: Computes mean (delta) and variance (V) for Gaussian approximation
- `loglik_w()`: Log-likelihood given weights on simplex
- `loglik_w_per_triplet()`: Per-triplet log-likelihood (for WAIC)
- `loglik_theta()`: Likelihood in unconstrained theta-space via softmax
- `fit()`: Legacy MLE via Adam (use `fit_bii` instead)
- `sample_posterior_nuts()`: Legacy NUTS (use `fit_bii` instead)

**horseshoe.py** - Horseshoe prior (Makalic & Schmidt 2015)
- `log_horseshoe_posterior()`: Full log-posterior with InvGamma auxiliaries
- `horseshoe_to_simplex()`: Extract w from packed position vector
- Position vector: phi (p), log_lam_sq (p), log_nu (p), log_tau_sq (1), log_xi (1) = 3p+2

**vi.py** - Mean-field variational inference
- `run_vi()`: Fit mean-field Gaussian via ELBO maximization
- `sample_vi()`: Draw samples from fitted variational posterior

**data.py** - Data generation
- `T_from_X()`: Convert X triplets to binary labels (anchor convention: column 0)
- `make_iid()`: Generate IID triplet observations
- `make_data()`: Extract disjoint triplets from a single dataset

**__init__.py** - Public API: exports `fit_bii` and `T_from_X`

### Data Flow

```
X_pool, Z_pool → partition anchors/destinations → form triplets → labels T from X → NUTS/VI on w → posterior samples
```

### Key Patterns

- **Weights**: Always constrained to simplex via softmax(theta)
- **Vectorization**: Heavy use of `jax.vmap()` for broadcasting
- **Random keys**: Always split before use with `random.split()`
- **Data shapes**: (N, p) for pool points, (n, 3, p) for triplets, (n,) for labels T

### Typical Workflow

```python
import jax.numpy as jnp
from jax import random
from bii import fit_bii

key = random.PRNGKey(42)
# X_pool: (N, p_x) clean reference, Z_pool: (N, p_z) noisy representation
result = fit_bii(key, X_pool, Z_pool, sig=0.1, prior="dirichlet",
                 n_triplets=500, num_samples=2000)
# result["w_samples"]: (num_samples, num_chains, p_z) on simplex
```

## Key Dependencies

- **JAX**: Numerical computation and automatic differentiation
- **BlackJAX**: MCMC samplers (NUTS)
- **Optax**: Gradient-based optimization (VI + legacy MLE)

## Project Structure

```
src/bii/           Core library
tests/             Test suite (pytest)
experiments/       Experiment scripts for paper (exp_*.py)
reports/paper/     LaTeX paper (jmlr2e style)
reports/theory/    Theory notes
reports/HCP/       HCP data documentation
data/interim/      HCP-YA CSV (not tracked)
```
