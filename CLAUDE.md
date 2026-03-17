# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bayesian estimation of metric weights from triplet comparisons. Given paired observations in a clean space X and noisy space Z, the code fits weights **w** on the simplex that parametrize a weighted Euclidean metric, using MCMC (NUTS) and mean-field variational inference.

Design principle: **functional programming** — pure functions, composable, no classes. JAX is naturally FP; the package aligns with that.

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests (includes coverage)
pytest tests

# Run a single test file
pytest tests/test_fit.py

# Linting and formatting
ruff check src tests
ruff format src tests

# Type checking
mypy src
```

## Architecture

### Core Modules (src/bii/)

**data.py** — Triplet formation (pure functions)
- `T_from_X()`: Convert X triplets to binary labels (anchor convention: column 0)
- `make_triplets()`: Partition pool into anchors/destinations, form triplet pairs. Returns `(T, X, Z, indices)`.

**inference.py** — Likelihood functions (all pure, JIT-compiled)
- `delta_V_one_triplet()`: Mean (delta) and variance (V) for Gaussian approximation
- `loglik_w()`: Log-likelihood given weights on simplex
- `loglik_w_per_triplet()`: Per-triplet log-likelihood (for WAIC)
- `loglik_theta()`: Likelihood in unconstrained theta-space via softmax
- Supports scalar, diagonal (p,), and full (p, p) covariance matrices

**priors.py** — Prior log-densities as composable factory functions
- `make_dirichlet_logposterior()`: Returns `logprob_fn(theta) -> scalar`
- `make_sparse_dirichlet_logposterior()`: Returns `logprob_fn(position) -> scalar`
- `sparse_dirichlet_dim()`, `sparse_dirichlet_to_simplex()`: Position vector helpers

**sampling.py** — MCMC + VI runners (prior-agnostic)
- `run_nuts()`: Multi-chain NUTS via BlackJAX
- `run_vi()`: Mean-field Gaussian via ELBO maximization
- `sample_vi()`: Draw from fitted variational posterior

**diagnostics.py** — Posterior diagnostics
- `compute_waic()`: WAIC from posterior samples
- `compute_rhat()`: Gelman-Rubin R-hat
- `compute_ess()`: Effective sample size

**fit.py** — Thin composition layer
- `fit_bii()`: Orchestrates triplets → logposterior → sampling → diagnostics

**__init__.py** — Full public API exports all building blocks

### Data Flow

```
X_pool, Z_pool → make_triplets → (T, X, Z, indices)
                                        ↓
                            make_*_logposterior(T, Z, sig, ...)
                                        ↓
                              run_nuts / run_vi → raw_samples
                                        ↓
                             softmax → w_samples → compute_waic
```

### Key Patterns

- **Weights**: Always constrained to simplex via softmax(theta)
- **Composable priors**: `make_*_logposterior` closes over data, returns pure `logprob_fn`
- **Vectorization**: Heavy use of `jax.vmap()` for broadcasting
- **Random keys**: Always split before use with `random.split()`
- **Data shapes**: (N, p) for pool points, (n, 3, p) for triplets, (n,) for labels T

### Typical Workflow

```python
from jax import random
from bii import fit_bii

key = random.PRNGKey(42)
result = fit_bii(key, X_pool, Z_pool, sig=0.1, prior="dirichlet",
                 n_triplets=500, num_samples=2000)
# result["w_samples"]: (num_samples, num_chains, p_z) on simplex
```

Or compose your own pipeline:

```python
from bii import make_triplets, make_dirichlet_logposterior, run_nuts

T, X, Z, idx = make_triplets(key, X_pool, Z_pool, n_triplets=500)
logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=ones(p))
raw_samples, acc = run_nuts(key, logprob_fn, zeros(p), num_samples=2000, ...)
```

## Key Dependencies

- **JAX**: Numerical computation and automatic differentiation
- **BlackJAX**: MCMC samplers (NUTS)
- **Optax**: Gradient-based optimization (VI)

## Project Structure

```
src/bii/           Core library
tests/             Test suite (pytest)
pyproject.toml     Package metadata
```
