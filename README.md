# bii — Bayesian Information Imbalance

Bayesian estimation of metric weights from triplet comparisons using MCMC (NUTS) and mean-field variational inference.

[![Documentation Status](https://readthedocs.org/projects/bii/badge/?version=latest)](https://bii.readthedocs.io/en/latest/?badge=latest)

## Documentation

Full documentation (guide + API reference) lives in `docs/` and is published with
[Read the Docs](https://readthedocs.org). Build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Installation

```bash
pip install git+https://github.com/mascaretti/bii.git
```

For development:

```bash
git clone https://github.com/mascaretti/bii.git
cd bii
pip install -e ".[dev]"
```

## Quick start

```python
from jax import random
from bii import fit_bii

key = random.PRNGKey(42)
result = fit_bii(key, X_pool, Z_pool, sig=0.1, prior="dirichlet",
                 n_triplets=500, num_samples=2000)
# result["w_samples"]: (num_samples, num_chains, p) on simplex
```

## Composable API

```python
from bii import make_triplets, make_dirichlet_logposterior, run_nuts

T, X, Z, idx = make_triplets(key, X_pool, Z_pool, n_triplets=500)
logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=ones(p))
raw_samples, acc = run_nuts(key, logprob_fn, zeros(p),
                            num_samples=2000, num_warmup=1000,
                            num_chains=4)
```
