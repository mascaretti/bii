# bii: Bayesian information imbalance

Weights `w` on the simplex for a weighted Euclidean metric on a noisy source `Z`,
learned from triplets: which of two candidates is closer to an anchor in a trusted
target `X`. Each label is a probit of the source's standardised gap, the weights carry
a Dirichlet prior, the likelihood is tempered by `kappa` for overlapping triplets, and
a calibration slope can be integrated out on a grid. Inference is NUTS (BlackJAX) or
the posterior mode.

## Install

```bash
pip install "git+https://github.com/mascaretti/bii.git@release"
```

## Use

```python
from jax import random
from bii import fit_bii, make_slope_grid

res = fit_bii(random.PRNGKey(0), X_pool, Z_pool, sig=S_pool, n_triplets=15,
              kappa=1 / 15, slope_grid=make_slope_grid())
res["w_samples"]   # (num_samples, num_chains, p), on the simplex
res["slope_mean"]  # posterior mean of the calibration slope
```

`sig` is the source's noise standard deviation: a scalar, per feature `(p,)`, per
point `(N,)`, or per point and feature `(N, p)`. `inference_method="map"` returns the
posterior mode instead of draws.

## Modules

- `data`: triplets from the pools (`make_triplets`), their labels (`T_from_X`) and
  `kappa_from_triplets`.
- `inference`: the moments of the gap (`delta_V_one_triplet`), the log-likelihood and
  the calibration slope.
- `priors`: the Dirichlet log posterior, with or without the slope grid.
- `sampling`: NUTS and MAP.
- `diagnostics`: R-hat, ESS, WAIC and aligned triplets.
- `fit`: `fit_bii`, which composes them.

## Reproducing the paper

The experiments (<https://github.com/mascaretti/bayesian-information-imbalance>) pin
the commit `aa921d9`; this branch gives the same numbers, bit for bit, on a CPU. They
ran with jax 0.8.2, blackjax 1.7.1 and optax 0.2.8. Blackjax 1.7.1 declares
`jax>=0.9`, so install it and this package without dependencies:

```bash
pip install "jax==0.8.2" "jaxlib==0.8.2" "optax==0.2.8" numpy scipy typing-extensions
pip install --no-deps "blackjax==1.7.1" -e .
```

On that stack, `tests/test_fit.py::test_reproduces_the_reference_numbers` checks the
numbers; elsewhere it is skipped.

## Tests

```bash
pip install -e ".[dev]"
pytest    # under a minute on a laptop CPU
```
