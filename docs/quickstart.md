# Quick start

## The one-call pipeline

{func}`bii.fit_bii` runs the whole pipeline — triplet construction, posterior
sampling, and diagnostics — and returns a dictionary of results.

```python
from jax import random
from bii import fit_bii

key = random.PRNGKey(42)
result = fit_bii(
    key, X_pool, Z_pool, sig=0.1,   # clean target, noisy source, noise std
    n_triplets=15, anchor_fraction=0.5,
    num_samples=2000, num_warmup=1000, num_chains=4,
)
w_samples = result["w_samples"]      # (num_samples, num_chains, p), each row on the simplex
w_mean = w_samples.reshape(-1, w_samples.shape[-1]).mean(0)
```

`X_pool` is the clean target embedding `(N, p_x)` and `Z_pool` the noisy or
rescaled source `(N, p_z)` of the same `N` objects; `sig` is the source noise
standard deviation (a scalar, a per-feature `(p,)` vector, or per-point).

### What `fit_bii` returns

| key | meaning |
| --- | --- |
| `w_samples` | posterior draws of the simplex weights, `(num_samples, num_chains, p)` |
| `raw_samples` | the unconstrained-space draws before the softmax transform |
| `diagnostics` | R-hat and ESS per coordinate (see {mod}`bii.diagnostics`) |
| `waic` | held-out predictive score (`None` if `compute_waic_flag=False`) |
| `alignment` | entropy, triplet accuracy, and alignment index of the fit |
| `kappa` | the power-likelihood (Godambe) correction actually used |
| `pi_samples`, `tau_samples` | draws for the optional inclusion-mixture / relation-noise extensions |
| `elapsed_seconds` | wall-clock time of the fit |

## The composable API

`fit_bii` is a thin composition of pure functions you can call directly when you
need finer control:

```python
import jax.numpy as jnp
from bii import make_triplets, make_dirichlet_logposterior, run_nuts

# 1. build triplets from the paired pools
T, X, Z, idx = make_triplets(key, X_pool, Z_pool, n_triplets=15, anchor_fraction=0.5)

# 2. assemble the log-posterior (probit likelihood + Dirichlet prior)
p = Z.shape[-1]
logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(p), kappa=1.0)

# 3. sample
raw_samples, acc = run_nuts(key, logprob_fn, jnp.zeros(p),
                            num_samples=2000, num_warmup=1000, num_chains=4)
```

See {mod}`bii.data` for alternative triplet samplers (Z-far, Y-far,
rank-weighted, informative, sparse), {mod}`bii.sampling` for the
variational-inference runner {func}`bii.run_vi`, and {mod}`bii.diagnostics` for
R-hat, ESS, WAIC, and alignment scores.

## Variational inference

For a fast approximate posterior, switch the inference method:

```python
result = fit_bii(key, X_pool, Z_pool, sig=0.1, inference_method="vi",
                 vi_steps=5000, vi_num_samples=2000)
```

## MAP (posterior mode)

For a **fast point estimate** — e.g. exploratory runs at large `p`, where NUTS is
expensive — use `inference_method="map"`. It ascends the same log-posterior the
sampler targets to its mode; there is no uncertainty quantification.

```python
result = fit_bii(key, X_pool, Z_pool, sig=0.1, inference_method="map",
                 map_steps=3000, map_restarts=1)
w_map = result["w_samples"][0, 0]                 # w_samples has shape (1, 1, p)
logprob_history = result["diagnostics"]["logprob_history"]
```

Or call {func}`bii.run_map` directly on any log-posterior:

```python
import jax
from bii import make_triplets, make_dirichlet_logposterior, run_map

T, X, Z, idx = make_triplets(key, X_pool, Z_pool, n_triplets=20)
logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(p))
pos, logprob_history = run_map(key, logprob_fn, dim=p, num_steps=3000)
w_map = jax.nn.softmax(pos)
```

MAP recovers the weights accurately and scales well: on a Gaussian design it
matches `w*` to a few parts in $10^{3}$ and runs at `p = 400` in well under a
minute on CPU. Use NUTS when you need calibrated uncertainty (VI tends to
*under*estimate it).
