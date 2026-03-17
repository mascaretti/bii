"""Tests for MCMC and VI runners."""

import jax.numpy as jnp
import jax.random as jr

from bii.data import make_triplets
from bii.fit import fit_bii
from bii.priors import make_dirichlet_logposterior
from bii.sampling import run_nuts, run_vi, sample_vi


def _make_pool(key, n=100, p=3, sig=0.1, w_star=None):
    k1, k2 = jr.split(key)
    x_pool = jr.normal(k1, (n, p))
    if w_star is not None:
        x_pool = x_pool * jnp.sqrt(w_star)[None, :]
    eps = jr.normal(k2, (n, p)) * sig
    z_pool = x_pool + eps
    return x_pool, z_pool


def _make_logprob(key, p=3):
    X, Z = _make_pool(key, n=100, p=p)
    T, _, Z_trip, _ = make_triplets(key, X, Z, n_triplets=50)
    return make_dirichlet_logposterior(T, Z_trip, sig=0.1, alpha=jnp.ones(p))


def test_run_nuts_smoke():
    key = jr.PRNGKey(0)
    logprob_fn = _make_logprob(key, p=3)
    raw, acc = run_nuts(
        key, logprob_fn, jnp.zeros(3),
        num_samples=30, num_warmup=30, num_chains=1,
    )
    assert raw.shape == (30, 1, 3)
    assert acc.shape == (30, 1)


def test_run_vi_smoke():
    key = jr.PRNGKey(1)
    logprob_fn = _make_logprob(key, p=3)
    mu, log_sigma, elbo = run_vi(key, logprob_fn, dim=3, num_steps=200)
    assert mu.shape == (3,)
    assert log_sigma.shape == (3,)
    assert elbo.shape == (200,)


def test_sample_vi_shapes():
    key = jr.PRNGKey(2)
    mu = jnp.zeros(4)
    log_sigma = -jnp.ones(4)
    theta, w = sample_vi(key, mu, log_sigma, num_samples=50)
    assert theta.shape == (50, 4)
    assert w.shape == (50, 4)
    assert jnp.allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-5)


def test_vi_elbo_increases():
    key = jr.PRNGKey(99)
    logprob_fn = _make_logprob(key, p=3)
    _, _, elbo = run_vi(key, logprob_fn, dim=3, num_steps=1000)
    assert elbo[-1] > elbo[0]


def test_vi_smoke_via_fit_bii():
    key = jr.PRNGKey(0)
    x_pool, z_pool = _make_pool(key, n=100, p=3)
    result = fit_bii(
        key, x_pool, z_pool, sig=0.1,
        inference_method="vi", n_triplets=50,
        vi_steps=500, vi_num_samples=100,
    )
    w = result["w_samples"]
    assert w.ndim == 3
    assert w.shape[1] == 1
    assert jnp.allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-5)
    assert jnp.isfinite(result["waic"])


def test_vi_posterior_not_uniform():
    """VI posterior should differ from uniform prior after seeing data."""
    key = jr.PRNGKey(42)
    w_star = jnp.array([0.7, 0.2, 0.1])
    x_pool, z_pool = _make_pool(key, n=500, p=3, sig=0.05, w_star=w_star)
    result = fit_bii(
        key, x_pool, z_pool, sig=0.05,
        inference_method="vi", n_triplets=100, anchor_fraction=0.5,
        vi_steps=5000, vi_num_samples=2000,
    )
    w_mean = result["w_samples"][:, 0, :].mean(0)
    # Posterior should not be exactly uniform
    assert not jnp.allclose(w_mean, jnp.ones(3) / 3, atol=0.01)
