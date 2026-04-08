"""Tests for prior factory functions."""

import jax
import jax.numpy as jnp
import jax.random as jr

from bii.data import make_triplets
from bii.priors import make_dirichlet_logposterior


def _make_data(key, p=5):
    k1, k2 = jr.split(key)
    N = 30
    X = jr.normal(k1, (N, p))
    Z = X + 0.1 * jr.normal(k2, (N, p))
    T, _, Z_trip, _ = make_triplets(jr.PRNGKey(0), X, Z, n_triplets=20)
    return T, Z_trip


def test_dirichlet_logposterior_finite():
    T, Z = _make_data(jr.PRNGKey(42), p=4)
    logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(4))
    lp = logprob_fn(jnp.zeros(4))
    assert jnp.isfinite(lp)


def test_dirichlet_logposterior_gradient_finite():
    T, Z = _make_data(jr.PRNGKey(7), p=3)
    logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(3))
    g = jax.grad(logprob_fn)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g))


def test_dirichlet_multiplicative_finite():
    T, Z = _make_data(jr.PRNGKey(42), p=4)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.3, alpha=jnp.ones(4), noise_model="multiplicative"
    )
    lp = logprob_fn(jnp.zeros(4))
    assert jnp.isfinite(lp)


def test_dirichlet_multiplicative_gradient_finite():
    T, Z = _make_data(jr.PRNGKey(7), p=3)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.3, alpha=jnp.ones(3), noise_model="multiplicative"
    )
    g = jax.grad(logprob_fn)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g))
