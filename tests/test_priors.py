"""Tests for prior factory functions."""

import jax
import jax.numpy as jnp
import jax.random as jr

from bii.data import make_triplets
from bii.priors import (
    _log_z_density,
    make_dirichlet_logposterior,
    make_sparse_dirichlet_logposterior,
    sparse_dirichlet_dim,
    sparse_dirichlet_to_simplex,
)


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


def test_sparse_dirichlet_dim():
    assert sparse_dirichlet_dim(5) == 12
    assert sparse_dirichlet_dim(68) == 138


def test_sparse_dirichlet_to_simplex():
    p = 4
    position = jnp.zeros(sparse_dirichlet_dim(p))
    w = sparse_dirichlet_to_simplex(position)
    assert w.shape == (p,)
    assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-6)
    assert jnp.allclose(w, 0.25, atol=1e-6)


def test_log_z_density_normalizes():
    z = jnp.linspace(-20, 20, 10000)
    dz = z[1] - z[0]
    log_p = jax.vmap(lambda zi: _log_z_density(zi, 0.5, 0.5, 0.0, 1.0))(z)
    integral = jnp.sum(jnp.exp(log_p)) * dz
    assert jnp.abs(integral - 1.0) < 0.01


def test_log_z_density_symmetric_horseshoe():
    z_pos = _log_z_density(2.0, 0.5, 0.5, 0.0, 1.0)
    z_neg = _log_z_density(-2.0, 0.5, 0.5, 0.0, 1.0)
    assert jnp.abs(z_pos - z_neg) < 1e-5


def test_sparse_dirichlet_logposterior_finite():
    T, Z = _make_data(jr.PRNGKey(42), p=5)
    logprob_fn = make_sparse_dirichlet_logposterior(T, Z, sig=0.1)
    position = jnp.zeros(sparse_dirichlet_dim(5))
    lp = logprob_fn(position)
    assert jnp.isfinite(lp)


def test_sparse_dirichlet_logposterior_gradient_finite():
    T, Z = _make_data(jr.PRNGKey(7), p=4)
    logprob_fn = make_sparse_dirichlet_logposterior(T, Z, sig=0.1)
    position = jnp.zeros(sparse_dirichlet_dim(4))
    g = jax.grad(logprob_fn)(position)
    assert jnp.all(jnp.isfinite(g))
