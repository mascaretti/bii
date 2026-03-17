"""Tests for WAIC, R-hat, and ESS diagnostics."""

import jax.numpy as jnp
import jax.random as jr

from bii.data import make_triplets
from bii.diagnostics import compute_ess, compute_rhat, compute_waic


def _make_triplet_data(key, n=100, p=3, sig=0.1):
    k1, k2 = jr.split(key)
    X = jr.normal(k1, (n, p))
    Z = X + jr.normal(k2, (n, p)) * sig
    T, _, Z_trip, _ = make_triplets(key, X, Z, n_triplets=30)
    return T, Z_trip


def test_waic_finite():
    key = jr.PRNGKey(0)
    T, Z = _make_triplet_data(key)
    w_samples = jnp.ones((50, 3)) / 3
    waic = compute_waic(w_samples, T, Z, sig=0.1)
    assert waic.shape == ()
    assert jnp.isfinite(waic)


def test_waic_varies_with_weights():
    key = jr.PRNGKey(2)
    T, Z = _make_triplet_data(key)
    S = 50
    w1 = jnp.ones((S, 3)) / 3
    w2 = jnp.broadcast_to(jnp.array([0.8, 0.1, 0.1]), (S, 3))
    waic1 = compute_waic(w1, T, Z, sig=0.1)
    waic2 = compute_waic(w2, T, Z, sig=0.1)
    assert not jnp.allclose(waic1, waic2, atol=1e-6)


def test_rhat_identical_chains():
    """Identical chains should have R-hat = 1."""
    samples = jnp.ones((100, 2, 3)) * 0.5
    # Add small noise to avoid zero variance
    key = jr.PRNGKey(0)
    samples = samples + 0.001 * jr.normal(key, (100, 2, 3))
    rhat = compute_rhat(samples)
    assert rhat.shape == (3,)
    assert jnp.all(rhat < 1.1)


def test_ess_positive():
    """ESS should be positive."""
    key = jr.PRNGKey(1)
    samples = jr.normal(key, (100, 2, 3))
    ess = compute_ess(samples)
    assert ess.shape == (3,)
    assert jnp.all(ess > 0)
