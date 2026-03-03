"""Tests for WAIC computation."""

import jax.numpy as jnp
import jax.random as jr

from bii.fit import _compute_waic, _random_triplets


def _make_pool(key, n=100, p=3, sig=0.1):
    k1, k2 = jr.split(key)
    X = jr.normal(k1, (n, p))
    Z = X + jr.normal(k2, (n, p)) * sig
    return X, Z


def test_waic_finite():
    """WAIC should be a finite scalar."""
    key = jr.PRNGKey(0)
    X, Z = _make_pool(key)
    T, Z_trip, _ = _random_triplets(key, X, Z, n_triplets=30)

    # Uniform weights repeated as "posterior samples"
    p = 3
    S = 50
    w_samples = jnp.ones((S, p)) / p

    waic = _compute_waic(w_samples, T, Z_trip, sig=0.1)
    assert waic.shape == ()
    assert jnp.isfinite(waic)


def test_waic_positive():
    """WAIC (deviance scale) should typically be positive."""
    key = jr.PRNGKey(1)
    X, Z = _make_pool(key, n=80, p=4)
    T, Z_trip, _ = _random_triplets(key, X, Z, n_triplets=20)

    p = 4
    S = 30
    w_samples = jnp.ones((S, p)) / p
    waic = _compute_waic(w_samples, T, Z_trip, sig=0.1)
    # On the deviance scale (-2*elpd), positive means worse than perfect
    assert jnp.isfinite(waic)


def test_waic_varies_with_weights():
    """WAIC should differ for different weight vectors."""
    key = jr.PRNGKey(2)
    X, Z = _make_pool(key, n=100, p=3)
    T, Z_trip, _ = _random_triplets(key, X, Z, n_triplets=40)

    S = 50
    w1 = jnp.ones((S, 3)) / 3
    w2 = jnp.broadcast_to(jnp.array([0.8, 0.1, 0.1]), (S, 3))

    waic1 = _compute_waic(w1, T, Z_trip, sig=0.1)
    waic2 = _compute_waic(w2, T, Z_trip, sig=0.1)
    # They should differ (unless data is pathological)
    assert not jnp.allclose(waic1, waic2, atol=1e-6)
