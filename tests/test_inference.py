"""Tests for likelihood functions in inference.py."""

import jax
import jax.numpy as jnp
import jax.random as jr

from bii.inference import (
    delta_V_one_triplet,
    loglik_w,
    loglik_w_per_triplet,
    loglik_theta,
)


def _make_simple_data(key, n=50, p=3, sig=0.1):
    """Generate simple triplet data for testing."""
    k1, k2 = jr.split(key)
    X_pool = jr.normal(k1, (200, p))
    Z_pool = X_pool + jr.normal(k2, (200, p)) * sig

    from bii.fit import _random_triplets
    T, Z, _ = _random_triplets(key, X_pool, Z_pool, n_triplets=n)
    return T, Z


def test_delta_V_shapes():
    """delta_V_one_triplet returns scalar delta and V."""
    p = 4
    zi = jnp.ones(p)
    zj = jnp.zeros(p)
    zk = 0.5 * jnp.ones(p)
    w = jnp.ones(p) / p
    sig2 = 0.01 * jnp.ones(p)

    delta, V = delta_V_one_triplet(zi, zj, zk, w, sig2)
    assert delta.shape == ()
    assert V.shape == ()
    assert V >= 0  # variance is non-negative


def test_delta_V_symmetry():
    """Swapping i,j should flip the sign of delta."""
    p = 3
    zi = jnp.array([1.0, 0.0, 0.0])
    zj = jnp.array([0.0, 1.0, 0.0])
    zk = jnp.array([0.5, 0.5, 0.5])
    w = jnp.ones(p) / p
    sig2 = 0.01 * jnp.ones(p)

    d1, V1 = delta_V_one_triplet(zi, zj, zk, w, sig2)
    d2, V2 = delta_V_one_triplet(zj, zi, zk, w, sig2)
    assert jnp.allclose(d1, -d2, atol=1e-6)
    assert jnp.allclose(V1, V2, atol=1e-6)


def test_loglik_w_finite():
    """loglik_w should return a finite scalar."""
    key = jr.PRNGKey(0)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    ll = loglik_w(w, T, Z, sig=0.1)
    assert ll.shape == ()
    assert jnp.isfinite(ll)
    assert ll <= 0  # log-likelihood is non-positive


def test_loglik_w_per_triplet_sums_to_total():
    """Per-triplet log-lik should sum to total log-lik."""
    key = jr.PRNGKey(1)
    T, Z = _make_simple_data(key, n=40, p=4)
    w = jnp.array([0.4, 0.3, 0.2, 0.1])

    total = loglik_w(w, T, Z, sig=0.1)
    per_triplet = loglik_w_per_triplet(w, T, Z, sig=0.1)
    assert per_triplet.shape == (40,)
    assert jnp.allclose(jnp.sum(per_triplet), total, atol=1e-4)


def test_loglik_theta_softmax_equivalence():
    """loglik_theta(theta) should equal loglik_w(softmax(theta))."""
    key = jr.PRNGKey(2)
    T, Z = _make_simple_data(key, n=30, p=3)
    theta = jnp.array([0.5, -0.3, 0.1])

    ll_theta = loglik_theta(theta, T, Z, sig=0.1)
    ll_w = loglik_w(jax.nn.softmax(theta), T, Z, sig=0.1)
    assert jnp.allclose(ll_theta, ll_w, atol=1e-6)


def test_loglik_w_gradient():
    """Gradient of loglik_w w.r.t. w should be finite."""
    key = jr.PRNGKey(3)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3

    grad_fn = jax.grad(loglik_w)
    g = grad_fn(w, T, Z, sig=0.1)
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))
