"""Tests for likelihood functions in inference.py."""

import jax
import jax.numpy as jnp
import jax.random as jr

from bii.data import make_triplets
from bii.inference import (
    delta_V_one_triplet,
    loglik_theta,
    loglik_w,
    loglik_w_per_triplet,
)


def _make_simple_data(key, n=50, p=3, sig=0.1):
    k1, k2 = jr.split(key)
    X_pool = jr.normal(k1, (200, p))
    Z_pool = X_pool + jr.normal(k2, (200, p)) * sig
    T, _, Z, _ = make_triplets(key, X_pool, Z_pool, n_triplets=n)
    return T, Z


def test_delta_V_shapes():
    p = 4
    zi = jnp.ones(p)
    zj = jnp.zeros(p)
    zk = 0.5 * jnp.ones(p)
    w = jnp.ones(p) / p
    sig2 = 0.01 * jnp.ones(p)

    delta, V = delta_V_one_triplet(zi, zj, zk, w, sig2)
    assert delta.shape == ()
    assert V.shape == ()
    assert V >= 0


def test_delta_V_symmetry():
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
    key = jr.PRNGKey(0)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    ll = loglik_w(w, T, Z, sig=0.1)
    assert ll.shape == ()
    assert jnp.isfinite(ll)
    assert ll <= 0


def test_loglik_w_per_triplet_sums_to_total():
    key = jr.PRNGKey(1)
    T, Z = _make_simple_data(key, n=40, p=4)
    w = jnp.array([0.4, 0.3, 0.2, 0.1])

    total = loglik_w(w, T, Z, sig=0.1)
    per_triplet = loglik_w_per_triplet(w, T, Z, sig=0.1)
    assert jnp.allclose(jnp.sum(per_triplet), total, atol=1e-4)


def test_loglik_theta_softmax_equivalence():
    key = jr.PRNGKey(2)
    T, Z = _make_simple_data(key, n=30, p=3)
    theta = jnp.array([0.5, -0.3, 0.1])

    ll_theta = loglik_theta(theta, T, Z, sig=0.1)
    ll_w = loglik_w(jax.nn.softmax(theta), T, Z, sig=0.1)
    assert jnp.allclose(ll_theta, ll_w, atol=1e-6)


def test_loglik_w_gradient():
    key = jr.PRNGKey(3)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3

    g = jax.grad(loglik_w)(w, T, Z, sig=0.1)
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))


# Full covariance tests

def test_delta_V_full_sigma_diagonal_equivalence():
    zi = jnp.array([1.0, 0.2, -0.5, 0.3])
    zj = jnp.array([0.0, 1.0, 0.1, -0.2])
    zk = jnp.array([0.5, 0.5, 0.5, 0.0])
    w = jnp.array([0.4, 0.3, 0.2, 0.1])
    sig2_diag = jnp.array([0.01, 0.04, 0.02, 0.03])

    d_diag, V_diag = delta_V_one_triplet(zi, zj, zk, w, sig2_diag)
    d_full, V_full = delta_V_one_triplet(zi, zj, zk, w, jnp.diag(sig2_diag))
    assert jnp.allclose(d_diag, d_full, atol=1e-6)
    assert jnp.allclose(V_diag, V_full, atol=1e-6)


def test_delta_V_full_sigma_off_diagonal():
    p = 3
    zi = jnp.array([1.0, 0.0, 0.0])
    zj = jnp.array([0.0, 1.0, 0.0])
    zk = jnp.array([0.5, 0.5, 0.5])
    w = jnp.ones(p) / p

    sig2_diag = 0.01 * jnp.ones(p)
    Sigma_full = 0.01 * jnp.eye(p) + 0.005 * jnp.ones((p, p))

    _, V_diag = delta_V_one_triplet(zi, zj, zk, w, sig2_diag)
    _, V_full = delta_V_one_triplet(zi, zj, zk, w, Sigma_full)
    assert V_full > V_diag


def test_loglik_w_matrix_sig():
    key = jr.PRNGKey(4)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3

    sig_vec = 0.1 * jnp.ones(3)
    sig_mat = jnp.diag(sig_vec**2)

    ll_vec = loglik_w(w, T, Z, sig=sig_vec)
    ll_mat = loglik_w(w, T, Z, sig=sig_mat)
    assert jnp.allclose(ll_vec, ll_mat, atol=1e-4)


def test_loglik_w_matrix_sig_gradient():
    key = jr.PRNGKey(5)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3

    A = jr.normal(jr.PRNGKey(99), (3, 3))
    Sigma = A @ A.T + 0.01 * jnp.eye(3)

    g = jax.grad(loglik_w)(w, T, Z, sig=Sigma)
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))
