"""Tests for likelihood functions in inference.py."""

import jax
import jax.numpy as jnp
import jax.random as jr
from hypothesis import given, settings
from hypothesis import strategies as st

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


# --- Shape and basic properties ---

def test_delta_V_shapes():
    p = 4
    zi = jnp.ones(p)
    zj = jnp.zeros(p)
    zk = 0.5 * jnp.ones(p)
    w = jnp.ones(p) / p
    sig2 = 0.01

    mu, V = delta_V_one_triplet(zi, zj, zk, w, sig2, sig2, sig2)
    assert mu.shape == ()
    assert V.shape == ()
    assert V >= 0


def test_delta_V_symmetry():
    """Swapping candidates i,j negates mu and preserves V."""
    p = 3
    zi = jnp.array([1.0, 0.0, 0.0])
    zj = jnp.array([0.0, 1.0, 0.0])
    zk = jnp.array([0.5, 0.5, 0.5])
    w = jnp.ones(p) / p
    sig2 = 0.01

    mu1, V1 = delta_V_one_triplet(zi, zj, zk, w, sig2, sig2, sig2)
    mu2, V2 = delta_V_one_triplet(zj, zi, zk, w, sig2, sig2, sig2)
    assert jnp.allclose(mu1, -mu2, atol=1e-6)
    assert jnp.allclose(V1, V2, atol=1e-6)


def test_delta_V_symmetry_heteroscedastic():
    """Swapping (i,j) with different σ: mu negates, V changes."""
    p = 3
    zi = jnp.array([1.0, 0.0, 0.0])
    zj = jnp.array([0.0, 1.0, 0.0])
    zk = jnp.array([0.5, 0.5, 0.5])
    w = jnp.ones(p) / p

    mu1, V1 = delta_V_one_triplet(zi, zj, zk, w, 0.01, 0.04, 0.02)
    mu2, V2 = delta_V_one_triplet(zj, zi, zk, w, 0.04, 0.01, 0.02)
    assert jnp.allclose(mu1, -mu2, atol=1e-6)
    assert jnp.allclose(V1, V2, atol=1e-6)


# --- Homoscedastic recovery ---

def test_delta_V_homoscedastic_recovery():
    """With equal σ for all points, should match the old formula:
    mu = Σ w(a²-b²), V = 8σ²(aa+bb-ab) + 12σ⁴·tr(W²).
    """
    zi = jnp.array([1.0, 0.2, -0.5, 0.3])
    zj = jnp.array([0.0, 1.0, 0.1, -0.2])
    zk = jnp.array([0.5, 0.5, 0.5, 0.0])
    w = jnp.array([0.4, 0.3, 0.2, 0.1])
    sig2 = 0.04  # σ² = 0.04

    mu, V = delta_V_one_triplet(zi, zj, zk, w, sig2, sig2, sig2)

    a = zi - zk
    b = zj - zk
    w2 = w * w
    expected_mu = jnp.sum(w * (a * a - b * b))
    aa = jnp.sum(w2 * a * a)
    bb = jnp.sum(w2 * b * b)
    ab = jnp.sum(w2 * a * b)
    tr = jnp.sum(w2 * sig2 * sig2)
    expected_V = 8.0 * sig2 * (aa + bb - ab) + 12.0 * sig2**2 * jnp.sum(w2)

    # No bias correction when σ_i = σ_j
    assert jnp.allclose(mu, expected_mu, atol=1e-6)
    assert jnp.allclose(V, expected_V, atol=1e-6)


# --- Bias correction ---

def test_delta_V_bias_correction():
    """When σ_i ≠ σ_j, mu includes the (σ_i² - σ_j²) tr(W) term."""
    p = 3
    # Make a = b so δ(w) = 0
    zi = jnp.array([1.0, 0.0, 0.5])
    zj = jnp.array([0.0, 1.0, 0.5])
    zk = jnp.array([0.5, 0.5, 0.5])
    w = jnp.ones(p) / p

    a = zi - zk
    b = zj - zk
    # Check δ(w) = 0: Σ w(a²-b²) should be 0 by construction
    assert jnp.allclose(jnp.sum(w * (a * a - b * b)), 0.0, atol=1e-6)

    sig2_i = 0.04
    sig2_j = 0.01
    sig2_k = 0.02

    mu, V = delta_V_one_triplet(zi, zj, zk, w, sig2_i, sig2_j, sig2_k)

    expected_bias = (sig2_i - sig2_j) * jnp.sum(w)
    assert jnp.allclose(mu, expected_bias, atol=1e-6)
    assert V > 0


# --- Zero noise ---

def test_delta_V_zero_noise():
    """σ = 0 for all → V = 0, mu = Σ w(a²-b²)."""
    zi = jnp.array([1.0, 0.0])
    zj = jnp.array([0.0, 1.0])
    zk = jnp.array([0.5, 0.5])
    w = jnp.array([0.6, 0.4])

    mu, V = delta_V_one_triplet(zi, zj, zk, w, 0.0, 0.0, 0.0)

    a = zi - zk
    b = zj - zk
    expected_mu = jnp.sum(w * (a * a - b * b))
    assert jnp.allclose(mu, expected_mu, atol=1e-12)
    assert jnp.allclose(V, 0.0, atol=1e-12)


# --- V non-negative (property test) ---

@given(
    sig2_i=st.floats(min_value=0.0, max_value=10.0),
    sig2_j=st.floats(min_value=0.0, max_value=10.0),
    sig2_k=st.floats(min_value=0.0, max_value=10.0),
)
@settings(max_examples=200)
def test_delta_V_V_nonnegative(sig2_i, sig2_j, sig2_k):
    """V(w) must be non-negative for any per-point variances."""
    p = 3
    zi = jnp.array([1.0, 0.0, -0.5])
    zj = jnp.array([0.0, 1.0, 0.3])
    zk = jnp.array([0.5, 0.5, 0.0])
    w = jnp.array([0.5, 0.3, 0.2])

    _, V = delta_V_one_triplet(zi, zj, zk, w, sig2_i, sig2_j, sig2_k)
    assert float(V) >= -1e-10


# --- Loglik tests ---

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


def test_loglik_w_vector_sig():
    """Per-component σ vector should work."""
    key = jr.PRNGKey(4)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3

    sig_scalar = 0.1
    sig_vec = 0.1 * jnp.ones(3)

    ll_scalar = loglik_w(w, T, Z, sig=sig_scalar)
    ll_vec = loglik_w(w, T, Z, sig=sig_vec)
    assert jnp.allclose(ll_scalar, ll_vec, atol=1e-4)


# --- Multiplicative noise model ---

def test_loglik_multiplicative_finite():
    """Multiplicative likelihood should be finite for positive Z."""
    key = jr.PRNGKey(5)
    T, Z = _make_simple_data(key, n=30, p=3)
    Z = jnp.abs(Z) + 1.0  # ensure positive
    w = jnp.ones(3) / 3

    ll = loglik_w(w, T, Z, sig=0.3, noise_model="multiplicative")
    assert jnp.isfinite(ll)
    assert ll <= 0


def test_loglik_multiplicative_per_triplet_sums():
    key = jr.PRNGKey(6)
    T, Z = _make_simple_data(key, n=40, p=4)
    Z = jnp.abs(Z) + 1.0
    w = jnp.array([0.4, 0.3, 0.2, 0.1])

    total = loglik_w(w, T, Z, sig=0.3, noise_model="multiplicative")
    per_t = loglik_w_per_triplet(w, T, Z, sig=0.3, noise_model="multiplicative")
    assert jnp.allclose(jnp.sum(per_t), total, atol=1e-4)


def test_loglik_multiplicative_gradient():
    key = jr.PRNGKey(7)
    T, Z = _make_simple_data(key, n=30, p=3)
    Z = jnp.abs(Z) + 1.0
    w = jnp.ones(3) / 3

    g = jax.grad(loglik_w)(w, T, Z, sig=0.3, noise_model="multiplicative")
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))


def test_loglik_multiplicative_differs_from_additive():
    """Multiplicative and additive likelihoods should differ for positive Z."""
    key = jr.PRNGKey(8)
    T, Z = _make_simple_data(key, n=30, p=3)
    Z = jnp.abs(Z) + 1.0
    w = jnp.ones(3) / 3

    ll_add = loglik_w(w, T, Z, sig=0.3, noise_model="additive")
    ll_mul = loglik_w(w, T, Z, sig=0.3, noise_model="multiplicative")
    assert not jnp.allclose(ll_add, ll_mul, atol=1e-4)
