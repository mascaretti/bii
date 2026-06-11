"""Tests for likelihood functions in inference.py."""

import jax
import jax.numpy as jnp
import jax.random as jr
from hypothesis import given, settings
from hypothesis import strategies as st

from bii.data import make_triplets
from bii.inference import (
    delta_V_one_triplet,
    inclusion_probs,
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


# --- clip_s (censored-probit robustifier) ---

def _saturating_data():
    """One saturating-wrong triplet: i is far in Z but labelled close (T=1)."""
    Z = jnp.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
    T = jnp.array([1.0])
    return T, Z


def test_clip_s_bounds_saturating_loss():
    """Clipping bounds a saturating-wrong triplet's loglik contribution."""
    T, Z = _saturating_data()
    w = jnp.ones(3) / 3
    ll_plain = loglik_w(w, T, Z, sig=0.1)
    ll_clipped = loglik_w(w, T, Z, sig=0.1, clip_s=2.5)
    assert ll_clipped > ll_plain
    # Clipped loss is exactly log Phi(-clip_s)
    from jax.scipy.special import log_ndtr
    assert jnp.allclose(ll_clipped, log_ndtr(-2.5), atol=1e-5)


def test_clip_s_noop_in_calibrated_regime():
    """A generous clip leaves non-saturating triplets untouched."""
    key = jr.PRNGKey(0)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    ll_plain = loglik_w(w, T, Z, sig=1.0)  # large sig -> small |s|
    ll_clipped = loglik_w(w, T, Z, sig=1.0, clip_s=50.0)
    assert jnp.allclose(ll_plain, ll_clipped, atol=1e-5)


def test_clip_s_zero_gradient_when_saturating():
    """Clipped triplets contribute zero gradient in w."""
    T, Z = _saturating_data()
    w = jnp.ones(3) / 3
    g = jax.grad(loglik_w)(w, T, Z, sig=0.1, clip_s=2.5)
    assert jnp.allclose(g, 0.0, atol=1e-6)
    g_plain = jax.grad(loglik_w)(w, T, Z, sig=0.1)
    assert not jnp.allclose(g_plain, 0.0, atol=1e-6)


# --- pi_inclusion (mixture likelihood) ---

def test_pi_inclusion_lower_bounds_per_triplet():
    """Mixture loglik is at least n * log((1-pi)/2) — noise floor."""
    T, Z = _saturating_data()
    w = jnp.ones(3) / 3
    pi = 0.8
    ll = loglik_w(w, T, Z, sig=0.1, pi_inclusion=pi)
    assert ll >= jnp.log((1.0 - pi) * 0.5)
    # And the plain likelihood for this saturating-wrong triplet is far below
    assert loglik_w(w, T, Z, sig=0.1) < ll


def test_pi_inclusion_recovers_plain_at_one():
    """pi -> 1 recovers the plain probit likelihood."""
    key = jr.PRNGKey(1)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    ll_plain = loglik_w(w, T, Z, sig=0.1)
    ll_mix = loglik_w(w, T, Z, sig=0.1, pi_inclusion=1.0 - 1e-7)
    assert jnp.allclose(ll_plain, ll_mix, atol=1e-3)


def test_pi_inclusion_gradient_finite():
    key = jr.PRNGKey(2)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    g = jax.grad(loglik_w)(w, T, Z, sig=0.1, pi_inclusion=0.7)
    assert jnp.all(jnp.isfinite(g))


# --- inclusion_probs ---

def test_inclusion_probs_range_and_shape():
    key = jr.PRNGKey(3)
    T, Z = _make_simple_data(key, n=40, p=3)
    w = jnp.ones(3) / 3
    probs = inclusion_probs(w, T, Z, sig=0.1, pi_inclusion=0.8)
    assert probs.shape == T.shape
    assert jnp.all(probs >= 0.0) and jnp.all(probs <= 1.0)


def test_inclusion_probs_discriminates():
    """Well-predicted triplet -> high inclusion; saturating-wrong -> low."""
    # Triplet 0: i close in Z and T=1 (consistent). Triplet 1: i far, T=1 (wrong).
    Z = jnp.array([
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    ])
    T = jnp.array([1.0, 1.0])
    w = jnp.ones(3) / 3
    pi = 0.8
    probs = inclusion_probs(w, T, Z, sig=0.1, pi_inclusion=pi)
    # Consistent triplet: P_t ~ 1 -> pi / (pi + (1-pi)*0.5)
    assert probs[0] > 0.85
    # Saturating-wrong triplet: P_t ~ 0 -> inclusion ~ 0
    assert probs[1] < 0.05


# --- triplet_weights and pre-resolved sigmas ---

def test_triplet_weights_scale_loglik():
    """Constant weights c scale the loglik by c."""
    key = jr.PRNGKey(4)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    ll = loglik_w(w, T, Z, sig=0.1)
    ll_weighted = loglik_w(w, T, Z, sig=0.1,
                           triplet_weights=2.0 * jnp.ones(T.shape[0]))
    assert jnp.allclose(ll_weighted, 2.0 * ll, atol=1e-4)


def test_preresolved_sig_matches_scalar():
    """(n, 3) per-triplet sigmas filled with one value == scalar sig."""
    key = jr.PRNGKey(5)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    sig_pre = jnp.full((T.shape[0], 3), 0.1)
    ll_scalar = loglik_w(w, T, Z, sig=0.1)
    ll_pre = loglik_w(w, T, Z, sig=sig_pre)
    assert jnp.allclose(ll_scalar, ll_pre, atol=1e-4)
    per_scalar = loglik_w_per_triplet(w, T, Z, sig=0.1)
    per_pre = loglik_w_per_triplet(w, T, Z, sig=sig_pre)
    assert jnp.allclose(per_scalar, per_pre, atol=1e-4)


# --- logit link ---

def test_logit_link_slope_matched_to_probit():
    """sigmoid(1.702 s) deviates from Phi(s) by < 0.01 uniformly in s."""
    from jax.scipy.special import ndtr

    from bii.inference import LOGIT_SCALE, logP_log1mP_from_deltaV

    s = jnp.linspace(-8.0, 8.0, 400)
    logP, _ = logP_log1mP_from_deltaV(s, jnp.ones_like(s), link="logit")
    p_logit = jnp.exp(logP)
    p_probit = ndtr(-s)
    assert LOGIT_SCALE == 1.702
    assert jnp.max(jnp.abs(p_logit - p_probit)) < 0.01


def test_logit_link_lighter_saturating_penalty():
    """On a saturating-wrong triplet, the logistic loglik is far less harsh."""
    T, Z = _saturating_data()
    w = jnp.ones(3) / 3
    ll_probit = loglik_w(w, T, Z, sig=0.1)
    ll_logit = loglik_w(w, T, Z, sig=0.1, link="logit")
    assert ll_logit > ll_probit
    # probit tail ~ -s^2/2, logit tail ~ -1.702|s|: orders of magnitude apart
    assert ll_logit / ll_probit < 0.5


def test_logit_link_gradient_finite():
    key = jr.PRNGKey(10)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    g = jax.grad(loglik_w)(w, T, Z, sig=0.1, link="logit")
    assert jnp.all(jnp.isfinite(g))


def test_logit_link_composes_with_mixture_and_clip():
    key = jr.PRNGKey(11)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    ll = loglik_w(w, T, Z, sig=0.1, link="logit", pi_inclusion=0.8, clip_s=2.5)
    assert jnp.isfinite(ll)


def test_logit_link_per_triplet_sums_to_total():
    key = jr.PRNGKey(12)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    total = loglik_w(w, T, Z, sig=0.1, link="logit")
    per_t = loglik_w_per_triplet(w, T, Z, sig=0.1, link="logit")
    assert jnp.allclose(jnp.sum(per_t), total, atol=1e-4)


def test_unknown_link_raises():
    import pytest

    from bii.inference import logP_log1mP_from_deltaV

    with pytest.raises(ValueError):
        logP_log1mP_from_deltaV(jnp.zeros(3), jnp.ones(3), link="cauchy")


def test_full_covariance_sig_raises():
    """A (p, p) covariance matrix must fail loudly, not be silently sliced."""
    import pytest

    key = jr.PRNGKey(13)
    T, Z = _make_simple_data(key, n=30, p=3)
    w = jnp.ones(3) / 3
    sig_full = 0.01 * jnp.eye(3)
    with pytest.raises(ValueError, match="not supported"):
        loglik_w(w, T, Z, sig=sig_full)
    with pytest.raises(ValueError, match="not supported"):
        loglik_w_per_triplet(w, T, Z, sig=sig_full)
