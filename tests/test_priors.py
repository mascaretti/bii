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


# --- pi_prior: Beta(a, b) on the inclusion-mixture pi ---

def test_pi_prior_position_grows_by_one():
    """With pi_prior, logprob_fn takes a (p+1,) position (theta + logit_pi)."""
    T, Z = _make_data(jr.PRNGKey(0), p=4)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(4), pi_prior=(2.0, 2.0)
    )
    lp = logprob_fn(jnp.zeros(5))
    assert lp.shape == ()
    assert jnp.isfinite(lp)


def test_pi_prior_gradient_finite():
    T, Z = _make_data(jr.PRNGKey(1), p=3)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), pi_prior=(2.0, 2.0)
    )
    g = jax.grad(logprob_fn)(jnp.zeros(4))
    assert g.shape == (4,)
    assert jnp.all(jnp.isfinite(g))


def test_pi_prior_overrides_pi_inclusion():
    """When pi_prior is given, the fixed pi_inclusion value is ignored."""
    T, Z = _make_data(jr.PRNGKey(2), p=3)
    kwargs = dict(sig=0.1, alpha=jnp.ones(3), pi_prior=(2.0, 2.0))
    fn_a = make_dirichlet_logposterior(T, Z, pi_inclusion=0.9, **kwargs)
    fn_b = make_dirichlet_logposterior(T, Z, pi_inclusion=0.1, **kwargs)
    pos = jnp.concatenate([jnp.zeros(3), jnp.array([0.5])])
    assert jnp.allclose(fn_a(pos), fn_b(pos), atol=1e-6)


def test_pi_prior_beta_term_shifts_logprob():
    """A larger Beta `a` tilts the (unnormalised) logposterior toward high pi.

    The Beta normalising constant is dropped, so compare differences between
    two positions (constants cancel) rather than values at a single point.
    """
    T, Z = _make_data(jr.PRNGKey(3), p=3)
    fn_weak = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), pi_prior=(2.0, 2.0)
    )
    fn_high = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), pi_prior=(20.0, 2.0)
    )
    pos_hi = jnp.concatenate([jnp.zeros(3), jnp.array([2.0])])  # pi ~ 0.88
    pos_lo = jnp.concatenate([jnp.zeros(3), jnp.array([0.0])])  # pi = 0.5
    slope_weak = fn_weak(pos_hi) - fn_weak(pos_lo)
    slope_high = fn_high(pos_hi) - fn_high(pos_lo)
    assert slope_high > slope_weak


# --- clip_s and pi_inclusion pass-through ---

def test_logposterior_clip_s_finite_gradient():
    T, Z = _make_data(jr.PRNGKey(4), p=3)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), clip_s=2.5
    )
    g = jax.grad(logprob_fn)(jnp.zeros(3))
    assert jnp.all(jnp.isfinite(g))


def test_logposterior_pi_inclusion_finite():
    T, Z = _make_data(jr.PRNGKey(5), p=3)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), pi_inclusion=0.8
    )
    lp = logprob_fn(jnp.zeros(3))
    assert jnp.isfinite(lp)


def test_logposterior_triplet_weights_forwarded():
    """Doubling all triplet weights doubles the likelihood part."""
    T, Z = _make_data(jr.PRNGKey(6), p=3)
    alpha = jnp.ones(3)
    n = T.shape[0]
    fn_plain = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=alpha)
    fn_double = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=alpha, triplet_weights=2.0 * jnp.ones(n)
    )
    theta = jnp.zeros(3)
    prior_term = jnp.sum(alpha * jnp.log(jax.nn.softmax(theta) + 1e-12))
    ll_plain = fn_plain(theta) - prior_term
    ll_double = fn_double(theta) - prior_term
    assert jnp.allclose(ll_double, 2.0 * ll_plain, atol=1e-4)
