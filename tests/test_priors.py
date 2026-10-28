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


# --- tau_prior: Gamma(a, b) on the extra relation-noise std tau ---

def test_tau_prior_position_grows_by_one():
    T, Z = _make_data(jr.PRNGKey(10), p=4)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(4), tau_prior=(2.0, 2.0)
    )
    lp = logprob_fn(jnp.zeros(5))
    assert lp.shape == ()
    assert jnp.isfinite(lp)


def test_tau_prior_gradient_finite():
    T, Z = _make_data(jr.PRNGKey(11), p=3)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), tau_prior=(2.0, 2.0)
    )
    g = jax.grad(logprob_fn)(jnp.zeros(4))
    assert g.shape == (4,)
    assert jnp.all(jnp.isfinite(g))


def test_tau_prior_gamma_term_penalises_large_tau():
    """The b (rate) hyperparameter penalises large tau (slopes differ)."""
    T, Z = _make_data(jr.PRNGKey(12), p=3)
    fn_loose = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), tau_prior=(2.0, 1.0)
    )
    fn_tight = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), tau_prior=(2.0, 10.0)
    )
    pos_hi = jnp.concatenate([jnp.zeros(3), jnp.array([2.0])])   # tau ~ 7.4
    pos_lo = jnp.concatenate([jnp.zeros(3), jnp.array([-1.0])])  # tau ~ 0.37
    # Difference of differences: the tight prior loses more when tau grows.
    assert (fn_tight(pos_hi) - fn_tight(pos_lo)) < (fn_loose(pos_hi) - fn_loose(pos_lo))


def test_tau_prior_with_pi_prior_raises():
    import pytest

    T, Z = _make_data(jr.PRNGKey(13), p=3)
    with pytest.raises(NotImplementedError):
        make_dirichlet_logposterior(
            T, Z, sig=0.1, alpha=jnp.ones(3),
            pi_prior=(2.0, 2.0), tau_prior=(2.0, 2.0),
        )


# --- slope_grid: calibration slope integrated on a grid ---

def test_make_slope_grid_log_spaced():
    from bii.priors import make_slope_grid

    grid = make_slope_grid(0.01, 10.0, num=31)
    assert grid.shape == (31,)
    assert jnp.allclose(grid[0], 0.01, rtol=1e-5)
    assert jnp.allclose(grid[-1], 10.0, rtol=1e-5)
    ratios = grid[1:] / grid[:-1]
    assert jnp.allclose(ratios, ratios[0], rtol=1e-4)


def test_slope_grid_logposterior_matches_logsumexp():
    """The slope is integrated inside the kappa tempering, with uniform weights."""
    from jax.scipy.special import logsumexp

    from bii.inference import loglik_w_slope_grid
    from bii.priors import make_slope_grid

    T, Z = _make_data(jr.PRNGKey(20), p=4)
    grid = make_slope_grid(num=7)
    kappa = 0.5
    logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(4),
                                             kappa=kappa, slope_grid=grid)
    theta = jnp.array([0.3, -0.2, 0.1, 0.0])
    w = jax.nn.softmax(theta)
    ll = loglik_w_slope_grid(w, T, Z, 0.1, grid)
    expected = (logsumexp(kappa * ll) - jnp.log(7.0) + jnp.sum(jnp.log(w + 1e-12))
                - 0.5 * jnp.sum(theta) ** 2 / 4)
    assert jnp.allclose(logprob_fn(theta), expected, rtol=1e-5)


def test_slope_grid_position_is_theta_and_gradient_finite():
    from bii.priors import make_slope_grid

    T, Z = _make_data(jr.PRNGKey(21), p=3)
    logprob_fn = make_dirichlet_logposterior(
        T, Z, sig=0.1, alpha=jnp.ones(3), slope_grid=make_slope_grid(num=11)
    )
    lp = logprob_fn(jnp.zeros(3))
    g = jax.grad(logprob_fn)(jnp.array([0.2, -0.1, 0.0]))
    assert lp.shape == ()
    assert jnp.isfinite(lp)
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))


def test_slope_grid_with_pi_or_tau_prior_raises():
    import pytest

    T, Z = _make_data(jr.PRNGKey(22), p=3)
    grid = jnp.array([0.5, 1.0])
    for extra in ({"pi_prior": (2.0, 2.0)}, {"tau_prior": (2.0, 2.0)}):
        with pytest.raises(NotImplementedError):
            make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(3),
                                        slope_grid=grid, **extra)


def test_slope_grid_rejects_nonpositive():
    import pytest

    T, Z = _make_data(jr.PRNGKey(23), p=3)
    with pytest.raises(ValueError):
        make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(3),
                                    slope_grid=jnp.array([0.0, 1.0]))


def test_logposterior_is_proper_along_the_sum_of_theta():
    """softmax ignores a constant added to every theta; the posterior does not.

    Shifting theta by c along (1, ..., 1) leaves w, the likelihood and the
    Dirichlet term unchanged, so the log posterior drops by exactly
    (p c)^2 / (2 p) from a point where sum(theta) = 0.
    """
    T, Z = _make_data(jr.PRNGKey(22), p=3)
    logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=jnp.ones(3))
    theta = jnp.array([0.2, -0.1, -0.1])
    drop = logprob_fn(theta) - logprob_fn(theta + 2.0)
    assert jnp.allclose(drop, 0.5 * 6.0**2 / 3, rtol=1e-4)


def test_prior_only_draws_follow_the_dirichlet():
    """Known answer: with kappa = 0, NUTS draws of w follow Dirichlet(alpha).

    The term on sum(theta) must leave the distribution of w unchanged:
    E[w] = alpha / sum(alpha) = (1/6, 2/6, 3/6).
    """
    from bii.sampling import run_nuts

    T, Z = _make_data(jr.PRNGKey(23), p=3)
    alpha = jnp.array([1.0, 2.0, 3.0])
    logprob_fn = make_dirichlet_logposterior(T, Z, sig=0.1, alpha=alpha, kappa=0.0)
    raw, _ = run_nuts(jr.PRNGKey(24), logprob_fn, jnp.zeros(3),
                      num_samples=1000, num_warmup=500, num_chains=2)
    w = jax.nn.softmax(raw.reshape(-1, 3), axis=-1)
    assert jnp.allclose(w.mean(axis=0), alpha / 6.0, atol=0.03)
