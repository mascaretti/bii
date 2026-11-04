import numpy as np
import pytest
from jax import numpy as jnp
from jax import random
from scipy.stats import norm

from bii import (
    delta_V_one_triplet,
    loglik_w,
    loglik_w_slope_grid,
    make_slope_grid,
    make_triplets,
    slope_posterior,
    slope_scale,
)

RNG = np.random.default_rng(0)


def gap_draws(zi, zj, zk, w, sig, n):
    """Monte Carlo draws of the weighted gap with Gaussian noise of std sig (p,)."""
    y = [z + sig * RNG.standard_normal((n, z.size)) for z in (zi, zj, zk)]
    return ((y[0] - y[2]) ** 2 - (y[1] - y[2]) ** 2) @ w


def test_moments_match_monte_carlo():
    # Lemma 1, known answer: per-feature noise, unequal across coordinates
    p = 4
    zi, zj, zk = RNG.standard_normal((3, p))
    w = np.array([0.4, 0.3, 0.2, 0.1])
    sig = np.array([0.2, 0.5, 0.3, 0.8])
    mu, V = delta_V_one_triplet(zi, zj, zk, w, sig**2, sig**2, sig**2)
    g = gap_draws(zi, zj, zk, w, sig, 400_000)
    assert abs(g.mean() - mu) < 4 * g.std() / np.sqrt(g.size)
    assert abs(g.var() / V - 1) < 0.02


def test_probit_approximates_the_triplet_probability():
    # Theorem 1: with many comparable weights, P(gap <= 0) is close to Phi(-mu/sqrt(V))
    p = 40
    zi, zj, zk = 0.3 * RNG.standard_normal((3, p))
    w = np.full(p, 1 / p)
    sig = np.full(p, 0.5)
    mu, V = delta_V_one_triplet(zi, zj, zk, w, sig**2, sig**2, sig**2)
    g = gap_draws(zi, zj, zk, w, sig, 50_000)
    assert abs(np.mean(g <= 0) - norm.cdf(-mu / np.sqrt(V))) < 0.02


@pytest.fixture(scope="module")
def triplets():
    X = random.normal(random.PRNGKey(0), (50, 3))
    Z = X + 0.3 * random.normal(random.PRNGKey(1), (50, 3))
    T, _, Zt, idx = make_triplets(random.PRNGKey(2), X, Z, 5, 0.5)
    return T, Zt, idx


def test_loglik_is_the_sum_of_log_probits(triplets):
    T, Z, _ = triplets
    w = jnp.array([0.5, 0.3, 0.2])
    mu, V = np.array([delta_V_one_triplet(z[1], z[2], z[0], w, 0.09, 0.09, 0.09) for z in Z]).T
    s = mu / np.sqrt(V)
    T = np.asarray(T)
    direct = np.sum(T * norm.logcdf(-s) + (1 - T) * norm.logcdf(s))
    assert np.isclose(loglik_w(w, T, Z, 0.3), direct, rtol=1e-5)


def test_per_triplet_noise_equals_shared_noise(triplets):
    T, Z, _ = triplets
    w = jnp.array([0.5, 0.3, 0.2])
    shared = loglik_w(w, T, Z, 0.3)
    assert np.isclose(loglik_w(w, T, Z, jnp.full((T.size, 3), 0.3)), shared, rtol=1e-6)
    assert np.isclose(loglik_w(w, T, Z, jnp.full((T.size, 3, 3), 0.3)), shared, rtol=1e-6)
    with pytest.raises(ValueError):
        loglik_w(w, T, Z, jnp.full((T.size, 2), 0.3))


def test_scale_invariance(triplets):
    # mu is linear in w and sqrt(V) too, so s does not depend on the scale of w
    T, Z, _ = triplets
    w = jnp.array([0.5, 0.3, 0.2])
    assert np.isclose(loglik_w(3 * w, T, Z, 0.3), loglik_w(w, T, Z, 0.3), rtol=1e-5)
    assert np.isclose(slope_scale(3 * w, Z, 0.3), slope_scale(w, Z, 0.3), rtol=1e-5)


def test_slope_grid(triplets):
    T, Z, _ = triplets
    w = jnp.array([0.5, 0.3, 0.2])
    # the calibrated slope s_bar(w) is the plain likelihood (lam = 1)
    one = loglik_w_slope_grid(w, T, Z, 0.3, jnp.array([slope_scale(w, Z, 0.3)]))
    assert np.isclose(one[0], loglik_w(w, T, Z, 0.3), rtol=1e-5)
    # a vanishing slope makes every triplet a coin flip
    flat = loglik_w_slope_grid(w, T, Z, 0.3, jnp.array([1e-8]))
    assert np.isclose(flat[0], T.size * np.log(0.5), rtol=1e-6)
    # the posterior over the grid is softmax(kappa * ll)
    grid = make_slope_grid(num=7)
    post = slope_posterior(w, T, Z, 0.3, grid, kappa=0.2)
    ll = np.asarray(loglik_w_slope_grid(w, T, Z, 0.3, grid))
    assert np.allclose(post, np.exp(0.2 * ll - np.logaddexp.reduce(0.2 * ll)), rtol=1e-4)
