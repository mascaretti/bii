import numpy as np
import pytest
from jax import numpy as jnp
from jax import random
from jax.scipy.special import logsumexp

from bii import (
    loglik_w,
    loglik_w_slope_grid,
    make_dirichlet_logposterior,
    make_slope_grid,
    make_triplets,
)


@pytest.fixture(scope="module")
def data():
    X = random.normal(random.PRNGKey(0), (40, 3))
    Z = X + 0.3 * random.normal(random.PRNGKey(1), (40, 3))
    T, _, Zt, _ = make_triplets(random.PRNGKey(2), X, Z, 4, 0.5)
    return T, Zt


THETA = jnp.array([0.3, -0.2, 0.5])
ALPHA = jnp.array([1.0, 2.0, 0.5])


def test_logposterior_known_answer(data):
    T, Z = data
    w = np.exp(THETA) / np.exp(THETA).sum()
    expected = (0.25 * loglik_w(w, T, Z, 0.3) + np.sum(ALPHA * np.log(w))
                - 0.5 * np.sum(THETA) ** 2 / 3)
    lp = make_dirichlet_logposterior(T, Z, 0.3, ALPHA, kappa=0.25)
    assert np.isclose(lp(THETA), expected, rtol=1e-5)


def test_slope_logposterior_known_answer(data):
    T, Z = data
    grid = make_slope_grid(num=9)
    w = np.exp(THETA) / np.exp(THETA).sum()
    ll = loglik_w_slope_grid(w, T, Z, 0.3, grid)
    expected = (logsumexp(0.25 * ll) - np.log(9) + np.sum(ALPHA * np.log(w))
                - 0.5 * np.sum(THETA) ** 2 / 3)
    lp = make_dirichlet_logposterior(T, Z, 0.3, ALPHA, kappa=0.25, slope_grid=grid)
    assert np.isclose(lp(THETA), expected, rtol=1e-5)
    with pytest.raises(ValueError):
        make_dirichlet_logposterior(T, Z, 0.3, ALPHA, slope_grid=jnp.array([0.0, 1.0]))


def test_direction_ignored_by_softmax(data):
    # shifting theta along (1, ..., 1) leaves w alone: only the normal term moves
    T, Z = data
    lp = make_dirichlet_logposterior(T, Z, 0.3, ALPHA)
    c, q = 0.7, 3
    shift = -0.5 * ((THETA.sum() + c * q) ** 2 - THETA.sum() ** 2) / q
    assert np.isclose(lp(THETA + c) - lp(THETA), shift, atol=1e-4)


def test_slope_grid_is_log_spaced():
    g = np.asarray(make_slope_grid())
    assert g.size == 49 and np.isclose(g[0], 1e-3) and np.isclose(g[-1], 10.0, rtol=1e-6)
    assert np.allclose(g[1:] / g[:-1], g[1] / g[0], rtol=1e-4)
