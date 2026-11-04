import numpy as np
from jax import numpy as jnp

from bii import aligned_triplets, compute_ess, compute_rhat, compute_waic
from bii.inference import loglik_w_per_triplet

RNG = np.random.default_rng(0)


def test_rhat_and_ess_on_independent_draws():
    x = RNG.standard_normal((1000, 4, 2))
    assert np.all(np.abs(compute_rhat(x) - 1) < 0.01)
    assert np.all(np.abs(compute_ess(x) / 4000 - 1) < 0.25)


def test_rhat_flags_chains_that_disagree():
    x = RNG.standard_normal((1000, 4, 1)) + np.array([0.0, 0.0, 0.0, 3.0])[None, :, None]
    assert compute_rhat(x)[0] > 1.1


def test_aligned_triplets_known_answer():
    # one coordinate; candidate i at 1 from the anchor, candidate j at 3
    Z = jnp.array([[[0.0], [1.0], [3.0]]] * 4)
    T = jnp.array([1.0, 1.0, 1.0, 0.0])  # the source says label 1: three of four agree
    assert int(aligned_triplets(jnp.ones((2, 1)), T, Z, 0.1)[0]) == 3


def test_waic_known_answer():
    Z = jnp.array([[[0.0, 0.0], [1.0, 0.5], [2.0, 0.2]], [[0.0, 1.0], [0.5, 0.0], [0.2, 2.0]]])
    T = jnp.array([1.0, 0.0])
    w = jnp.array([[0.5, 0.5], [0.8, 0.2], [0.3, 0.7]])
    ll = np.stack([np.asarray(loglik_w_per_triplet(wi, T, Z, 0.3)) for wi in w])
    lppd = np.sum(np.log(np.mean(np.exp(ll), axis=0)))
    expected = -2 * (lppd - np.sum(np.var(ll, axis=0)))
    assert np.isclose(compute_waic(w, T, Z, 0.3), expected, rtol=1e-5)
