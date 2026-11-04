import numpy as np
from jax import numpy as jnp
from jax import random

from bii import T_from_X, kappa_from_triplets, make_triplets


def test_labels_known_answer():
    # anchor at 0; candidate i at distance 1, candidate j at distance 2
    X = jnp.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
                   [[0.0, 0.0], [0.0, 2.0], [1.0, 0.0]]])
    assert np.array_equal(np.asarray(T_from_X(X)), [1.0, 0.0])


def test_make_triplets_layout():
    n_pool, m = 40, 4
    X_pool = random.normal(random.PRNGKey(0), (n_pool, 2))
    Z_pool = random.normal(random.PRNGKey(1), (n_pool, 3))
    T, X, Z, idx = make_triplets(random.PRNGKey(2), X_pool, Z_pool, m, anchor_fraction=0.5)
    idx = np.asarray(idx)

    assert T.shape == (20 * m,) and X.shape == (20 * m, 3, 2) and Z.shape == (20 * m, 3, 3)
    # m triplets per anchor, anchors never candidates, two distinct candidates
    _, counts = np.unique(idx[:, 0], return_counts=True)
    assert np.all(counts == m)
    assert not set(idx[:, 0]) & set(idx[:, 1:].ravel())
    assert np.all(idx[:, 1] != idx[:, 2])
    # the triplets are the pool's points, and the labels are the target's
    assert np.array_equal(np.asarray(Z), np.asarray(Z_pool)[idx])
    assert np.array_equal(np.asarray(T), np.asarray(T_from_X(X)))
    # kappa = 1/m when every anchor has m triplets
    assert kappa_from_triplets(idx) == 1.0 / m
