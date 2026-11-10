import numpy as np
from jax import numpy as jnp
from jax import random

from bii import T_from_X, kappa_from_triplets, make_triplets, make_triplets_local


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


def test_make_triplets_local_shells():
    n_pool, m = 200, 5
    X_pool = random.normal(random.PRNGKey(0), (n_pool, 3))
    Z_pool = random.normal(random.PRNGKey(1), (n_pool, 4))
    T, X, _, idx = make_triplets_local(random.PRNGKey(2), X_pool, Z_pool, None, m,
                                       anchor_fraction=0.5)
    idx, Xp = np.asarray(idx), np.asarray(X_pool, dtype=np.float64)
    # the anchors of make_triplets for the same key, m triplets each
    _, _, _, idx_random = make_triplets(random.PRNGKey(2), X_pool, Z_pool, m, anchor_fraction=0.5)
    assert set(np.asarray(idx_random)[:, 0]) == set(idx[:, 0])
    anchors = np.unique(idx[:, 0])
    dests = np.setdiff1d(np.arange(n_pool), anchors)
    for a in anchors:
        rows = idx[idx[:, 0] == a]
        assert len(rows) == m
        d = np.sqrt(((Xp[dests] - Xp[a]) ** 2).sum(axis=1))
        sigma = np.sort(d)[m - 1]  # distance to the m-th nearest destination
        d1, d2 = (np.sqrt(((Xp[rows[:, c]] - Xp[a]) ** 2).sum(axis=1)) for c in (1, 2))
        near, far = np.minimum(d1, d2), np.maximum(d1, d2)
        assert np.all(near <= sigma + 1e-5) and np.all(far > sigma - 1e-5)
        # far within 2 sigma, unless that shell holds fewer than m destinations
        assert np.all(far <= 2 * sigma + 1e-5) or np.sum((d > sigma) & (d <= 2 * sigma)) < m
    # the labels are the target's, balanced by the random order of the candidates
    assert np.array_equal(np.asarray(T), np.asarray(T_from_X(X)))
    assert 0.3 < float(np.mean(T)) < 0.7
