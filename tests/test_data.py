import numpy as np
from jax import numpy as jnp
from jax import random

from bii import T_from_X, kappa_from_triplets, make_triplets, make_triplets_softmax


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


def test_make_triplets_softmax():
    n_pool, m, rank = 12, 20000, 3
    X_pool = random.normal(random.PRNGKey(0), (n_pool, 2))
    Z_pool = random.normal(random.PRNGKey(1), (n_pool, 3))
    T, X, _, idx = make_triplets_softmax(random.PRNGKey(2), X_pool, Z_pool, None, m,
                                         anchor_fraction=0.2, rank=rank)
    idx, Xp = np.asarray(idx), np.asarray(X_pool, dtype=np.float64)
    # the anchors of make_triplets for the same key, m triplets each; anchors are never
    # candidates, and the two candidates differ
    _, _, _, idx_random = make_triplets(random.PRNGKey(2), X_pool, Z_pool, 4, anchor_fraction=0.2)
    anchors = np.unique(idx[:, 0])
    assert set(np.asarray(idx_random)[:, 0]) == set(anchors) and len(anchors) == 2
    assert np.all(np.unique(idx[:, 0], return_counts=True)[1] == m)
    assert not set(anchors) & set(idx[:, 1:].ravel()) and np.all(idx[:, 1] != idx[:, 2])
    dests = np.setdiff1d(np.arange(n_pool), anchors)
    for a in anchors:
        # known answer: destination l has weight p_l proportional to exp(-d^2 / lambda),
        # lambda the squared distance to the rank-th nearest; it is in a pair if drawn
        # first, or second after any other one
        d2 = ((Xp[dests] - Xp[a]) ** 2).sum(axis=1)
        p = np.exp(-d2 / np.sort(d2)[rank - 1])
        p /= p.sum()
        second = np.array([np.sum(np.delete(p, i) * p[i] / (1 - np.delete(p, i)))
                           for i in range(len(p))])
        drawn = idx[idx[:, 0] == a][:, 1:]
        freq = np.array([(drawn == d).any(axis=1).mean() for d in dests])
        assert np.allclose(freq, p + second, atol=0.015)  # about four standard errors
    # the labels are the target's, balanced by the random order of the candidates
    assert np.array_equal(np.asarray(T), np.asarray(T_from_X(X)))
    assert abs(float(np.mean(T)) - 0.5) < 0.02
