"""Tests for triplet formation (data.py)."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bii.data import T_from_X, make_triplets


def _make_pool(key, n=100, p=3, sig=0.1):
    k1, k2 = jr.split(key)
    X = jr.normal(k1, (n, p))
    Z = X + jr.normal(k2, (n, p)) * sig
    return X, Z


def test_T_from_X_binary():
    """T_from_X should produce only 0.0 and 1.0."""
    key = jr.PRNGKey(0)
    X, Z = _make_pool(key, n=50, p=3)
    T, X_trip, Z_trip, _ = make_triplets(key, X, Z, n_triplets=10)
    assert set(np.array(T).tolist()).issubset({0.0, 1.0})


def test_T_from_X_deterministic():
    """Same input should give same output."""
    X = jr.normal(jr.PRNGKey(1), (30, 3, 4))
    T1 = T_from_X(X)
    T2 = T_from_X(X)
    assert jnp.array_equal(T1, T2)


def test_T_from_X_convention():
    """T=1 means column 1 closer to column 0 (anchor) than column 2."""
    X = jnp.array([
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],  # point 1 closer -> T=1
        [[0.0, 0.0], [10.0, 0.0], [1.0, 0.0]],  # point 2 closer -> T=0
    ])
    T = T_from_X(X)
    assert float(T[0]) == 1.0
    assert float(T[1]) == 0.0


def test_make_triplets_shapes():
    """make_triplets returns correct shapes."""
    key = jr.PRNGKey(1)
    X, Z = _make_pool(key, n=80, p=4)
    T, X_trip, Z_trip, idx = make_triplets(key, X, Z, n_triplets=30)
    n_anchors = max(1, int(80 * 0.1))
    total = n_anchors * 30
    assert T.shape == (total,)
    assert X_trip.shape == (total, 3, 4)
    assert Z_trip.shape == (total, 3, 4)
    assert idx.shape == (total, 3)


def test_make_triplets_binary_labels():
    """Labels should be binary."""
    key = jr.PRNGKey(2)
    X, Z = _make_pool(key, n=50, p=3)
    T, _, _, _ = make_triplets(key, X, Z, n_triplets=20)
    assert set(np.array(T).tolist()).issubset({0.0, 1.0})


def test_make_triplets_returns_X():
    """make_triplets now returns clean triplets X as well."""
    key = jr.PRNGKey(3)
    X_pool, Z_pool = _make_pool(key, n=60, p=3)
    T, X_trip, Z_trip, indices = make_triplets(key, X_pool, Z_pool, n_triplets=10)
    # X_trip should be from X_pool via indices
    assert X_trip.shape[1] == 3
    assert X_trip.shape[2] == 3


def test_make_triplets_indices_valid():
    """All indices should be valid pool indices."""
    key = jr.PRNGKey(4)
    n = 100
    X, Z = _make_pool(key, n=n, p=3)
    _, _, _, idx = make_triplets(key, X, Z, n_triplets=20)
    assert jnp.all(idx >= 0)
    assert jnp.all(idx < n)
